import cv2
import numpy as np
import yaml
import torch
from collections import deque, Counter
from ultralytics import RTDETR, YOLO
from src.utils import get_line_intersection, get_box_center, is_point_in_poly
from src.engine_emissions import EmissionCalculator

class KalmanFilterCA:
    """Constant Acceleration Kalman Filter (2D State: [x, y, vx, vy, ax, ay])"""
    def __init__(self, dt=1/30, process_noise=0.1, measure_noise=2.0): 
        self.state = np.zeros(6) 
        self.dt = dt
        self.F = np.eye(6)
        for i in range(2):
            self.F[i, i+2] = dt
            self.F[i, i+4] = 0.5 * dt**2
            self.F[i+2, i+4] = dt
        
        self.P = np.eye(6) * 500.0
        self.Q = np.eye(6) * process_noise 
        self.R = np.eye(2) * measure_noise 
        self.H = np.zeros((2, 6))
        self.H[0,0] = 1; self.H[1,1] = 1

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[:2]

    def update(self, z):
        y = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        I = np.eye(6)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.state

    def get_speed(self): 
        return np.sqrt(self.state[2]**2 + self.state[3]**2)

    def get_accel(self):
        # State: [x, y, vx, vy, ax, ay]
        vx, vy = self.state[2], self.state[3]
        ax, ay = self.state[4], self.state[5]
        
        # Current Speed
        speed = np.sqrt(vx**2 + vy**2)
        
        if speed < 0.1: # Avoid division by zero at very low speeds
            return 0.0
            
        # Project Acceleration onto Velocity vector (Dot Product)
        # a_long = (v . a) / |v|
        dot_product = (vx * ax) + (vy * ay)
        return dot_product / speed

class VisionEngine:
    def __init__(self, model_path, config_path="config/vehicle_specs.yaml", conf=0.5):
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
        if "rtdetr" in model_path.lower():
            self.model = RTDETR(model_path)
        else:
            self.model = YOLO(model_path)
            
        self.conf = conf
        self.em_calc = EmissionCalculator(config_path)
        with open(config_path) as f: self.cfg = yaml.safe_load(f)
        
        self.tier1_factors = {
            'PC': 170.0, 'SUV': 215.0, 'LCV': 268.0, 
            'MBT': 271.0, 'HV': 938.0, 'Bus': 938.0, 'Motorcycle': 120.0
        }
        
        self.history = {}       
        self.state = {}
        self.filters = {} 
        self.frame_cnt = 0
        self.global_count = 0 
        self.stream_sum = 0.0
        self.stream_count = 0

    def get_poly_centroid(self, points):
        x = sum([p[0] for p in points]) / len(points)
        y = sum([p[1] for p in points]) / len(points)
        return (x, y)

    def process(self, frame, config, fps, preview_mode, steady_state=True, standard='euro_2'):
        results = self.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", conf=self.conf, device=self.device)
        self.frame_cnt += 1
        
        method = config.get('method', 'linear')
        roi = config.get('roi', [])
        d1_real = config.get('d1', 15.0)
        d2_real = config.get('d2', 18.0)
        grade = config.get('grade', 0.0)
        
        # Sort lines by ID
        sorted_lines = []
        dist_map = {}
        if 'lines' in config:
            sorted_lines = sorted(config['lines'], key=lambda x: x['id'])
            if len(sorted_lines) >= 1: dist_map[sorted_lines[0]['id']] = 0.0
            if len(sorted_lines) >= 2: dist_map[sorted_lines[1]['id']] = d1_real
            if len(sorted_lines) >= 3: dist_map[sorted_lines[2]['id']] = d1_real + d2_real

        H_mat = None
        H_inv = None
        if 'homography_matrix' in config and config['homography_matrix']:
             H_mat = np.array(config['homography_matrix']).reshape(3,3)
             try: H_inv = np.linalg.inv(H_mat)
             except: pass

        correction_factor = 1.0
        if method == 'hybrid' and H_mat is not None and len(sorted_lines) >= 2:
            try:
                c1 = self.get_poly_centroid(sorted_lines[0]['points'])
                c2 = self.get_poly_centroid(sorted_lines[1]['points'])
                w1 = self.project(c1, H_mat)
                w2 = self.project(c2, H_mat)
                d1_homog = np.sqrt((w2[0]-w1[0])**2 + (w2[1]-w1[1])**2)
                if d1_homog > 0.1:
                    correction_factor = d1_real / d1_homog
            except: pass

        output = []; logs = None; vis = frame.copy() if preview_mode else None
        frame_dump = [] 

        if vis is not None:
            if roi: 
                cv2.polylines(vis, [np.array(roi, np.int32)], True, (0, 200, 0), 2)
            
            # Draw Lines (Linear) or Zones (Zone Mode)
            for l in sorted_lines:
                 pts = np.array(l['points'], np.int32)
                 if method == 'zone':
                     # Visualize the "Virtual Zone" around the line
                     cv2.polylines(vis, [pts], False, (0, 255, 0), 2)
                     overlay = vis.copy()
                     cv2.polylines(overlay, [pts], False, (0, 255, 0), 20) # Thick line = Zone
                     cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
                 else:
                     cv2.polylines(vis, [pts], False, (0, 255, 255), 2)
                 
                 txt_org = (int(l['points'][0][0]), int(l['points'][0][1]))
                 lbl = f"Z{l['id']}" if method == 'zone' else f"L{l['id']}"
                 cv2.putText(vis, lbl, txt_org, 0, 0.6, (0, 255, 255), 2)
            
            if method in ['homography', 'hybrid'] and H_inv is not None and preview_mode:
                try:
                    p1 = self.project_inv((0, 0), H_inv)
                    p2 = self.project_inv((0, 50), H_inv) 
                    if p1 and p2: cv2.line(vis, p1, p2, (255, 100, 0), 1)
                except: pass

        # TRACK ACTIVE IDS
        active_ids = []
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            
            for box, tid, c_idx in zip(boxes, ids, clss):
                tid = int(tid)
                active_ids.append(tid)
                
                if method in ['homography', 'hybrid']:
                    track_point = ((box[0] + box[2]) / 2, box[3]) 
                else:
                    track_point = get_box_center(box) 
                
                if roi and not is_point_in_poly(track_point, roi): continue
                
                # INIT TRACK STATE
                if tid not in self.state:
                    self.state[tid] = {
                        'class': [], 'done': False, 'valid': False, # [FIX 1] Added 'valid' flag
                        'hits': {}, 'start_ts': self.frame_cnt,
                        'last_seen': self.frame_cnt,
                        'curr_speed': 0.0, 'curr_accel': 0.0, # [FIX 2] Added curr_accel tracking
                        'start_pos': track_point,
                        'accum_data': {
                            'dist_m': 0.0, 'vsp_sum': 0.0, 'acc_sum': 0.0, 'frames': 0,
                            # CHANGE: Add accumulators for both standards
                            'e2_co2': 0.0, 'e2_nox': 0.0, 'e2_fuel': 0.0, 
                            'e2_hc': 0.0, 'e2_co': 0.0, 'e2_pm': 0.0,
                            'e5_co2': 0.0, 'e5_nox': 0.0, 'e5_fuel': 0.0,
                            'e5_hc': 0.0, 'e5_co': 0.0, 'e5_pm': 0.0
                        }
                    }
                    if (method in ['homography', 'hybrid']) and H_mat is not None:
                        self.filters[tid] = KalmanFilterCA(dt=1.0/fps)
                        pt_world = self.project(track_point, H_mat)
                        self.filters[tid].state[0] = pt_world[0]
                        self.filters[tid].state[1] = pt_world[1]
                else:
                    self.state[tid]['last_seen'] = self.frame_cnt
                
                try: nm = self.cfg['classes'][int(c_idx)].get('short_name','PC')
                except: nm = 'PC'
                self.state[tid]['class'].append(nm)
                
                # --- MODE 1: HOMOGRAPHY / HYBRID ---
                if (method in ['homography', 'hybrid']) and H_mat is not None:
                    pt_world = self.project(track_point, H_mat)
                    kf = self.filters[tid]
                    kf.predict()
                    kf.update(pt_world)
                    
                    v_raw = kf.get_speed()
                    v_smooth = v_raw * correction_factor
                    if v_smooth * 3.6 > 140.0: v_smooth = 33.3
                    
                    a_smooth = 0.0
                    if not steady_state:
                        a_smooth = kf.get_accel() * correction_factor
                        a_smooth = max(min(a_smooth, 3.5), -5.0)
                        a_smooth *= 0.3 # Dampening factor
                    
                    self.state[tid]['curr_speed'] = v_smooth * 3.6
                    self.state[tid]['curr_accel'] = a_smooth
                    
                    dt = 1.0 / fps
                    dist_step = v_smooth * dt 
                    inst_vsp = self.em_calc.calculate_vsp(nm, v_smooth, a_smooth, grade)
                    inst_ems = self.em_calc.calculate_emissions(inst_vsp, nm, duration_sec=dt)
                    
                    # Log to CSV dump
                    frame_dump.append({
                        'frame': self.frame_cnt, 'track_id': tid, 'class': nm, 
                        'method': 'homography', 'speed_kmh': round(v_smooth*3.6, 2), 
                        'accel_mps2': round(a_smooth, 3), 'x_px': int(track_point[0]), 'y_px': int(track_point[1])
                    })
                    
                    acc = self.state[tid]['accum_data']
                    acc['dist_m'] += dist_step
                    acc['vsp_sum'] += inst_vsp
                    acc['acc_sum'] += a_smooth
                    acc['frames'] += 1

                    # DYNAMIC ACCUMULATION
                    for k, v in inst_ems.items():
                        if k in acc: acc[k] += v

                    # [FIX 3] GREEN BOX LOGIC
                    # If distance > 20m, mark as valid (Green Box)
                    if acc['dist_m'] > 20.0:
                        self.state[tid]['valid'] = True

                    # [FIX 4] REMOVED EARLY FINALIZATION
                    # The code that was here (if total_disp > 2.0: finalize) is GONE.
                    # Vehicles now track until they timeout (leave screen).

                # --- MODE 2 & 3: LINEAR TRAP OR ZONE ---
                elif method in ['linear', 'zone'] or (method == 'hybrid' and H_mat is None):
                    # For ZONE mode
                    if method == 'zone':
                        for line in sorted_lines:
                            lid = line['id']
                            pts = line['points']
                            l_x = [p[0] for p in pts]; l_y = [p[1] for p in pts]
                            min_x, max_x = min(l_x)-30, max(l_x)+30
                            min_y, max_y = min(l_y)-30, max(l_y)+30
                            px, py = track_point
                            if min_x < px < max_x and min_y < py < max_y:
                                if lid not in self.state[tid]['hits']:
                                    self.state[tid]['hits'][lid] = self.frame_cnt
                                    print(f"DEBUG: Track {tid} entered Zone {lid} at {self.frame_cnt}")

                    # For LINEAR mode
                    else: 
                        if tid not in self.history: self.history[tid] = deque(maxlen=40)
                        self.history[tid].append(track_point) 
                        if len(self.history[tid]) >= 2:
                            prev = self.history[tid][-2]
                            curr = self.history[tid][-1]
                            for line in sorted_lines:
                                lid = line['id']
                                if lid not in self.state[tid]['hits']:
                                    pts = line['points']
                                    for i in range(len(pts)-1):
                                        pt, t = get_line_intersection(prev, curr, tuple(pts[i]), tuple(pts[i+1]))
                                        if pt:
                                            self.state[tid]['hits'][lid] = (self.frame_cnt-1)+t
                                            print(f"DEBUG: Track {tid} hit Line {lid} at frame {self.frame_cnt}")
                                            break
                    
                    s = self.state[tid]
                    hits = s['hits']
                    exit_line_id = sorted_lines[-1]['id'] if sorted_lines else -1
                    hit_exit = (exit_line_id in hits)

                    # 3-LINE CASE: All 3 lines hit (full measurement with acceleration)
                    if len(hits) >= 3 and not s['done']:
                        self.calculate_linear_event(tid, fps, dist_map, grade, output, logs, frame_dump, track_point, is_timeout=False, config=config, steady_state=steady_state)
                        s['done'] = True
                    # 2-LINE CASE: Only 2 lines hit AND the vehicle has reached the exit line
                    elif len(hits) == 2 and not s['done'] and hit_exit:
                        self.calculate_linear_event(tid, fps, dist_map, grade, output, logs, frame_dump, track_point, is_timeout=False, config=config, steady_state=steady_state)
                        s['done'] = True
                
                if vis is not None:
                    s = self.state[tid]
                    # [FIX 5] Visualization Updates
                    if method == 'homography':
                        curr_spd = int(s.get('curr_speed', 0))
                        # Turn GREEN only if valid (>20m), else ORANGE
                        color = (0, 255, 0) if s.get('valid') else (0, 165, 255)
                        label = f"{tid} {nm} {curr_spd}km/h"
                    else:
                        if s.get('done'):
                            color = (0, 255, 0)
                            spd_text = s.get('final_spd', '0km/h')
                            label = f"#{s.get('final_id', '')} {nm} {spd_text}"
                        else:
                            color = (0, 165, 255)
                            curr_spd = int(s.get('curr_speed', 0))
                            label = f"{tid} {nm} {curr_spd}km/h" if curr_spd > 0 else f"{tid} {nm}"
                        
                    cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    cv2.putText(vis, label, (int(box[0]), int(box[1])-5), 0, 0.5, (255,255,255), 1)

        # GARBAGE COLLECTION
        for tid in list(self.state.keys()):
            if tid not in active_ids and not self.state[tid]['done']:
                s = self.state[tid]
                time_since_seen = (self.frame_cnt - s.get('last_seen', 0)) / fps
                
                # TIMEOUT: Vehicle has left the screen
                if time_since_seen > 1.0: 
                    # For Homography: Finalize NOW (at end of track)
                    if method in ['homography', 'hybrid']:
                        acc = s['accum_data']
                        # NO FILTER: Saving everything, even short tracks
                        
                        # [FIX 6] RECENT MAJORITY VOTE
                        # Use last 60 frames (~2 sec) for classification
                        recent_history = s['class'][-60:]
                        if not recent_history: recent_history = s['class'] # Fallback
                        recent_history = s['class'][-60:]
                        if not recent_history: recent_history = s['class'] # Fallback
                        nm = Counter(recent_history).most_common(1)[0][0]
                        
                        self.finalize_track(tid, nm, s['curr_speed'], s['curr_accel'], output, logs, method, config=config)
                        s['done'] = True
                    
                    elif method != 'homography':
                        # Timeout logic for Linear
                        hits = s['hits']
                        if len(hits) >= 2:
                            print(f"DEBUG: Track {tid} TIMEOUT FINALIZED (Hits: {len(hits)})")
                            self.calculate_linear_event(tid, fps, dist_map, grade, output, logs, frame_dump, (0,0), is_timeout=True, config=config, steady_state=steady_state)
                    
                    if time_since_seen > 5.0:
                        del self.state[tid]
                        if tid in self.history: del self.history[tid]
                        if tid in self.filters: del self.filters[tid]

        if vis is not None:
            cv2.rectangle(vis, (10, 10), (260, 90), (0, 0, 0), -1)
            cv2.rectangle(vis, (10, 10), (260, 90), (255, 255, 255), 1)
            cf_str = f" x{correction_factor:.2f}" if method == 'hybrid' else ""
            acc_str = " (a=0)" if steady_state else " (a=CALC)"
            cv2.putText(vis, f"FPS: {fps:.1f}", (20, 35), 0, 0.7, (0, 255, 0), 1)
            cv2.putText(vis, f"Count: {self.global_count}", (20, 60), 0, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, f"Mode: {method.upper()}{cf_str}{acc_str}", (20, 80), 0, 0.5, (200, 200, 200), 1)

        return vis, output, logs, frame_dump

    def calculate_linear_event(self, tid, fps, dist_map, grade, output, logs, frame_dump, track_point, is_timeout=False, config=None, steady_state=False):
        s = self.state[tid]
        hits = s['hits']
        
        v_kmh = 0.0; a_final = 0.0; v1_audit = 0.0; v2_audit = 0.0
        
        # =========================================================
        # CASE A: ALL 3 LINES HIT — Full measurement with acceleration
        # =========================================================
        if 1 in hits and 2 in hits and 3 in hits:
             t1 = hits[1]; t2 = hits[2]; t3 = hits[3]
             
             # Forward Traffic (1 -> 2 -> 3)
             if t1 < t2 < t3:
                 dt1 = (t2 - t1) / fps; dt2 = (t3 - t2) / fps
                 dist1 = abs(dist_map[2] - dist_map[1]); dist2 = abs(dist_map[3] - dist_map[2])
                 
                 if dt1 > 0.2 and dt2 > 0.2:
                     v1 = dist1 / dt1; v2 = dist2 / dt2
                     if v1 < 50.0 and v2 < 50.0:
                         v1_audit = v1; v2_audit = v2
                         a_raw = (v2 - v1) / ((dt1 + dt2) / 2.0)
                         if steady_state:
                             a_final = 0.0
                         else:
                             a_final = max(min(a_raw, 3.5), -5.0)
                         total_dist = dist1 + dist2; total_time = dt1 + dt2
                         v_kmh = (total_dist / total_time) * 3.6
                     else: return
                 else: return

             # Reverse Traffic (3 -> 2 -> 1)
             elif t3 < t2 < t1:
                 dt1 = (t2 - t3) / fps; dt2 = (t1 - t2) / fps
                 dist1 = abs(dist_map[2] - dist_map[3]); dist2 = abs(dist_map[1] - dist_map[2])
                 
                 if dt1 > 0.2 and dt2 > 0.2:
                     v1 = dist1 / dt1; v2 = dist2 / dt2
                     if v1 < 50.0 and v2 < 50.0:
                         v1_audit = v1; v2_audit = v2
                         a_raw = (v2 - v1) / ((dt1 + dt2) / 2.0)
                         if steady_state:
                             a_final = 0.0
                         else:
                             a_final = max(min(a_raw, 3.5), -5.0)
                         total_dist = dist1 + dist2; total_time = dt1 + dt2
                         v_kmh = (total_dist / total_time) * 3.6
                     else: return
                 else: return
             else:
                 return  # Timestamps not in valid order

        # =========================================================
        # CASE B: ONLY 2 LINES HIT — Single speed, zero acceleration
        # =========================================================
        elif len(hits) >= 2:
            hit_ids = sorted(hits.keys())
            id_a = hit_ids[0]; id_b = hit_ids[1]
            t_a = hits[id_a]; t_b = hits[id_b]
            
            # Ensure correct time ordering
            if t_a > t_b:
                id_a, id_b = id_b, id_a
                t_a, t_b = t_b, t_a
            
            dt = (t_b - t_a) / fps
            if id_a in dist_map and id_b in dist_map and dt > 0.2:
                dist = abs(dist_map[id_b] - dist_map[id_a])
                v1 = dist / dt
                if v1 < 50.0:
                    v1_audit = v1; v2_audit = 0.0
                    v_kmh = v1 * 3.6
                    a_final = 0.0  # Cannot compute acceleration with only 2 lines
                else: return
            else: return
        else:
            return  # Not enough lines hit
             
        # =========================================================
        # COMMON: Compute emissions for both cases
        # =========================================================
        if v_kmh > 0:
            self.global_count += 1
            
            # RECENT MAJORITY VOTE FOR LINEAR
            recent_history = s['class'][-60:]
            if not recent_history: recent_history = s['class']
            best_cls = Counter(recent_history).most_common(1)[0][0]
            
            self.stream_sum += v_kmh; self.stream_count += 1
            
            # GET STANDARD LENGTH
            roi_std_m = 100.0
            if config and 'roi_length_m' in config:
                roi_std_m = float(config['roi_length_m'])
            
            # Calculate Duration based on STANDARD Length
            total_duration = roi_std_m / (v_kmh / 3.6)
            
            # VSP and Emissions
            vsp = self.em_calc.calculate_vsp(best_cls, v_kmh/3.6, a_final, grade)
            ems = self.em_calc.calculate_emissions(vsp, best_cls, duration_sec=total_duration)
            
            # VKT is fixed to Standard Length
            vkt_km = roi_std_m / 1000.0
            
            tier1_co2 = vkt_km * self.tier1_factors.get(best_cls, 170.0)
            diff_pct = 0.0
            if tier1_co2 > 0: diff_pct = ((ems['e2_co2'] - tier1_co2) / tier1_co2) * 100.0

            row = {
                'count_id': self.global_count, 'track_id': tid, 'class': best_cls,
                'speed_kmh': round(v_kmh, 1), 'accel_avg_mps2': round(a_final, 3),
                'vel_1_mps': round(v1_audit, 2), 'vel_2_mps': round(v2_audit, 2),
                'vsp_kw_ton': round(vsp, 2), 'vkt_km': round(vkt_km, 5),
                'tier1_co2_g': round(tier1_co2, 2),
                'diff_co2_pct': round(diff_pct, 1),

                # EURO 2 COLUMNS
                'e2_co2_g': round(ems['e2_co2'], 2), 'e2_nox_g': round(ems['e2_nox'], 4),
                'e2_fuel_g': round(ems['e2_fuel'], 2),
                'e2_hc_g': round(ems['e2_hc'], 4), 'e2_co_g': round(ems['e2_co'], 4),
                'e2_pm25_g': round(ems['e2_pm'], 5),
                
                # EURO 5 COLUMNS
                'e5_co2_g': round(ems['e5_co2'], 2), 'e5_nox_g': round(ems['e5_nox'], 4),
                'e5_fuel_g': round(ems['e5_fuel'], 2),
                'e5_hc_g': round(ems['e5_hc'], 4), 'e5_co_g': round(ems['e5_co'], 4),
                'e5_pm25_g': round(ems['e5_pm'], 5)
            }
            output.append(row)
            logs = f"#{self.global_count} | {best_cls} | {row['speed_kmh']} km/h"
            
            s['done'] = True
            s['final_spd'] = f"{int(v_kmh)}km/h"
            s['final_id'] = self.global_count

            frame_dump.append({
                'frame': self.frame_cnt, 'track_id': tid, 'class': best_cls, 
                'method': 'linear_trap_event', 'speed_kmh': round(v_kmh, 2), 
                'accel_mps2': round(a_final, 3), 'x_px': int(track_point[0]), 'y_px': int(track_point[1])
            })

    def finalize_track(self, tid, nm, speed, accel, output, logs, method, config=None):
        s = self.state[tid]
        final_acc = s['accum_data']
        
        # ACTUAL tracked data (for rate calculation if needed, but we rely on totals)
        dist_actual_m = final_acc['dist_m']
        if dist_actual_m < 1.0: return # Prevent div/0
        
        # STANDARD TARGET
        roi_std_m = 100.0
        if config and 'roi_length_m' in config:
            roi_std_m = float(config['roi_length_m'])
        
        vkt_km = roi_std_m / 1000.0
        
        # SCALE FACTOR (Normalize to Standard Length)
        scale = roi_std_m / dist_actual_m

        self.global_count += 1
        best_cls = nm 
        
        self.stream_sum += speed; self.stream_count += 1
        
        tier1_co2 = vkt_km * self.tier1_factors.get(best_cls, 170.0)
        valid_frames = max(1, final_acc['frames'])
        avg_vsp = final_acc['vsp_sum'] / valid_frames
        avg_acc = final_acc['acc_sum'] / valid_frames
        
        # Calculate Scaled Emissions
        e2_co2_scaled = final_acc['e2_co2'] * scale
        
        diff_pct = 0.0
        # Using E2 CO2 (Scaled) for diff calc
        if tier1_co2 > 0: diff_pct = ((e2_co2_scaled - tier1_co2) / tier1_co2) * 100.0
        
        row = {
            'count_id': self.global_count, 'track_id': tid, 'class': best_cls,
            'speed_kmh': round(speed, 1), 'accel_avg_mps2': round(avg_acc, 3),
            'vel_1_mps': 0.0, 'vel_2_mps': 0.0, 'vsp_kw_ton': round(avg_vsp, 2),
            'vkt_km': round(vkt_km, 5), # FIXED VALUE
            'tier1_co2_g': round(tier1_co2, 2), 
            'diff_co2_pct': round(diff_pct, 1),

            # EURO 2 COLUMNS (Scaled)
            'e2_co2_g': round(e2_co2_scaled, 2), 
            'e2_nox_g': round(final_acc['e2_nox'] * scale, 4),
            'e2_fuel_g': round(final_acc['e2_fuel'] * scale, 2),
            'e2_hc_g': round(final_acc['e2_hc'] * scale, 4), 'e2_co_g': round(final_acc['e2_co'] * scale, 4),
            'e2_pm25_g': round(final_acc['e2_pm'] * scale, 5),
            
            # EURO 5 COLUMNS (Scaled)
            'e5_co2_g': round(final_acc['e5_co2'] * scale, 2), 
            'e5_nox_g': round(final_acc['e5_nox'] * scale, 4),
            'e5_fuel_g': round(final_acc['e5_fuel'] * scale, 2),
            'e5_hc_g': round(final_acc['e5_hc'] * scale, 4), 'e5_co_g': round(final_acc['e5_co'] * scale, 4),
            'e5_pm25_g': round(final_acc['e5_pm'] * scale, 5)
        }
        output.append(row)
        logs = f"#{self.global_count} | {best_cls} | {row['speed_kmh']} km/h"
        
        s['done'] = True
        s['final_spd'] = f"{int(row['speed_kmh'])}km/h"
        s['final_id'] = self.global_count

    def project(self, pt, H):
        vec = np.array([pt[0], pt[1], 1]).reshape(3, 1)
        res = np.dot(H, vec)
        if res[2] != 0: res /= res[2]
        return res[0][0], res[1][0]

    def project_inv(self, pt_world, H_inv):
        vec = np.array([pt_world[0], pt_world[1], 1]).reshape(3, 1)
        res = np.dot(H_inv, vec)
        if res[2] != 0: res /= res[2]
        if -500 < res[0][0] < 3000 and -500 < res[1][0] < 3000:
            return int(res[0][0]), int(res[1][0])
        return None