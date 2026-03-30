import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import threading
import queue
import os
import time
import datetime
import pandas as pd
import yaml
import json
import hashlib
from PIL import Image, ImageOps 
import numpy as np
import csv
import gc

# --- MATPLOTLIB SETUP ---
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

from src.gui_wizard import CalibrationWizard
from src.engine_vision import VisionEngine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

SETTINGS_FILE = "config/settings.json"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Thesis Emission Suite: V56 (Confidence & Live Physics)")
        
        try: self.state('zoomed')
        except: self.attributes('-fullscreen', True)
        
        self.processing = False
        self.paused = False
        self.stop_event = threading.Event()
        self.data_queue = queue.Queue(maxsize=10)
        self.current_calib_path = ""
        self.last_image_ref = None 
        
        self.stats = {'counts': {}, 'vsp_co2': {}, 'tier1_co2': {}, 
                      'nox': {}, 'hc': {}, 'co': {}, 'pm25': {}}
        self.traffic_metrics = {'speed_sum': 0.0, 'count': 0, 'start_time': 0}
        self.calculated_grade = 0.0
        
        self.setup_ui()
        self.load_settings() 
        
        if self.video_path:
            self.lbl_video.configure(text=f"Video: {os.path.basename(self.video_path)}")
            self.refresh_calibration_state()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.tabs.add("Configuration")
        self.tabs.add("Live Dashboard")
        
        self.setup_config_tab()
        self.setup_dashboard_tab()

    def setup_config_tab(self):
        tab = self.tabs.tab("Configuration")
        tab.grid_columnconfigure(0, weight=1); tab.grid_columnconfigure(1, weight=2)
        
        col1 = ctk.CTkFrame(tab)
        col1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(col1, text="1. VIDEO & MODEL", font=("Arial", 16, "bold"), text_color="cyan").pack(pady=10)
        self.lbl_model = ctk.CTkLabel(col1, text="Model: --")
        self.lbl_model.pack(pady=5)
        ctk.CTkButton(col1, text="Select Model", command=self.sel_model).pack(pady=5)
        
        self.lbl_video = ctk.CTkLabel(col1, text="Video: None")
        self.lbl_video.pack(pady=15)
        ctk.CTkButton(col1, text="Select Video", command=self.sel_video).pack(pady=5)
        
        # --- NEW CONFIDENCE INPUT ---
        f_conf = ctk.CTkFrame(col1, fg_color="transparent")
        f_conf.pack(pady=10)
        ctk.CTkLabel(f_conf, text="Confidence (0.0-1.0):").pack(side="left", padx=5)
        self.entry_conf = ctk.CTkEntry(f_conf, width=60)
        self.entry_conf.insert(0, "0.57")
        self.entry_conf.pack(side="left", padx=5)
        # ----------------------------
        
        # [REMOVED] Emission Standard Selector (Now Dual-Standard by default)

        ctk.CTkLabel(col1, text="2. RUN MODE", font=("Arial", 12, "bold")).pack(pady=(15, 0))
        self.calib_mode = ctk.CTkOptionMenu(col1, values=["linear", "homography", "hybrid", "zone"], command=self.on_mode_change)
        self.calib_mode.set("linear") 
        self.calib_mode.pack(pady=5)
        
        self.lbl_status = ctk.CTkLabel(col1, text="Not Calibrated", font=("Arial", 14, "bold"), text_color="red")
        self.lbl_status.pack(pady=20)
        self.lbl_mode_warn = ctk.CTkLabel(col1, text="", font=("Arial", 12, "bold"), text_color="yellow")
        self.lbl_mode_warn.pack(pady=2)
        
        ctk.CTkButton(col1, text="SETUP SCENE (Wizard)", fg_color="orange", text_color="black", command=self.launch_wizard).pack(pady=5)

        ctk.CTkLabel(col1, text="----------------").pack(pady=10)
        ctk.CTkLabel(col1, text="3. PHYSICS (Meters)", font=("Arial", 14, "bold"), text_color="cyan").pack(pady=5)
        
        self.inputs = {}
        f1 = ctk.CTkFrame(col1); f1.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(f1, text="Dist L1 (Entry) -> L2:").pack(side="left", padx=10)
        e1 = ctk.CTkEntry(f1, width=80); e1.insert(0, "17.0"); e1.pack(side="right", padx=10)
        self.inputs['d1'] = e1
        
        f2 = ctk.CTkFrame(col1); f2.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(f2, text="Dist L2 -> L3 (Exit):").pack(side="left", padx=10)
        e2 = ctk.CTkEntry(f2, width=80); e2.insert(0, "24.4"); e2.pack(side="right", padx=10)
        self.inputs['d2'] = e2

        # ROI LENGTH INPUT
        self.lbl_roi_len = ctk.CTkLabel(col1, text="Standard ROI Length (m):", anchor="w")
        self.lbl_roi_len.pack(padx=20, pady=(10, 0), fill="x")
        self.entry_roi_len = ctk.CTkEntry(col1)
        self.entry_roi_len.pack(padx=20, pady=(0, 10), fill="x")
        self.entry_roi_len.insert(0, "100.0") # Default to 100m
        
        f_grade_calc = ctk.CTkFrame(col1)
        f_grade_calc.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(f_grade_calc, text="Elev @ Entry (L1):").grid(row=0, column=0, padx=5, pady=2)
        self.elev1 = ctk.CTkEntry(f_grade_calc, width=60); self.elev1.insert(0, "0.0"); self.elev1.grid(row=0, column=1, padx=5)
        
        ctk.CTkLabel(f_grade_calc, text="Elev @ Exit (L3):").grid(row=1, column=0, padx=5, pady=2)
        self.elev2 = ctk.CTkEntry(f_grade_calc, width=60); self.elev2.insert(0, "0.0"); self.elev2.grid(row=1, column=1, padx=5)
        
        ctk.CTkButton(f_grade_calc, text="Calc Grade", width=80, fg_color="#444", command=self.calc_grade).grid(row=0, column=2, rowspan=2, padx=10)

        f3 = ctk.CTkFrame(col1); f3.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(f3, text="Final Grade (%):").pack(side="left", padx=10)
        self.grade_input = ctk.CTkEntry(f3, width=80); self.grade_input.insert(0, "3.2"); self.grade_input.pack(side="right", padx=10)

        col2 = ctk.CTkFrame(tab)
        col2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(col2, text="2. SCENE SNAPSHOT", font=("Arial", 14, "bold")).pack(pady=5)
        self.img_preview = ctk.CTkLabel(col2, text="No Snapshot", width=800, height=533, fg_color="#111")
        self.img_preview.pack(pady=10, expand=True)

    def calc_grade(self):
        try:
            e1 = float(self.elev1.get())
            e3 = float(self.elev2.get()) 
            d1 = float(self.inputs['d1'].get())
            d2 = float(self.inputs['d2'].get())
            total_dist = d1 + d2
            if total_dist == 0: return
            grade_pct = ((e3 - e1) / total_dist) * 100.0
            
            # [NEW] Store it in the class!
            self.calculated_grade = grade_pct
            
            self.grade_input.delete(0, "end")
            self.grade_input.insert(0, f"{grade_pct:.2f}")
            self.check_mode_compatibility(grade_pct)
        except ValueError:
            messagebox.showerror("Error", "Invalid numbers for Elevation or Distance")

    def check_mode_compatibility(self, grade):
        mode = self.calib_mode.get()
        if abs(grade) > 2.0 and mode in ['homography', 'hybrid']:
            self.lbl_mode_warn.configure(text=f"⚠️ High Grade ({grade:.1f}%)\nHomography invalid.\nSwitch to LINEAR.")
        else:
            self.lbl_mode_warn.configure(text="")

    def setup_dashboard_tab(self):
        tab = self.tabs.tab("Live Dashboard")
        tab.grid_columnconfigure(0, weight=3)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        self.view_panel = ctk.CTkLabel(tab, text="", fg_color="#111", width=1280, height=720)
        self.view_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        side = ctk.CTkFrame(tab)
        side.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        head_frame = ctk.CTkFrame(side, fg_color="transparent")
        head_frame.pack(fill="x", padx=5, pady=5)
        
        sys_frame = ctk.CTkFrame(head_frame, fg_color="#222")
        sys_frame.pack(side="left", fill="x", expand=True)
        self.lbl_fps_real = ctk.CTkLabel(sys_frame, text="FPS: --", font=("Arial", 11))
        self.lbl_fps_real.pack(side="left", padx=5)
        self.lbl_device = ctk.CTkLabel(sys_frame, text="DEV: --", font=("Arial", 11))
        self.lbl_device.pack(side="left", padx=5)
        
        self.theme_switch = ctk.CTkSwitch(head_frame, text="Dark", command=self.toggle_theme)
        self.theme_switch.select()
        self.theme_switch.pack(side="right", padx=5)

        metrics_frame = ctk.CTkFrame(side)
        metrics_frame.pack(fill="x", padx=5, pady=5)
        self.card_los = self.create_metric_card(metrics_frame, "LOS", "--", "#555")
        self.card_los.pack(side="left", fill="x", expand=True, padx=2)
        self.card_flow = self.create_metric_card(metrics_frame, "Flow (v/h)", "0", "#333")
        self.card_flow.pack(side="left", fill="x", expand=True, padx=2)
        self.card_speed = self.create_metric_card(metrics_frame, "Avg Spd", "0", "#333")
        self.card_speed.pack(side="left", fill="x", expand=True, padx=2)

        ctrl = ctk.CTkFrame(side)
        ctrl.pack(fill="x", pady=5)
        
        self.switch_steady = ctk.CTkSwitch(ctrl, text="Force Steady State (a=0)")
        self.switch_steady.deselect() 
        self.switch_steady.pack(side="top", pady=2)
        
        self.switch_save = ctk.CTkSwitch(ctrl, text="Save Data (CSV)")
        self.switch_save.deselect() 
        self.switch_save.pack(side="top", pady=5)

        self.switch_video = ctk.CTkSwitch(ctrl, text="Save Video (MP4)")
        self.switch_video.deselect()
        self.switch_video.pack(side="top", pady=5)
        
        btn_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        btn_row.pack(fill="x")
        
        ctk.CTkButton(btn_row, text="START (Full)", fg_color="green", width=70, 
                      command=lambda: self.run(limit_duration=False, render_gui=False)).pack(side="left", padx=2)
        
        ctk.CTkButton(btn_row, text="PREVIEW (GUI)", fg_color="#555", width=70, 
                      command=lambda: self.run(limit_duration=True, render_gui=True)).pack(side="left", padx=2)

        ctk.CTkButton(btn_row, text="EXP. DATA", fg_color="purple", width=70, 
                      command=lambda: self.run(limit_duration=True, render_gui=False)).pack(side="left", padx=2)
        
        self.btn_pause = ctk.CTkButton(side, text="PAUSE", fg_color="orange", state="disabled", command=self.toggle_pause)
        self.btn_pause.pack(fill="x", padx=5, pady=2)
        self.btn_stop = ctk.CTkButton(side, text="STOP", fg_color="red", state="disabled", command=self.stop)
        self.btn_stop.pack(fill="x", padx=5, pady=2)

        self.chart_frame = ctk.CTkFrame(side)
        self.chart_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(4, 5))
        self.fig.patch.set_facecolor('#2b2b2b')
        self.fig.subplots_adjust(hspace=0.5)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

        ctk.CTkLabel(side, text="Live Log", font=("Arial", 10)).pack(pady=(5,0))
        self.speed_list = ctk.CTkTextbox(side, height=100)
        self.speed_list.pack(fill="x", padx=5, pady=5)

    def create_metric_card(self, parent, title, value, color):
        f = ctk.CTkFrame(parent, fg_color=color)
        ctk.CTkLabel(f, text=title, font=("Arial", 10)).pack(pady=(2,0))
        lbl = ctk.CTkLabel(f, text=value, font=("Arial", 18, "bold"))
        lbl.pack(pady=(0,5))
        f.lbl_val = lbl
        return f

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.configure(text="RESUME", fg_color="cyan", text_color="black")
            self.speed_list.insert("0.0", ">> PAUSED\n")
        else:
            self.btn_pause.configure(text="PAUSE", fg_color="orange", text_color="white")
            self.speed_list.insert("0.0", ">> RESUMED\n")

    def toggle_theme(self):
        if self.theme_switch.get():
            ctk.set_appearance_mode("Dark")
            self.fig.patch.set_facecolor('#2b2b2b')
            self.update_charts_ui()
        else:
            ctk.set_appearance_mode("Light")
            self.fig.patch.set_facecolor('#f0f0f0')
            self.update_charts_ui()

    def update_metrics_ui(self, flow, speed, los):
        self.card_flow.lbl_val.configure(text=f"{int(flow)}")
        self.card_speed.lbl_val.configure(text=f"{int(speed)}")
        self.card_los.lbl_val.configure(text=los)
        colors = {'A': '#00AA00', 'B': '#66AA00', 'C': '#AAAA00', 'D': '#AA6600', 'E': '#AA0000', 'F': '#550000'}
        self.card_los.configure(fg_color=colors.get(los, "#555"))

    def update_charts_ui(self):
        bg = '#2b2b2b' if ctk.get_appearance_mode() == "Dark" else '#f0f0f0'
        fg = 'white' if ctk.get_appearance_mode() == "Dark" else 'black'
        
        self.ax1.clear(); self.ax1.set_facecolor(bg)
        counts = self.stats['counts']
        if counts:
            self.ax1.bar(list(counts.keys()), list(counts.values()), color='cyan')
        self.ax1.set_title("Vehicle Count", color=fg, fontsize=10)
        self.ax1.tick_params(colors=fg)

        self.ax2.clear(); self.ax2.set_facecolor(bg)
        vsp_c = self.stats['vsp_co2']
        tier_c = self.stats['tier1_co2']
        all_keys = list(vsp_c.keys()) + list(tier_c.keys())
        classes = sorted(list(set(all_keys)))
        if classes:
            x = np.arange(len(classes))
            w = 0.35
            self.ax2.bar(x - w/2, [vsp_c.get(c, 0) for c in classes], w, label='VSP (Dyn)', color='orange')
            self.ax2.bar(x + w/2, [tier_c.get(c, 0) for c in classes], w, label='Tier 1 (Stat)', color='dodgerblue')
            self.ax2.set_xticks(x); self.ax2.set_xticklabels(classes)
            self.ax2.legend(fontsize=8, facecolor=bg, labelcolor=fg)
        self.ax2.set_title("Total CO2 (g)", color=fg, fontsize=10)
        self.ax2.tick_params(colors=fg)
        self.chart_canvas.draw()

    def get_unified_path(self):
        if not self.video_path: return ""
        base = os.path.basename(self.video_path)
        name, ext = os.path.splitext(base)
        h = hashlib.md5(self.video_path.encode()).hexdigest()[:6]
        return f"config/calib_{name}_{h}_unified.yaml"

    def refresh_calibration_state(self):
        path = self.get_unified_path()
        if os.path.exists(path):
            with open(path, 'r') as f: data = yaml.safe_load(f)
            
            # 1. Update Status Labels
            has_lines = 'lines' in data and len(data['lines']) >= 2
            has_grid = 'homography_matrix' in data and len(data['homography_matrix']) > 0
            has_zones = has_lines 
            
            mode = self.calib_mode.get()
            ready = False
            if mode == 'linear' and has_lines: ready = True
            elif mode == 'homography' and has_grid: ready = True
            elif mode == 'hybrid' and has_lines and has_grid: ready = True
            elif mode == 'zone' and has_zones: ready = True 
            
            if ready:
                self.lbl_status.configure(text=f"Ready ({mode.upper()})", text_color="#00FF00")
            else:
                self.lbl_status.configure(text=f"Missing Setup ({mode.upper()})", text_color="orange")

            if 'd1' in data: 
                self.inputs['d1'].delete(0, "end"); self.inputs['d1'].insert(0, str(data['d1']))
            if 'd2' in data: 
                self.inputs['d2'].delete(0, "end"); self.inputs['d2'].insert(0, str(data['d2']))
            if 'elev1' in data: 
                self.elev1.delete(0, "end"); self.elev1.insert(0, str(data['elev1']))
            if 'elev2' in data: 
                self.elev2.delete(0, "end"); self.elev2.insert(0, str(data['elev2']))
                
            self.calc_grade()

        else:
            self.lbl_status.configure(text="Not Calibrated", text_color="red")
        
        self.load_preview()

    def load_preview(self):
        path = self.get_unified_path().replace('.yaml', '.jpg')
        if os.path.exists(path):
            try:
                p = Image.open(path)
                w_box, h_box = 800, 533
                p.thumbnail((w_box, h_box))
                self.img_preview.configure(image=ctk.CTkImage(p, size=p.size), text="")
            except: pass

    def on_mode_change(self, choice):
        self.refresh_calibration_state()

    def launch_wizard(self):
        if not self.video_path: messagebox.showwarning("Err", "Select Video"); return
        path = self.get_unified_path()
        root = ctk.CTkToplevel(self)
        CalibrationWizard(root, self.video_path, path)
        root.wait_window()
        self.refresh_calibration_state()

    def sel_video(self):
        p = filedialog.askopenfilename()
        if p: 
            self.video_path = p
            self.lbl_video.configure(text=f"Video: {os.path.basename(p)}")
            self.refresh_calibration_state()
            self.save_settings()

    def sel_model(self):
        p = filedialog.askopenfilename()
        if p: 
            self.model_path = p
            self.lbl_model.configure(text=f"Model: {os.path.basename(p)}")
            self.save_settings()

    def load_settings(self):
        self.video_path = ""; self.model_path = "models/yolo11n.pt"
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    d = json.load(f)
                    self.video_path = d.get("video", "")
                    self.model_path = d.get("model", "models/yolo11n.pt")
                    if self.model_path: self.lbl_model.configure(text=f"Model: {os.path.basename(self.model_path)}")
            except: pass

    def save_settings(self):
        os.makedirs("config", exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f: json.dump({"video": self.video_path, "model": self.model_path}, f)

    def stop(self): self.stop_event.set()

    def run(self, limit_duration, render_gui):
        if self.processing or not self.video_path: return
        
        path = self.get_unified_path()
        if not os.path.exists(path):
             messagebox.showerror("Error", f"No calibration found.\nRun Wizard."); return

        with open(path) as f: full_config = yaml.safe_load(f)
        
        run_mode = self.calib_mode.get()
        run_config = full_config.copy()
        run_config['method'] = run_mode
        
        has_lines = 'lines' in full_config and full_config['lines']
        has_grid = 'homography_matrix' in full_config
        
        if run_mode == 'linear' and not has_lines: messagebox.showerror("Err", "Linear mode needs Lines!"); return
        if run_mode == 'homography' and not has_grid: messagebox.showerror("Err", "Homography needs Grid!"); return
        if run_mode == 'hybrid' and (not has_lines or not has_grid): messagebox.showerror("Err", "Hybrid needs BOTH Lines and Grid!"); return
        if run_mode == 'zone' and not has_lines: messagebox.showerror("Err", "Zone mode needs Lines (to act as centers)!"); return

        try:
            run_config['d1'] = float(self.inputs['d1'].get())
            run_config['d2'] = float(self.inputs['d2'].get())
            run_config['grade'] = float(self.grade_input.get())
            
            # ROI Length
            try: roi_len = float(self.entry_roi_len.get())
            except: roi_len = 100.0
            run_config['roi_length_m'] = roi_len
        except: messagebox.showerror("Err", "Check Distances"); return
        
        # --- GET CONFIDENCE ---
        try:
            conf_val = float(self.entry_conf.get())
            if conf_val < 0.0 or conf_val > 1.0: raise ValueError
        except:
            messagebox.showerror("Err", "Confidence must be 0.0-1.0")
            return
        # ----------------------
        
        # Parse Standard
        # Standard is now Dual (Euro 2 + 5) automatically

        self.processing = True; self.paused = False; self.stop_event.clear()
        self.btn_stop.configure(state="normal")
        self.btn_pause.configure(state="normal", text="PAUSE", fg_color="orange")
        self.tabs.set("Live Dashboard")
        self.speed_list.delete("0.0", "end")
        
        self.last_image_ref = None
        try:
            self.view_panel.configure(image=None, fg_color="#111", text="")
            self.view_panel.update()
            self.speed_list.insert("0.0", f">> INITIALIZING ENGINE (Conf={conf_val})...\n")
        except Exception as e: print(f"Reset Warn: {e}")
        
        self.stats = {'counts': {}, 'vsp_co2': {}, 'tier1_co2': {}, 
                      'nox': {}, 'hc': {}, 'co': {}, 'pm25': {}}
        self.traffic_metrics = {'speed_sum': 0.0, 'count': 0, 'start_time': time.time()}
        self.update_charts_ui()
        
        with self.data_queue.mutex: self.data_queue.queue.clear()
        
        should_save = bool(self.switch_save.get())
        should_save_video = bool(self.switch_video.get())
        force_steady = bool(self.switch_steady.get())
        
        args = {
            'video': self.video_path, 'model': self.model_path, 
            'config': run_config, 
            'limit_duration': limit_duration,
            'render_gui': render_gui,
            'save_output': should_save,
            'save_video': should_save_video,
            'steady_state': force_steady,
            'conf': conf_val,
            'grade': getattr(self, 'calculated_grade', 0.0) 
        }
        
        self.thread = threading.Thread(target=self.engine_thread, args=(args,))
        self.thread.daemon = True
        self.thread.start()
        self.after(100, self.update_loop)

    def engine_thread(self, args):
        try:
            # PASS CONFIDENCE TO ENGINE
            eng = VisionEngine(args['model'], conf=args.get('conf', 0.5))
            
            cap = cv2.VideoCapture(args['video'])
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
            if total_frames < 0: total_frames = 1  # Prevent division by zero
            self.data_queue.put(('LOG', f"Processing Video: {total_frames} frames total..."))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            limit_duration = args.get('limit_duration', False)
            render_gui = args.get('render_gui', True)
            do_save = args.get('save_output', False)
            do_video = args.get('save_video', False)
            
            max_frames = int(fps * 300) if limit_duration else total_frames
            
            out_writer = None; csv_file = None; csv_writer = None
            traj_file = None; traj_writer = None; out_dir = None
            
            if do_save:
                vid_name = os.path.splitext(os.path.basename(args['video']))[0]
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                method_tag = args['config'].get('method', 'unknown').upper()
                out_dir = f"outputs/{vid_name}/Experiment_{method_tag}_{ts}"
                os.makedirs(out_dir, exist_ok=True)
                
                if do_video:
                    out_w, out_h = 1280, 720
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_writer = cv2.VideoWriter(f"{out_dir}/video.mp4", fourcc, fps, (out_w, out_h))
                
                tag = "dual_std"
                csv_path = f"{out_dir}/emissions_data_{tag}.csv"
                csv_file = open(csv_path, 'w', newline='')
                headers = [
                    'count_id', 'track_id', 'class', 
                    'speed_kmh', 'accel_avg_mps2',       
                    'vel_1_mps', 'vel_2_mps',            
                    'vsp_kw_ton',                        
                    'vkt_km', 'tier1_co2_g', 
                    'diff_co2_pct',
                    
                    # EURO 2
                    'e2_co2_g', 'e2_nox_g', 'e2_fuel_g', 'e2_hc_g', 'e2_co_g', 'e2_pm25_g',
                    
                    # EURO 5
                    'e5_co2_g', 'e5_nox_g', 'e5_fuel_g', 'e5_hc_g', 'e5_co_g', 'e5_pm25_g'
                ]
                csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
                csv_writer.writeheader()
                
                traj_file = open(f"{out_dir}/trajectory_dump.csv", 'w', newline='')
                traj_writer = csv.DictWriter(traj_file, fieldnames=[
                    'frame', 'track_id', 'class', 'method', 
                    'speed_kmh', 'accel_mps2', 'x_px', 'y_px'
                ])
                traj_writer.writeheader()
            
            cnt=0
            prev_time = time.time()
            start_time_proc = time.time()
            last_ui_update = time.time()
            gc_freq = 50 if not render_gui else 200

            while cap.isOpened() and not self.stop_event.is_set():
                if self.paused: time.sleep(0.1); continue
                if limit_duration and cnt >= max_frames:
                    self.data_queue.put(('DONE', "Experiment Duration Reached (5min)."))
                    break

                ret, frame = cap.read()
                if not ret: break
                cnt += 1
                
                time.sleep(0.002)
                if cnt % gc_freq == 0: gc.collect()
                
                vis, res, log, frame_dump = eng.process(frame, args['config'], fps, True, args.get('steady_state', True)) 
                
                if traj_writer and frame_dump:
                    for item in frame_dump: traj_writer.writerow(item)

                if res: 
                    for row in res:
                        if csv_writer: csv_writer.writerow(row)
                        cls = row['class']; spd = row['speed_kmh']
                        self.stats['counts'][cls] = self.stats['counts'].get(cls, 0) + 1
                        self.stats['vsp_co2'][cls] = self.stats['vsp_co2'].get(cls, 0) + row['e2_co2_g']
                        self.stats['tier1_co2'][cls] = self.stats['tier1_co2'].get(cls, 0) + row['tier1_co2_g']
                        self.stats['nox'][cls] = self.stats['nox'].get(cls, 0) + row['e2_nox_g']
                        self.stats['hc'][cls] = self.stats['hc'].get(cls, 0) + row['e2_hc_g']
                        self.stats['co'][cls] = self.stats['co'].get(cls, 0) + row['e2_co_g']
                        self.stats['pm25'][cls] = self.stats['pm25'].get(cls, 0) + row['e2_pm25_g']
                        self.traffic_metrics['count'] += 1
                        self.traffic_metrics['speed_sum'] += spd

                if vis is not None and out_writer:
                    try:
                        vis_out = cv2.resize(vis, (1280, 720))
                        out_writer.write(vis_out)
                        del vis_out
                    except Exception as e: print(f"Write Skip: {e}")
                
                if not render_gui: del vis
                del frame

                curr_time = time.time()
                if (curr_time - last_ui_update) > 0.5:
                     last_ui_update = curr_time
                     if render_gui:
                         proc_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                         elapsed_s = curr_time - self.traffic_metrics['start_time']
                         total_c = self.traffic_metrics['count']
                         flow = (total_c / elapsed_s) * 3600 if elapsed_s > 1 else 0
                         avg_spd = 0
                         if total_c > 0: avg_spd = self.traffic_metrics['speed_sum'] / total_c
                         los = "F"
                         if avg_spd > 80: los = "A"
                         elif avg_spd > 65: los = "B"
                         elif avg_spd > 50: los = "C"
                         elif avg_spd > 35: los = "D"
                         elif avg_spd > 20: los = "E"
                         dev_str = "GPU" if eng.device == 0 else "CPU"
                         self.data_queue.put(('METRICS', (proc_fps, dev_str, flow, avg_spd, los)))
                         self.data_queue.put(('CHART', None))

                     # [NEW] ETA Calculation - Update every 100 frames
                     if cnt % 100 == 0:
                         elapsed_time = curr_time - start_time_proc
                         if elapsed_time > 0:
                             fps_proc = cnt / elapsed_time
                             percent = (cnt / total_frames) * 100
                             remaining_frames = total_frames - cnt
                             eta_seconds = remaining_frames / fps_proc if fps_proc > 0 else 0
                             eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                             log_msg = f"Prog: {percent:.1f}% | FPS: {fps_proc:.1f} | ETA: {eta_str}"
                             self.data_queue.put(('LOG', log_msg))

                prev_time = curr_time
                if render_gui and cnt % 3 == 0 and 'vis' in locals():
                     if self.data_queue.qsize() < 2: 
                        h_vis, w_vis = vis.shape[:2]
                        scale = 1280 / w_vis
                        vis_big = cv2.resize(vis, (1280, int(h_vis*scale)))
                        self.data_queue.put(('IMG', cv2.cvtColor(vis_big, cv2.COLOR_BGR2RGB)))

                if log and (curr_time - last_ui_update) < 0.1: self.data_queue.put(('LOG', log))
            
            if do_save and out_dir:
                self.plot_detailed_pollutants(out_dir)
                self.data_queue.put(('SAVE_FINAL', (out_dir, out_dir)))
            else:
                self.data_queue.put(('DONE', "Run Finished (No Output Saved)."))

        except Exception as e: 
            self.data_queue.put(('DONE', f"Error: {e}"))
            import traceback; traceback.print_exc()
            
        finally:
            if 'cap' in locals(): cap.release()
            if out_writer: out_writer.release()
            if csv_file: csv_file.close()
            if traj_file: traj_file.close()
            gc.collect()

    def plot_detailed_pollutants(self, out_dir):
        try:
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            c_data = self.stats['counts']
            if c_data:
                ax.bar(c_data.keys(), c_data.values(), color='cyan')
                ax.set_title('Vehicle Count by Class')
                ax.set_ylabel('Count')
            fig.tight_layout()
            FigureCanvasAgg(fig).print_png(f"{out_dir}/plot_vehicle_counts.png")
            plt.close(fig); del fig 

            vsp_c = self.stats['vsp_co2']
            tier_c = self.stats['tier1_co2']
            all_keys = list(vsp_c.keys()) + list(tier_c.keys())
            classes = sorted(list(set(all_keys)))
            if classes:
                x = np.arange(len(classes))
                w = 0.35
                fig = Figure(figsize=(6, 4))
                ax = fig.add_subplot(111)
                ax.bar(x - w/2, [vsp_c.get(c, 0) for c in classes], w, label='VSP (Dyn)', color='orange')
                ax.bar(x + w/2, [tier_c.get(c, 0) for c in classes], w, label='Tier 1 (Stat)', color='dodgerblue')
                ax.set_xticks(x); ax.set_xticklabels(classes)
                ax.legend()
                ax.set_title('Total CO2 Emissions (g)')
                fig.tight_layout()
                FigureCanvasAgg(fig).print_png(f"{out_dir}/plot_co2_analysis.png")
                plt.close(fig); del fig

                fig = Figure(figsize=(6, 4))
                ax = fig.add_subplot(111)
                nox_d = self.stats['nox']; pm_d = self.stats['pm25']
                ax.bar(classes, [nox_d.get(c, 0) for c in classes], label='NOx', color='red')
                ax.bar(classes, [pm_d.get(c, 0) for c in classes], bottom=[nox_d.get(c, 0) for c in classes], label='PM2.5', color='black')
                ax.legend()
                ax.set_title('Respiratory Health (NOx + PM2.5)')
                fig.tight_layout()
                FigureCanvasAgg(fig).print_png(f"{out_dir}/plot_respiratory_health.png")
                plt.close(fig); del fig

                fig = Figure(figsize=(6, 4))
                ax = fig.add_subplot(111)
                hc_d = self.stats['hc']; co_d = self.stats['co']
                ax.bar(classes, [hc_d.get(c, 0) for c in classes], label='HC', color='green')
                ax.bar(classes, [co_d.get(c, 0) for c in classes], bottom=[hc_d.get(c, 0) for c in classes], label='CO', color='purple')
                ax.legend()
                ax.set_title('Regulated Pollutants (HC + CO)')
                fig.tight_layout()
                FigureCanvasAgg(fig).print_png(f"{out_dir}/plot_regulated_pollutants.png")
                plt.close(fig); del fig

        except Exception as e:
            print(f"Plot Error: {e}")
            plt.close('all')

    def update_loop(self):
        while True:
            try: m, c = self.data_queue.get_nowait()
            except queue.Empty: break
            
            try:
                if m=='IMG':
                    p = Image.fromarray(c)
                    w_widget = self.view_panel.winfo_width()
                    h_widget = self.view_panel.winfo_height()
                    if w_widget > 10 and h_widget > 10:
                        p = ImageOps.contain(p, (w_widget, h_widget), Image.Resampling.LANCZOS)
                    self.last_image_ref = ctk.CTkImage(p, size=p.size)
                    self.view_panel.configure(image=self.last_image_ref)
                elif m=='PROGRESS_LOG':
                    pass
                elif m=='METRICS':
                    fps, dev, flow, spd, los = c
                    self.lbl_fps_real.configure(text=f"FPS: {fps:.1f}")
                    self.lbl_device.configure(text=f"DEV: {dev}", text_color="#00FF00" if "GPU" in dev else "orange")
                    self.update_metrics_ui(flow, spd, los)
                elif m=='LOG':
                    curr = self.speed_list.get("0.0", "end")
                    if len(curr) > 2000: self.speed_list.delete("1.0", "50.0")
                    self.speed_list.insert("0.0", c + "\n")
                elif m=='CHART': self.update_charts_ui()
                elif m=='SAVE_FINAL':
                    path, out_dir = c
                    msg = f"RUN COMPLETE.\nData saved to: {out_dir}"
                    self.speed_list.insert("0.0", f"\n{msg}\n")
                    messagebox.showinfo("Done", msg)
                    self.processing=False
                    self.btn_stop.configure(state="disabled"); self.btn_pause.configure(state="disabled")
                    self.last_image_ref = None; self.view_panel.configure(image=None)
                    return
                elif m=='DONE': 
                    self.processing=False
                    self.btn_stop.configure(state="disabled"); self.btn_pause.configure(state="disabled")
                    self.speed_list.insert("0.0", f"\nSTATUS: {c}\n")
                    self.last_image_ref = None; self.view_panel.configure(image=None)
                    messagebox.showinfo("Status", c); return
            except Exception as e: print(f"UI Error: {e}")

        if self.processing: self.after(30, self.update_loop)

if __name__ == "__main__": App().mainloop()