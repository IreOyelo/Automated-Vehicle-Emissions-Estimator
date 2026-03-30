import customtkinter as ctk
from tkinter import messagebox
import cv2
import yaml
import numpy as np
import os
from PIL import Image, ImageTk

class CalibrationWizard:
    def __init__(self, root, video_path, output_path):
        self.root = root
        self.root.title("Unified Calibration Wizard (Layered)")
        
        try: self.root.state('zoomed')
        except: self.root.attributes('-fullscreen', True)
        
        self.root.lift()
        self.root.focus_force()
        self.root.grab_set()
        
        self.video_path = video_path
        self.output_path = output_path.replace('_linear.yaml', '_unified.yaml').replace('_homography.yaml', '_unified.yaml').replace('_hybrid.yaml', '_unified.yaml')
        
        # Data
        self.roi_points = []       
        self.lines = []            
        self.grid_points = []      
        self.H_mat = None
        
        # Interaction
        self.active_tool = "roi"   
        self.temp_points = []      
        
        # Viewport
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.last_mouse = (0, 0)
        self.mouse_img_pos = (0, 0)
        
        # UI Vars
        self.real_w_var = ctk.StringVar(value="7.0") 
        self.real_h_var = ctk.StringVar(value="20.0")
        self.d1_var = ctk.StringVar(value="17.0")
        self.d2_var = ctk.StringVar(value="24.4")
        self.el1_var = ctk.StringVar(value="0.0")
        self.el2_var = ctk.StringVar(value="0.0")
        
        # Video
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.base_img = None
        
        self.load_frame(0)
        self.setup_ui()
        self.load_defaults()
        self.draw()

    def load_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.base_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.img_h, self.img_w = self.base_img.shape[:2]
            if self.zoom == 1.0:
                self.zoom = min(1100/self.img_w, 750/self.img_h)

    def load_defaults(self):
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)
                    if 'lines' in data: self.lines = data['lines']
                    if 'roi' in data: self.roi_points = data['roi']
                    if 'grid_points' in data: self.grid_points = data['grid_points']
                    if 'homography_matrix' in data: self.H_mat = np.array(data['homography_matrix'])
                    
                    if 'real_dims' in data:
                        self.real_w_var.set(str(data['real_dims'][0]))
                        self.real_h_var.set(str(data['real_dims'][1]))
                    if 'd1' in data: self.d1_var.set(str(data['d1']))
                    if 'd2' in data: self.d2_var.set(str(data['d2']))
                    if 'elev1' in data: self.el1_var.set(str(data['elev1']))
                    if 'elev2' in data: self.el2_var.set(str(data['elev2']))
            except: pass

    def setup_ui(self):
        main = ctk.CTkFrame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)
        
        tools = ctk.CTkFrame(main, width=300)
        tools.pack(side="right", fill="y", padx=5, pady=5)
        
        ctk.CTkLabel(tools, text="SCENE SETUP", font=("Arial", 20, "bold")).pack(pady=15)
        
        # 1. ROI
        ctk.CTkLabel(tools, text="1. Detection Zone", text_color="lightgreen", font=("Arial", 14, "bold")).pack(pady=(5,2))
        ctk.CTkButton(tools, text="Draw ROI (Green)", fg_color="green", text_color="white", command=lambda: self.set_tool("roi")).pack(fill="x", padx=10, pady=2)
        
        # 2. LINES
        ctk.CTkLabel(tools, text="2. Linear Lines", text_color="orange", font=("Arial", 14, "bold")).pack(pady=(15,2))
        ctk.CTkButton(tools, text="Draw Lines (Yellow)", fg_color="orange", text_color="black", command=lambda: self.set_tool("line")).pack(fill="x", padx=10, pady=2)
        
        # 3. GRID
        ctk.CTkLabel(tools, text="3. Homography Grid", text_color="cyan", font=("Arial", 14, "bold")).pack(pady=(15,2))
        f_dim = ctk.CTkFrame(tools, fg_color="transparent")
        f_dim.pack(fill="x", padx=10)
        ctk.CTkLabel(f_dim, text="W(m):").pack(side="left"); ctk.CTkEntry(f_dim, textvariable=self.real_w_var, width=50).pack(side="left", padx=2)
        ctk.CTkLabel(f_dim, text="L(m):").pack(side="left"); ctk.CTkEntry(f_dim, textvariable=self.real_h_var, width=50).pack(side="left", padx=2)
        
        ctk.CTkButton(tools, text="Set 4-Point Grid", fg_color="cyan", text_color="black", command=lambda: self.set_tool("grid")).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(tools, text="Compute Matrix", fg_color="#0055FF", command=self.compute_matrix).pack(fill="x", padx=10, pady=2)

        # 4. PHYSICS
        ctk.CTkLabel(tools, text="4. PHYSICS CONSTANTS", text_color="magenta", font=("Arial", 14, "bold")).pack(pady=(20,2))
        p_frame1 = ctk.CTkFrame(tools)
        p_frame1.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(p_frame1, text="L1->L2 (m):").pack(side="left", padx=2)
        ctk.CTkEntry(p_frame1, textvariable=self.d1_var, width=60).pack(side="right", padx=2)
        
        p_frame2 = ctk.CTkFrame(tools)
        p_frame2.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(p_frame2, text="L2->L3 (m):").pack(side="left", padx=2)
        ctk.CTkEntry(p_frame2, textvariable=self.d2_var, width=60).pack(side="right", padx=2)
        
        p_frame3 = ctk.CTkFrame(tools)
        p_frame3.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(p_frame3, text="Elev Entry:").pack(side="left", padx=2)
        ctk.CTkEntry(p_frame3, textvariable=self.el1_var, width=60).pack(side="right", padx=2)
        
        p_frame4 = ctk.CTkFrame(tools)
        p_frame4.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(p_frame4, text="Elev Exit:").pack(side="left", padx=2)
        ctk.CTkEntry(p_frame4, textvariable=self.el2_var, width=60).pack(side="right", padx=2)

        # Actions
        ctk.CTkLabel(tools, text="ACTIONS", font=("Arial", 14, "bold")).pack(pady=(20,5))
        ctk.CTkButton(tools, text="↩ UNDO LAST", fg_color="gray", hover_color="#555", command=self.undo_last).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(tools, text="🗑 CLEAR LAYER", fg_color="#550000", hover_color="#880000", command=self.clear_current_layer).pack(fill="x", padx=10, pady=5)

        self.lbl_status = ctk.CTkLabel(tools, text="Ready", text_color="silver")
        self.lbl_status.pack(side="bottom", pady=10)
        
        ctk.CTkButton(tools, text="SAVE & EXIT", fg_color="#00AA00", height=50, command=self.save).pack(side="bottom", fill="x", padx=10)

        # Viewport
        view_area = ctk.CTkFrame(main)
        view_area.pack(side="left", fill="both", expand=True)
        
        scrub_frame = ctk.CTkFrame(view_area, height=40)
        scrub_frame.pack(side="bottom", fill="x", padx=5, pady=5)
        self.scrubber = ctk.CTkSlider(scrub_frame, from_=0, to=self.total_frames, command=self.on_scrub)
        self.scrubber.set(0)
        self.scrubber.pack(fill="x", padx=10, pady=10)
        
        self.canvas = ctk.CTkCanvas(view_area, bg="#151515", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas.bind("<Button-1>", self.on_lclick)
        self.canvas.bind("<Button-3>", self.on_rclick)
        self.canvas.bind("<Motion>", self.on_mousemove)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", self.on_zoom)
        self.canvas.bind("<Button-5>", self.on_zoom)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)

    def set_tool(self, tool):
        self.active_tool = tool
        self.temp_points = []
        self.lbl_status.configure(text=f"Active Tool: {tool.upper()}")
        self.draw()

    def undo_last(self):
        if self.temp_points:
            self.temp_points.pop()
        else:
            if self.active_tool == 'roi' and self.roi_points: self.roi_points.pop()
            elif self.active_tool == 'line' and self.lines: self.lines.pop()
            elif self.active_tool == 'grid' and self.grid_points: self.grid_points.pop()
        self.draw()

    def clear_current_layer(self):
        if self.active_tool == 'roi': self.roi_points = []
        elif self.active_tool == 'line': self.lines = []
        elif self.active_tool == 'grid': self.grid_points = []; self.H_mat = None
        self.temp_points = []
        self.draw()

    def on_scrub(self, val):
        self.load_frame(int(val))
        self.draw()

    def screen_to_img(self, sx, sy):
        ix = (sx - self.pan_x) / self.zoom
        iy = (sy - self.pan_y) / self.zoom
        return ix, iy

    def on_lclick(self, event):
        ix, iy = self.screen_to_img(event.x, event.y)
        if self.active_tool in ["roi", "line"]:
            self.temp_points.append((ix, iy))
        elif self.active_tool == "grid":
            if len(self.grid_points) < 4:
                self.grid_points.append((ix, iy))
                if len(self.grid_points) == 4:
                    self.compute_matrix()
        self.draw()

    def on_rclick(self, event):
        if self.active_tool == "roi" and len(self.temp_points) > 2:
            self.roi_points = self.temp_points[:]
            self.temp_points = []
            self.lbl_status.configure(text="ROI Saved.", text_color="green")
        elif self.active_tool == "line" and len(self.temp_points) >= 2:
            lid = len(self.lines) + 1
            self.lines.append({'id': lid, 'points': self.temp_points[:]})
            self.temp_points = []
            self.lbl_status.configure(text=f"Line {lid} Saved.", text_color="green")
        self.draw()

    def on_mousemove(self, event):
        ix, iy = self.screen_to_img(event.x, event.y)
        self.mouse_img_pos = (ix, iy)
        if self.temp_points: self.draw()

    def on_zoom(self, event):
        if event.num == 5 or event.delta < 0: factor = 0.9
        else: factor = 1.1
        new_zoom = self.zoom * factor
        if new_zoom < 0.1 or new_zoom > 10.0: return
        mx, my = event.x, event.y
        ix = (mx - self.pan_x) / self.zoom
        iy = (my - self.pan_y) / self.zoom
        self.zoom = new_zoom
        self.pan_x = mx - (ix * self.zoom)
        self.pan_y = my - (iy * self.zoom)
        self.draw()

    def start_pan(self, event):
        self.is_panning = True
        self.last_mouse = (event.x, event.y)

    def do_pan(self, event):
        if self.is_panning:
            dx = event.x - self.last_mouse[0]
            dy = event.y - self.last_mouse[1]
            self.pan_x += dx
            self.pan_y += dy
            self.last_mouse = (event.x, event.y)
            self.draw()

    def end_pan(self, event): self.is_panning = False

    def compute_matrix(self):
        if len(self.grid_points) != 4: return
        try:
            rw = float(self.real_w_var.get()); rh = float(self.real_h_var.get())
            src = np.array(self.grid_points, dtype=np.float32)
            dst = np.array([[0,0], [rw, 0], [rw, rh], [0, rh]], dtype=np.float32)
            self.H_mat, _ = cv2.findHomography(src, dst)
            self.draw()
        except: messagebox.showerror("Error", "Matrix Failed")

    def draw(self):
        if self.base_img is None: return
        h = int(self.img_h * self.zoom)
        w = int(self.img_w * self.zoom)
        if w>12000: return
        
        vis = cv2.resize(self.base_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        def to_px(pt): return (int(pt[0]*self.zoom), int(pt[1]*self.zoom))
        def to_arr(pts): return np.array([to_px(p) for p in pts], np.int32)

        if len(self.roi_points) > 1:
            pts = to_arr(self.roi_points)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            for p in pts: cv2.circle(vis, tuple(p), 4, (0, 255, 0), -1)

        for line in self.lines:
            pts = to_arr(line['points'])
            if len(pts) > 1:
                cv2.polylines(vis, [pts], False, (0, 255, 255), 2)
                cv2.putText(vis, f"L{line['id']}", tuple(pts[0]), 0, 0.6, (0, 255, 255), 2)
                for p in pts: cv2.circle(vis, tuple(p), 4, (0, 255, 255), -1)

        for i, pt in enumerate(self.grid_points):
            p = to_px(pt)
            cv2.circle(vis, p, 5, (255, 0, 0), -1)
            cv2.putText(vis, str(i+1), (p[0]+10, p[1]), 0, 0.6, (255, 0, 0), 2)

        if self.H_mat is not None:
            try:
                H_inv = np.linalg.inv(self.H_mat)
                rw = float(self.real_w_var.get()); rh = float(self.real_h_var.get())
                def proj(x, y):
                    v = np.dot(H_inv, np.array([x, y, 1])); v /= v[2]
                    return to_px((v[0], v[1]))
                for x in [0, rw/2, rw]: cv2.line(vis, proj(x,0), proj(x, rh*1.2), (255, 100, 0), 1)
                for y in np.arange(0, rh*1.2, 5.0): cv2.line(vis, proj(-2,y), proj(rw+2,y), (255, 100, 0), 1)
            except: pass

        if self.temp_points:
            t_pts = to_arr(self.temp_points)
            for p in t_pts: cv2.circle(vis, tuple(p), 3, (255, 255, 255), -1)
            if len(t_pts) > 1: cv2.polylines(vis, [t_pts], False, (255, 255, 255), 1)
            
            if self.active_tool in ['line', 'roi']:
                mouse_px = to_px(self.mouse_img_pos)
                cv2.line(vis, tuple(t_pts[-1]), mouse_px, (200, 200, 200), 1)

        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(vis))
        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, image=self.tk_img, anchor="nw")

    def save(self):
        clean_lines = []
        for l in self.lines:
            clean_lines.append({'id': l['id'], 'points': [list(p) for p in l['points']]})
            
        data = {
            'roi': [list(p) for p in self.roi_points],
            'lines': clean_lines,
            'grid_points': [list(p) for p in self.grid_points],
            'homography_matrix': self.H_mat.tolist() if self.H_mat is not None else [],
            'real_dims': [float(self.real_w_var.get()), float(self.real_h_var.get())],
            'd1': float(self.d1_var.get()),
            'd2': float(self.d2_var.get()),
            'elev1': float(self.el1_var.get()),
            'elev2': float(self.el2_var.get())
        }
        
        os.makedirs("config", exist_ok=True)
        with open(self.output_path, 'w') as f:
            yaml.safe_dump(data, f)
            
        # FIX: Redraw layers on the base image for the snapshot
        if self.base_img is not None:
            snap_vis = self.base_img.copy()
            
            if len(self.roi_points) > 1:
                cv2.polylines(snap_vis, [np.array(self.roi_points, np.int32)], True, (0, 255, 0), 2)
            
            for l in self.lines:
                pts = np.array(l['points'], np.int32)
                if len(pts) > 1:
                    cv2.polylines(snap_vis, [pts], False, (0, 255, 255), 2)
                    cv2.putText(snap_vis, f"L{l['id']}", tuple(pts[0]), 0, 1.0, (0, 255, 255), 2)
            
            if self.H_mat is not None:
                try:
                    H_inv = np.linalg.inv(self.H_mat)
                    rw = float(self.real_w_var.get())
                    rh = float(self.real_h_var.get())
                    def proj(x, y):
                        v = np.dot(H_inv, np.array([x, y, 1])); v /= v[2]
                        return (int(v[0]), int(v[1]))
                    for x in [0, rw/2, rw]: cv2.line(snap_vis, proj(x,0), proj(x, rh*1.2), (255, 100, 0), 2)
                    for y in np.arange(0, rh*1.2, 5.0): cv2.line(snap_vis, proj(-2,y), proj(rw+2, y), (255, 100, 0), 2)
                except: pass

            snap_path = self.output_path.replace('.yaml', '.jpg')
            cv2.imwrite(snap_path, cv2.cvtColor(snap_vis, cv2.COLOR_RGB2BGR))
            
        self.cap.release()
        self.root.destroy()