from urllib.parse import urlparse
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw
import json
import os
from datetime import datetime

class CCTVRegionSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("CCTV Region Selection Tool")
        self.root.geometry("1400x900")
        
        # Variables
        self.image = None
        self.photo = None
        self.canvas = None
        self.regions = {}
        self.current_region = []
        self.drawing = False
        self.region_counter = 1
        self.draw_mode = tk.StringVar(value="polygon")  # polygon, rectangle, line
        
        # Rectangle drawing variables
        self.rect_start = None
        self.temp_rect = None
        self.tempurl=None
        
        # Colors for regions
        self.colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Camera controls
        camera_frame = ttk.LabelFrame(control_frame, text="Camera Controls", padding=5)
        camera_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(camera_frame, text="Capture from Webcam", 
                  command=self.capture_from_webcam).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(camera_frame, text="Load Image File", 
                  command=self.load_image_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(camera_frame, text="Capture from IP Camera", 
                  command=self.capture_from_ip_camera).pack(side=tk.LEFT)
        
        # Drawing mode controls
        mode_frame = ttk.LabelFrame(control_frame, text="Drawing Mode", padding=5)
        mode_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(mode_frame, text="Polygon", variable=self.draw_mode, 
                       value="polygon").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_frame, text="Rectangle", variable=self.draw_mode, 
                       value="rectangle").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_frame, text="Line", variable=self.draw_mode, 
                       value="line").pack(side=tk.LEFT, padx=2)
        
        # Region controls
        region_frame = ttk.LabelFrame(control_frame, text="Region Controls", padding=5)
        region_frame.pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(region_frame, text="Clear All", 
                  command=self.clear_regions).pack(side=tk.LEFT, padx=2)
        ttk.Button(region_frame, text="Save Regions", 
                  command=self.save_regions).pack(side=tk.LEFT, padx=2)
        ttk.Button(region_frame, text="Load Regions", 
                  command=self.load_regions).pack(side=tk.LEFT, padx=2)
        ttk.Button(region_frame, text="Edit Region", 
                  command=self.edit_region).pack(side=tk.LEFT, padx=2)
        
        # Main content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas frame (left side)
        canvas_frame = ttk.Frame(content_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg='white', cursor='crosshair')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind canvas events
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<B1-Motion>', self.on_canvas_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_canvas_release)
        self.canvas.bind('<Button-3>', self.finish_polygon)  # Right click to finish polygon
        
        # Region list frame (right side)
        list_frame = ttk.LabelFrame(content_frame, text="Regions", padding=5)
        list_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Region listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.region_listbox = tk.Listbox(list_container, width=25, height=15)
        list_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.region_listbox.yview)
        self.region_listbox.configure(yscrollcommand=list_scrollbar.set)
        
        self.region_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Region details
        details_frame = ttk.Frame(list_frame)
        details_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(details_frame, text="Delete Selected", 
                  command=self.delete_selected_region).pack(fill=tk.X, pady=2)
        ttk.Button(details_frame, text="Rename Selected", 
                  command=self.rename_selected_region).pack(fill=tk.X, pady=2)
        
        # Bind listbox selection
        self.region_listbox.bind('<<ListboxSelect>>', self.on_region_select)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load an image to start selecting regions")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
        # Instructions
        instructions = """
Instructions:
• Polygon Mode: Click to add points, right-click to finish
• Rectangle Mode: Click and drag to draw rectangle
• Line Mode: Click and drag to draw line
• Regions are automatically named or you can specify custom name/ID
        """
        ttk.Label(main_frame, text=instructions, justify=tk.LEFT, 
                 foreground='gray').pack(anchor=tk.W, pady=(5, 0))
    
    def capture_from_webcam(self):
        """Capture a frame from the default webcam"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.load_image_from_array(frame_rgb)
                self.status_var.set("Webcam frame captured successfully")
            else:
                messagebox.showerror("Error", "Could not capture frame from webcam")
                
        except Exception as e:
            messagebox.showerror("Error", f"Webcam capture failed: {str(e)}")
    
    def capture_from_ip_camera(self):
        """Capture from IP camera with user-provided URL"""
        url = simpledialog.askstring("IP Camera URL", 
                                    "Enter IP camera URL (e.g., http://192.168.1.100:8080/video):")
        self.tempurl=urlparse(url).hostname
        if not url:
            return
            
        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not connect to IP camera")
                return
            
            ret, frame = cap.read()
            frame=cv2.resize(frame,(1000,1000))
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.load_image_from_array(frame_rgb)
                self.status_var.set("IP camera frame captured successfully")
            else:
                messagebox.showerror("Error", "Could not capture frame from IP camera")
                
        except Exception as e:
            messagebox.showerror("Error", f"IP camera capture failed: {str(e)}")
    
    def load_image_file(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
        )
        
        if file_path:
            try:
                image = Image.open(file_path)
                self.load_image_from_pil(image)
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def load_image_from_array(self, image_array):
        """Load image from numpy array"""
        image = Image.fromarray(image_array)
        self.load_image_from_pil(image)
    
    def load_image_from_pil(self, image):
        """Load image from PIL Image object"""
        self.image = image.copy()
        
        # Resize image if too large
        max_size = 1000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, image.size[0], image.size[1]))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Clear regions
        self.regions = {}
        self.region_counter = 1
        self.update_region_list()
    
    def on_canvas_click(self, event):
        """Handle canvas click based on drawing mode"""
        if not self.image:
            return
        
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        mode = self.draw_mode.get()
        
        if mode == "polygon":
            self.add_polygon_point(x, y)
        elif mode == "rectangle":
            self.start_rectangle(x, y)
        elif mode == "line":
            self.start_line(x, y)
    
    def on_canvas_drag(self, event):
        """Handle canvas drag"""
        if not self.image or not self.drawing:
            return
        
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        mode = self.draw_mode.get()
        
        if mode == "rectangle":
            self.update_rectangle(x, y)
        elif mode == "line":
            self.update_line(x, y)
    
    def on_canvas_release(self, event):
        """Handle canvas release"""
        if not self.image:
            return
        
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        mode = self.draw_mode.get()
        
        if mode == "rectangle":
            self.finish_rectangle(x, y)
        elif mode == "line":
            self.finish_line(x, y)
    
    def add_polygon_point(self, x, y):
        """Add point to current polygon"""
        color = self.colors[(self.region_counter - 1) % len(self.colors)]
        
        if not self.drawing:
            # Start new polygon
            self.current_region = [(x, y)]
            self.drawing = True
            # Draw starting point
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill=color, tags="temp_point")
        else:
            # Add point to existing polygon
            last_x, last_y = self.current_region[-1]
            self.canvas.create_line(last_x, last_y, x, y, fill=color, width=2, tags="temp_line")
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill=color, tags="temp_point")
            self.current_region.append((x, y))
    
    def finish_polygon(self, event):
        """Finish current polygon with right click"""
        if not self.drawing or len(self.current_region) < 3:
            return
        
        color = self.colors[(self.region_counter - 1) % len(self.colors)]
        
        # Close the polygon
        first_x, first_y = self.current_region[0]
        last_x, last_y = self.current_region[-1]
        self.canvas.create_line(last_x, last_y, first_x, first_y, fill=color, width=2, tags="temp_line")
        
        # Finish the region
        self.save_current_region("polygon", color)
    
    def start_rectangle(self, x, y):
        """Start drawing rectangle"""
        self.rect_start = (x, y)
        self.drawing = True
        color = self.colors[(self.region_counter - 1) % len(self.colors)]
        self.temp_rect = self.canvas.create_rectangle(x, y, x, y, outline=color, width=2, tags="temp_rect")
    
    def update_rectangle(self, x, y):
        """Update rectangle while dragging"""
        if self.temp_rect and self.rect_start:
            start_x, start_y = self.rect_start
            self.canvas.coords(self.temp_rect, start_x, start_y, x, y)
    
    def finish_rectangle(self, x, y):
        """Finish rectangle drawing"""
        if not self.drawing or not self.rect_start:
            return
        
        start_x, start_y = self.rect_start
        self.current_region = [(start_x, start_y), (x, start_y), (x, y), (start_x, y)]
        
        color = self.colors[(self.region_counter - 1) % len(self.colors)]
        self.canvas.delete("temp_rect")
        
        self.save_current_region("rectangle", color)
    
    def start_line(self, x, y):
        """Start drawing line"""
        self.rect_start = (x, y)  # Reuse rect_start for line start
        self.drawing = True
        color = self.colors[(self.region_counter - 1) % len(self.colors)]
        self.temp_rect = self.canvas.create_line(x, y, x, y, fill=color, width=2, tags="temp_line")
    
    def update_line(self, x, y):
        """Update line while dragging"""
        if self.temp_rect and self.rect_start:
            start_x, start_y = self.rect_start
            self.canvas.coords(self.temp_rect, start_x, start_y, x, y)
    
    def finish_line(self, x, y):
        """Finish line drawing"""
        if not self.drawing or not self.rect_start:
            return
        
        start_x, start_y = self.rect_start
        self.current_region = [(start_x, start_y), (x, y)]
        
        color = self.colors[(self.region_counter - 1) % len(self.colors)]
        self.canvas.delete("temp_line")
        
        self.save_current_region("line", color)
    
    def save_current_region(self, shape_type, color):
        """Save the current region with custom name and ID"""
        if not self.current_region:
            return
        
        # Remove temporary drawings
        self.canvas.delete("temp_line")
        self.canvas.delete("temp_point")
        self.canvas.delete("temp_rect")
        
        # Get custom name and ID from user
        dialog = RegionDialog(self.root, f"r{self.region_counter}")
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            region_name = dialog.result['name']
            region_id = dialog.result['id']
            description = dialog.result['description']
            relay_ip=dialog.result['relay_ip']
            relay_number=dialog.result['relay_number']
        else:
            # User cancelled, restore default
            region_name = f"r{self.region_counter}"
            region_id = str(self.region_counter)
            description = ""
            relay_ip=str("192.168.1.200")
            relay_number='1'
        
        # Draw the final shape
        if shape_type == "polygon":
            points = []
            for point in self.current_region:
                points.extend([point[0], point[1]])
            self.canvas.create_polygon(points, outline=color, fill='', width=2, tags=f"region_{region_name}")
        elif shape_type == "rectangle":
            x1, y1 = self.current_region[0]
            x2, y2 = self.current_region[2]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=f"region_{region_name}")
        elif shape_type == "line":
            x1, y1 = self.current_region[0]
            x2, y2 = self.current_region[1]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2, tags=f"region_{region_name}")
        
        # Add region label
        center_x = sum(p[0] for p in self.current_region) / len(self.current_region)
        center_y = sum(p[1] for p in self.current_region) / len(self.current_region)
        
        self.canvas.create_text(center_x, center_y, text=region_name, 
                               fill=color, font=('Arial', 10, 'bold'), tags=f"label_{region_name}")
        
        # Save region data
        self.regions[region_name] = {
            'id': region_id,
            'name': region_name,
            'description': description,
            'relay_ip':relay_ip,
            'relay_number':relay_number,
            'points': self.current_region.copy(),
            'shape_type': shape_type,
            'color': color,
            'created': datetime.now().isoformat()
        }
        
        self.status_var.set(f"Region {region_name} (ID: {region_id}) created")
        
        # Reset for next region
        self.current_region = []
        self.drawing = False
        self.rect_start = None
        self.temp_rect = None
        self.region_counter += 1
        
        self.update_region_list()
    
    def update_region_list(self):
        """Update the region listbox"""
        self.region_listbox.delete(0, tk.END)
        for name, data in self.regions.items():
            display_text = f"{name} (ID: {data.get('id', 'N/A')})"
            if data.get('description'):
                display_text += f" - {data['description'][:20]}..."
            self.region_listbox.insert(tk.END, display_text)
    
    def on_region_select(self, event):
        """Handle region selection in listbox"""
        selection = self.region_listbox.curselection()
        if selection:
            index = selection[0]
            region_names = list(self.regions.keys())
            if index < len(region_names):
                region_name = region_names[index]
                region_data = self.regions[region_name]
                
                # Show region details in status
                details = f"Selected: {region_name} | ID: {region_data.get('id', 'N/A')} | Type: {region_data.get('shape_type', 'unknown')} | Points: {len(region_data.get('points', []))}"
                self.status_var.set(details)
    
    def delete_selected_region(self):
        """Delete selected region"""
        selection = self.region_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a region to delete")
            return
        
        index = selection[0]
        region_names = list(self.regions.keys())
        if index < len(region_names):
            region_name = region_names[index]
            
            if messagebox.askyesno("Delete Region", f"Delete region '{region_name}'?"):
                # Remove from canvas
                self.canvas.delete(f"region_{region_name}")
                self.canvas.delete(f"label_{region_name}")
                
                # Remove from data
                del self.regions[region_name]
                
                self.update_region_list()
                self.status_var.set(f"Region {region_name} deleted")
    
    def rename_selected_region(self):
        """Rename selected region"""
        selection = self.region_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a region to rename")
            return
        
        index = selection[0]
        region_names = list(self.regions.keys())
        if index < len(region_names):
            old_name = region_names[index]
            old_data = self.regions[old_name]
            
            dialog = RegionDialog(self.root, old_name, old_data.get('id', ''), old_data.get('description', ''))
            self.root.wait_window(dialog.dialog)
            
            if dialog.result:
                new_name = dialog.result['name']
                new_id = dialog.result['id']
                new_description = dialog.result['description']
                
                # Update data
                self.regions[new_name] = self.regions.pop(old_name)
                self.regions[new_name]['name'] = new_name
                self.regions[new_name]['id'] = new_id
                self.regions[new_name]['description'] = new_description
                
                # Update canvas tags and label
                for item in self.canvas.find_withtag(f"region_{old_name}"):
                    self.canvas.itemconfig(item, tags=f"region_{new_name}")
                
                label_items = self.canvas.find_withtag(f"label_{old_name}")
                for item in label_items:
                    self.canvas.itemconfig(item, text=new_name, tags=f"label_{new_name}")
                
                self.update_region_list()
                self.status_var.set(f"Region renamed to {new_name}")
    
    def edit_region(self):
        """Edit selected region (alias for rename)"""
        self.rename_selected_region()
    
    def clear_regions(self):
        """Clear all regions"""
        if messagebox.askyesno("Clear Regions", "Are you sure you want to clear all regions?"):
            # Clear canvas
            for region_name in self.regions.keys():
                self.canvas.delete(f"region_{region_name}")
                self.canvas.delete(f"label_{region_name}")
            
            self.regions = {}
            self.region_counter = 1
            self.update_region_list()
            self.status_var.set("All regions cleared")
    
    def save_regions(self):
        """Save regions to JSON file"""
        if not self.regions:
            messagebox.showwarning("No Regions", "No regions to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Regions",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                # Add metadata to the data
                data = {
                    'regions': self.regions,
                    'image_size': self.image.size if self.image else None,
                    'created': datetime.now().isoformat(),
                    'total_regions': len(self.regions),
                    'version': '2.0',
                    'ip':self.tempurl
                    
                }
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.status_var.set(f"Regions saved to {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Saved {len(self.regions)} regions to file")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not save regions: {str(e)}")
    
    def load_regions(self):
        """Load regions from JSON file"""
        file_path = filedialog.askopenfilename(
            title="Load Regions",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'regions' not in data:
                    messagebox.showerror("Error", "Invalid region file format")
                    return
                
                self.regions = data['regions']
                self.region_counter = len(self.regions) + 1
                
                # Redraw regions if image is loaded
                if self.image:
                    self.redraw_regions()
                
                self.update_region_list()
                self.status_var.set(f"Loaded {len(self.regions)} regions from {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load regions: {str(e)}")
    
    def redraw_regions(self):
        """Redraw all regions on the canvas"""
        if not self.image:
            return
        
        # Clear existing region drawings
        for region_name in self.regions.keys():
            self.canvas.delete(f"region_{region_name}")
            self.canvas.delete(f"label_{region_name}")
        
        # Draw all regions
        for region_name, region_data in self.regions.items():
            points = region_data.get('points', [])
            color = region_data.get('color', 'red')
            shape_type = region_data.get('shape_type', 'polygon')
            
            # Draw shape based on type
            if shape_type == "polygon" and len(points) > 2:
                canvas_points = []
                for point in points:
                    canvas_points.extend([point[0], point[1]])
                self.canvas.create_polygon(canvas_points, outline=color, fill='', width=2, tags=f"region_{region_name}")
            elif shape_type == "rectangle" and len(points) == 4:
                x1, y1 = points[0]
                x2, y2 = points[2]
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=f"region_{region_name}")
            elif shape_type == "line" and len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2, tags=f"region_{region_name}")
            
            # Add label
            if points:
                center_x = sum(p[0] for p in points) / len(points)
                center_y = sum(p[1] for p in points) / len(points)
                self.canvas.create_text(center_x, center_y, text=region_name, 
                                       fill=color, font=('Arial', 10, 'bold'), tags=f"label_{region_name}")


class RegionDialog:
    def __init__(self, parent, default_name="", default_id="", default_description="",defuilt_ip='',defuilt_relay=''):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Region Details")
        self.dialog.geometry("400x400")
        self.dialog.resizable(True, True)
        
        # Center the dialog
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Name field
        ttk.Label(main_frame, text="Region Name:").pack(anchor=tk.W, pady=(0, 5))
        self.name_var = tk.StringVar(value=default_name)
        ttk.Entry(main_frame, textvariable=self.name_var, width=40).pack(fill=tk.X, pady=(0, 10))
        
        # ID field
        ttk.Label(main_frame, text="Region ID:").pack(anchor=tk.W, pady=(0, 5))
        self.id_var = tk.StringVar(value=default_id)
        ttk.Entry(main_frame, textvariable=self.id_var, width=40).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(main_frame, text="Relay Ip:").pack(anchor=tk.W, pady=(0, 5))
        self.relay_ip = tk.StringVar(value=defuilt_ip)
        ttk.Entry(main_frame, textvariable=self.relay_ip,
                  width=40).pack(fill=tk.X, pady=(0, 10))
        ttk.Label(main_frame, text="Relay Number:").pack(anchor=tk.W, pady=(0, 5))
        self.relay_number = tk.StringVar(value=defuilt_ip)
        ttk.Entry(main_frame, textvariable=self.relay_number,
                  width=40).pack(fill=tk.X, pady=(0, 10))
        
        # Description field
        ttk.Label(main_frame, text="Description (optional):").pack(anchor=tk.W, pady=(0, 5))
        self.desc_text = tk.Text(main_frame, width=40, height=4)
        self.desc_text.pack(fill=tk.X, pady=(0, 15))
        if default_description:
            self.desc_text.insert(tk.END, default_description)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.RIGHT)
        
        # Focus on name field
        self.dialog.focus_set()
        
        # Bind Enter key
        self.dialog.bind('<Return>', lambda e: self.ok_clicked())
        self.dialog.bind('<Escape>', lambda e: self.cancel_clicked())
    
    def ok_clicked(self):
        name = self.name_var.get().strip()
        region_id = self.id_var.get().strip()
        description = self.desc_text.get(1.0, tk.END).strip()
        relay_ip=self.relay_ip.get().strip()
        relay_number=self.relay_number.get().strip()
        
        if not name:
            messagebox.showwarning("Invalid Input", "Region name cannot be empty")
            return
        
        if not region_id:
            messagebox.showwarning("Invalid Input", "Region ID cannot be empty")
            return
        
        self.result = {
            'name': name,
            'id': region_id,
            'description': description,
            'relay_ip':relay_ip,
            'relay_number':relay_number
        }
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.result = None
        self.dialog.destroy()


def main():
    root = tk.Tk()
    app = CCTVRegionSelector(root)
    root.mainloop()

if __name__ == "__main__":
    main()