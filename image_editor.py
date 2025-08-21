import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from typing import Optional, Tuple
import threading
import time

class ImageEditor:
    def __init__(self, parent):
        """Initialize the image editor GUI."""
        self.parent = parent
        
        # Only set title and geometry if parent is a root window
        if hasattr(parent, 'title'):
            self.parent.title("Gesture-Controlled Image Editor")
            self.parent.geometry("1600x1000")
            self.parent.state('zoomed')  # Start maximized
        
        # Image variables
        self.current_image = None
        self.original_image = None
        self.displayed_image = None
        self.image_files = []
        self.current_image_index = 0
        
        # Image transformations
        self.zoom_factor = 1.0
        self.rotation_angle = 0
        self.brightness = 0
        self.contrast = 1.0
        
        # Action history for undo
        self.action_history = []
        self.max_history = 10
        
        # UI setup
        self.setup_ui()
        
        # Status
        self.is_paused = False
        self.last_action_time = 0
        self.action_cooldown = 1.0  # 1 second cooldown between actions
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights - give more weight to image row
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=8)  # Give image area 8x more space
        main_frame.rowconfigure(1, weight=1)  # Status bar gets minimal space
        
        # Control panel (left side) - make it more compact
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="8")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.N), padx=(0, 10))
        
        # File operations
        ttk.Button(control_frame, text="Open Image", command=self.open_image).pack(pady=2, fill='x')
        ttk.Button(control_frame, text="Open Folder", command=self.open_folder).pack(pady=2, fill='x')
        ttk.Button(control_frame, text="Save Image", command=self.save_image).pack(pady=2, fill='x')
        
        ttk.Separator(control_frame, orient='horizontal').pack(pady=5, fill='x')
        
        # Manual controls
        ttk.Label(control_frame, text="Manual Controls:", font=('Arial', 9, 'bold')).pack(pady=(0, 3))
        ttk.Button(control_frame, text="Zoom In (+)", command=lambda: self.perform_action('zoom_in')).pack(pady=1, fill='x')
        ttk.Button(control_frame, text="Zoom Out (-)", command=lambda: self.perform_action('zoom_out')).pack(pady=1, fill='x')
        ttk.Button(control_frame, text="Rotate Left", command=lambda: self.perform_action('rotate_left')).pack(pady=1, fill='x')
        ttk.Button(control_frame, text="Rotate Right", command=lambda: self.perform_action('rotate_right')).pack(pady=1, fill='x')
        ttk.Button(control_frame, text="Fit to Window", command=lambda: self.perform_action('fit_window')).pack(pady=1, fill='x')
        ttk.Button(control_frame, text="Reset", command=lambda: self.perform_action('reset')).pack(pady=1, fill='x')
        ttk.Button(control_frame, text="Previous Image", command=lambda: self.perform_action('prev_image')).pack(pady=1, fill='x')
        ttk.Button(control_frame, text="Next Image", command=lambda: self.perform_action('next_image')).pack(pady=1, fill='x')
        
        ttk.Separator(control_frame, orient='horizontal').pack(pady=5, fill='x')
        
        # Gesture mapping info
        ttk.Label(control_frame, text="Gesture Mapping:", font=('Arial', 9, 'bold')).pack(pady=(0, 3))
        gesture_info = [
            "Open Palm: Reset",
            "Closed Palm: Pause", 
            "Back Hand: Previous",
            "Point Up: Zoom In",
            "Point Down: Zoom Out",
            "Point Left: Rotate Left",
            "Point Right: Rotate Right"
        ]
        
        for info in gesture_info:
            ttk.Label(control_frame, text=info, font=('Arial', 7)).pack(anchor='w', pady=1)
        
        # Image display area with improved layout
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Display", padding="5")
        self.image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        
        # Create a frame for canvas and scrollbars
        canvas_frame = ttk.Frame(self.image_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Canvas for image display with larger default size
        self.canvas = tk.Canvas(canvas_frame, bg='white', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for canvas
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient='horizontal', command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical', command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Enable mouse wheel scrolling
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        
        # Bind canvas resize event to re-fit image
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        
        # Status bar
        self.status_frame = ttk.Frame(main_frame)
        self.status_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.status_label = ttk.Label(self.status_frame, text="Ready - Load an image to start")
        self.status_label.pack(side='left')
        
        self.gesture_label = ttk.Label(self.status_frame, text="No gesture detected", foreground='blue')
        self.gesture_label.pack(side='right')
    
    def open_image(self):
        """Open a single image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
            # Set up single image mode
            self.image_files = [file_path]
            self.current_image_index = 0
    
    def open_folder(self):
        """Open a folder containing images."""
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        
        if folder_path:
            # Get all image files in the folder
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
            self.image_files = []
            
            for file in os.listdir(folder_path):
                if file.lower().endswith(image_extensions):
                    self.image_files.append(os.path.join(folder_path, file))
            
            if self.image_files:
                self.image_files.sort()  # Sort alphabetically
                self.current_image_index = 0
                self.load_image(self.image_files[0])
            else:
                messagebox.showwarning("No Images", "No image files found in the selected folder.")
    
    def load_image(self, file_path: str):
        """Load an image from file path."""
        try:
            # Load image using OpenCV
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Reset transformations
            self.rotation_angle = 0
            self.brightness = 0
            self.contrast = 1.0
            
            # Set initial zoom to auto-fit the image
            self.zoom_factor = self.calculate_fit_zoom_factor()
            
            # Apply transformations and display
            self.apply_transformations()
            
            # Update status
            filename = os.path.basename(file_path)
            if self.image_files:
                self.status_label.config(text=f"Image {self.current_image_index + 1}/{len(self.image_files)}: {filename}")
            else:
                self.status_label.config(text=f"Loaded: {filename}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def get_canvas_size(self):
        """Get the current canvas size for auto-fitting images."""
        self.canvas.update_idletasks()  # Ensure canvas is updated
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Use minimum size if canvas hasn't been drawn yet
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 600
            
        return canvas_width, canvas_height
    
    def auto_fit_image(self, image):
        """Automatically resize image to fit within canvas while maintaining aspect ratio."""
        if image is None:
            return image
            
        canvas_width, canvas_height = self.get_canvas_size()
        
        # Add some padding so image doesn't touch edges
        max_width = canvas_width - 20
        max_height = canvas_height - 20
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Calculate scaling factor to fit within canvas
        width_ratio = max_width / img_width
        height_ratio = max_height / img_height
        scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale if image is smaller
        
        # Only resize if scaling is needed
        if scale_factor < 1.0:
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def calculate_fit_zoom_factor(self):
        """Calculate the zoom factor needed to fit the original image in the canvas."""
        if self.original_image is None:
            return 1.0
            
        canvas_width, canvas_height = self.get_canvas_size()
        img_height, img_width = self.original_image.shape[:2]
        
        # Add padding
        max_width = canvas_width - 20
        max_height = canvas_height - 20
        
        # Calculate scaling factor to fit within canvas
        width_ratio = max_width / img_width
        height_ratio = max_height / img_height
        fit_factor = min(width_ratio, height_ratio, 1.0)
        
        return fit_factor
    
    def smart_zoom_in(self):
        """Intelligent zoom in that considers canvas size."""
        if self.original_image is None:
            return
            
        # Get current fit factor for reference
        fit_factor = self.calculate_fit_zoom_factor()
        
        # Define zoom levels relative to fit size
        zoom_levels = [
            fit_factor * 0.5,   # 50% of fit size
            fit_factor * 0.75,  # 75% of fit size  
            fit_factor * 1.0,   # Fit to window
            fit_factor * 1.25,  # 125% of fit size
            fit_factor * 1.5,   # 150% of fit size
            fit_factor * 2.0,   # 200% of fit size
            fit_factor * 3.0,   # 300% of fit size
            fit_factor * 4.0,   # 400% of fit size
            fit_factor * 5.0    # 500% of fit size
        ]
        
        # Find the next higher zoom level
        current_zoom = self.zoom_factor
        next_zoom = None
        
        for zoom_level in zoom_levels:
            if zoom_level > current_zoom + 0.01:  # Small tolerance for floating point
                next_zoom = zoom_level
                break
        
        # If no higher level found, use maximum
        if next_zoom is None:
            next_zoom = zoom_levels[-1]
        
        self.zoom_factor = next_zoom
    
    def smart_zoom_out(self):
        """Intelligent zoom out that considers canvas size."""
        if self.original_image is None:
            return
            
        # Get current fit factor for reference
        fit_factor = self.calculate_fit_zoom_factor()
        
        # Define zoom levels relative to fit size
        zoom_levels = [
            fit_factor * 0.5,   # 50% of fit size
            fit_factor * 0.75,  # 75% of fit size  
            fit_factor * 1.0,   # Fit to window
            fit_factor * 1.25,  # 125% of fit size
            fit_factor * 1.5,   # 150% of fit size
            fit_factor * 2.0,   # 200% of fit size
            fit_factor * 3.0,   # 300% of fit size
            fit_factor * 4.0,   # 400% of fit size
            fit_factor * 5.0    # 500% of fit size
        ]
        
        # Find the next lower zoom level
        current_zoom = self.zoom_factor
        next_zoom = None
        
        for zoom_level in reversed(zoom_levels):
            if zoom_level < current_zoom - 0.01:  # Small tolerance for floating point
                next_zoom = zoom_level
                break
        
        # If no lower level found, use minimum
        if next_zoom is None:
            next_zoom = zoom_levels[0]
        
        self.zoom_factor = next_zoom
    
    def apply_transformations(self):
        """Apply current transformations to the image and display it."""
        if self.original_image is None:
            return
        
        # Start with original image
        image = self.original_image.copy()
        
        # Apply brightness and contrast first
        if self.brightness != 0 or self.contrast != 1.0:
            image = cv2.convertScaleAbs(image, alpha=self.contrast, beta=self.brightness)
        
        # Apply rotation
        if self.rotation_angle != 0:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Apply zoom - now always apply the zoom factor
        if self.zoom_factor != 1.0:
            height, width = image.shape[:2]
            new_height, new_width = int(height * self.zoom_factor), int(width * self.zoom_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert to PIL Image for display
        self.current_image = Image.fromarray(image)
        self.display_image()
    
    def display_image(self):
        """Display the current image on the canvas."""
        if self.current_image is None:
            return
        
        # Convert PIL image to PhotoImage
        self.displayed_image = ImageTk.PhotoImage(self.current_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        
        # Get canvas and image dimensions
        canvas_width, canvas_height = self.get_canvas_size()
        img_width, img_height = self.current_image.size
        
        # Center the image if it's smaller than canvas and at fit zoom level
        fit_factor = self.calculate_fit_zoom_factor()
        is_at_fit_level = abs(self.zoom_factor - fit_factor) < 0.01
        
        if img_width < canvas_width and img_height < canvas_height and is_at_fit_level:
            x_offset = (canvas_width - img_width) // 2
            y_offset = (canvas_height - img_height) // 2
            self.canvas.create_image(x_offset, y_offset, anchor='nw', image=self.displayed_image)
        else:
            # For larger images or zoomed images, place at top-left
            self.canvas.create_image(0, 0, anchor='nw', image=self.displayed_image)
        
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def save_image(self):
        """Save the current image."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "No image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image.save(file_path)
                messagebox.showinfo("Success", f"Image saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def can_perform_action(self) -> bool:
        """Check if enough time has passed to perform another action."""
        current_time = time.time()
        return (current_time - self.last_action_time) >= self.action_cooldown
    
    def perform_action(self, action: str, from_gesture: bool = False):
        """Perform an image transformation action."""
        if self.is_paused and from_gesture:
            return
        
        if from_gesture and not self.can_perform_action():
            return
        
        if self.original_image is None and action not in ['pause']:
            return
        
        # Save current state for undo
        if len(self.action_history) >= self.max_history:
            self.action_history.pop(0)
        
        self.action_history.append({
            'zoom_factor': self.zoom_factor,
            'rotation_angle': self.rotation_angle,
            'brightness': self.brightness,
            'contrast': self.contrast
        })
        
        # Perform action
        if action == 'zoom_in':
            self.smart_zoom_in()
        
        elif action == 'zoom_out':
            self.smart_zoom_out()
        
        elif action == 'rotate_left':
            self.rotation_angle -= 15
            self.rotation_angle %= 360
        
        elif action == 'rotate_right':
            self.rotation_angle += 15
            self.rotation_angle %= 360
        
        elif action == 'fit_window':
            # Set zoom to auto-fit factor
            self.zoom_factor = self.calculate_fit_zoom_factor()
        
        elif action == 'reset':
            self.zoom_factor = self.calculate_fit_zoom_factor()
            self.rotation_angle = 0
            self.brightness = 0
            self.contrast = 1.0
        
        elif action == 'pause':
            self.is_paused = not self.is_paused
            status = "PAUSED" if self.is_paused else "ACTIVE"
            self.gesture_label.config(text=f"Gesture Control: {status}")
            return
        
        elif action == 'prev_image':
            if len(self.image_files) > 1:
                self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
                self.load_image(self.image_files[self.current_image_index])
                return
        
        elif action == 'next_image':
            if len(self.image_files) > 1:
                self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
                self.load_image(self.image_files[self.current_image_index])
                return
        
        # Update last action time
        if from_gesture:
            self.last_action_time = time.time()
        
        # Apply transformations and refresh display
        self.apply_transformations()
    
    def update_gesture_status(self, gesture: str, confidence: float, action: str):
        """Update the gesture status display."""
        if gesture:
            status_text = f"Gesture: {gesture} ({confidence:.2f}) -> {action.upper()}"
        else:
            status_text = "No gesture detected"
        
        self.gesture_label.config(text=status_text)
    
    def get_current_image_array(self) -> Optional[np.ndarray]:
        """Get the current image as numpy array for gesture detection."""
        if self.current_image is None:
            return None
        
        # Convert PIL image back to numpy array
        return np.array(self.current_image)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling on canvas."""
        # Handle different mouse wheel events for different platforms
        if event.delta:
            delta = event.delta / 120  # Windows
        elif event.num == 4:
            delta = 1  # Linux scroll up
        elif event.num == 5:
            delta = -1  # Linux scroll down
        else:
            delta = 0
        
        # Scroll vertically by default, horizontally with Shift
        if event.state & 0x1:  # Shift key held
            self.canvas.xview_scroll(int(-delta), "units")
        else:
            self.canvas.yview_scroll(int(-delta), "units")
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize events to re-fit image if needed."""
        # Only re-fit if we have an image and zoom is at fit level
        if self.original_image is not None:
            # Check if current zoom is close to the old fit factor
            current_fit_factor = self.calculate_fit_zoom_factor()
            if abs(self.zoom_factor - current_fit_factor) < 0.05:  # Within 5% tolerance
                # Update to new fit factor and redraw
                self.zoom_factor = current_fit_factor
                # Small delay to avoid too many rapid redraws during resize
                self.parent.after(100, self.apply_transformations)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()
