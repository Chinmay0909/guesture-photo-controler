import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from gesture_recognizer import GestureRecognizer
from image_editor import ImageEditor

class GestureControlledApp:
    def __init__(self):
        """Initialize the gesture-controlled image editor application."""
        self.root = tk.Tk()
        self.root.title("Gesture-Controlled Image Editor")
        self.root.geometry("1600x1000")
        self.root.state('zoomed')  # Start maximized on Windows
        
        # Initialize components
        self.gesture_recognizer = GestureRecognizer()
        
        # Camera variables
        self.camera = None
        self.camera_running = False
        self.camera_thread = None
        
        # Setup main UI
        self.setup_main_ui()
        
        # Gesture detection variables
        self.last_gesture = None
        self.gesture_count = 0
        self.gesture_threshold = 3  # Need 3 consistent detections
        self.confidence_threshold = 0.7
    
    def setup_main_ui(self):
        """Set up the main user interface."""
        # Create main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights to give more space to image editor
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=4)  # Give image section 4x more space
        main_container.rowconfigure(0, weight=1)
        
        # Camera control panel with fixed width - more compact
        camera_frame = ttk.LabelFrame(main_container, text="Camera & Gesture Control", padding="8", width=350)
        camera_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=(0, 8))
        camera_frame.grid_propagate(False)  # Maintain fixed width
        
        # Camera controls
        ttk.Label(camera_frame, text="Camera Controls:", font=('Arial', 9, 'bold')).pack(pady=(0, 3))
        self.start_camera_btn = ttk.Button(camera_frame, text="Start Camera", command=self.start_camera)
        self.start_camera_btn.pack(pady=2, fill='x')
        
        self.stop_camera_btn = ttk.Button(camera_frame, text="Stop Camera", command=self.stop_camera, state='disabled')
        self.stop_camera_btn.pack(pady=2, fill='x')
        
        ttk.Separator(camera_frame, orient='horizontal').pack(pady=5, fill='x')
        
        # Model training
        ttk.Label(camera_frame, text="Model Management:", font=('Arial', 9, 'bold')).pack(pady=(0, 3))
        ttk.Button(camera_frame, text="Train Model", command=self.train_model).pack(pady=2, fill='x')
        
        ttk.Separator(camera_frame, orient='horizontal').pack(pady=5, fill='x')
        
        # Camera display
        ttk.Label(camera_frame, text="Camera Feed:", font=('Arial', 9, 'bold')).pack(pady=(0, 3))
        self.camera_label = ttk.Label(camera_frame, text="Camera not started", background='black', foreground='white')
        self.camera_label.pack(pady=3, fill='both', expand=True)
        
        # Gesture info
        ttk.Separator(camera_frame, orient='horizontal').pack(pady=5, fill='x')
        ttk.Label(camera_frame, text="Current Gesture:", font=('Arial', 9, 'bold')).pack(pady=(0, 3))
        self.gesture_info_label = ttk.Label(camera_frame, text="No gesture detected", foreground='blue')
        self.gesture_info_label.pack(pady=2)
        
        # Detection settings
        ttk.Separator(camera_frame, orient='horizontal').pack(pady=5, fill='x')
        ttk.Label(camera_frame, text="Detection Settings:", font=('Arial', 9, 'bold')).pack(pady=(0, 3))
        
        # Confidence threshold
        ttk.Label(camera_frame, text="Confidence Threshold:").pack(anchor='w')
        self.confidence_var = tk.DoubleVar(value=0.7)
        confidence_scale = ttk.Scale(camera_frame, from_=0.5, to=0.95, variable=self.confidence_var, orient='horizontal')
        confidence_scale.pack(fill='x', pady=(0, 5))
        
        # Gesture threshold
        ttk.Label(camera_frame, text="Gesture Stability:").pack(anchor='w')
        self.stability_var = tk.IntVar(value=3)
        stability_scale = ttk.Scale(camera_frame, from_=1, to=10, variable=self.stability_var, orient='horizontal')
        stability_scale.pack(fill='x', pady=(0, 5))
        
        # Image editor container
        editor_container = ttk.Frame(main_container)
        editor_container.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        editor_container.columnconfigure(0, weight=1)
        editor_container.rowconfigure(0, weight=1)
        
        # Initialize image editor
        self.image_editor = ImageEditor(editor_container)
    
    def train_model(self):
        """Train the gesture recognition model."""
        def train_thread():
            try:
                self.gesture_info_label.config(text="Training model... Please wait.")
                self.root.update()
                
                # Train the model
                self.gesture_recognizer.train_model()
                
                # Update UI
                self.root.after(0, lambda: self.gesture_info_label.config(text="Model training completed!"))
                self.root.after(3000, lambda: self.gesture_info_label.config(text="No gesture detected"))
                
            except Exception as e:
                error_msg = f"Training failed: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("Training Error", error_msg))
                self.root.after(0, lambda: self.gesture_info_label.config(text="Training failed"))
        
        # Start training in a separate thread
        training_thread = threading.Thread(target=train_thread, daemon=True)
        training_thread.start()
    
    def start_camera(self):
        """Start the camera for gesture detection."""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera.")
                return
            
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            # Update UI
            self.start_camera_btn.config(state='disabled')
            self.stop_camera_btn.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera."""
        self.camera_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Update UI
        self.start_camera_btn.config(state='normal')
        self.stop_camera_btn.config(state='disabled')
        self.camera_label.config(image='', text="Camera stopped")
        self.gesture_info_label.config(text="Camera stopped")
    
    def camera_loop(self):
        """Main camera loop for gesture detection."""
        while self.camera_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect gesture
                self.detect_and_process_gesture(frame)
                
                # Draw landmarks on frame
                frame_with_landmarks = self.gesture_recognizer.draw_landmarks(frame.copy())
                
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Resize for display
                display_size = (320, 240)
                frame_pil = frame_pil.resize(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                # Update camera display
                self.root.after(0, lambda img=frame_tk: self.update_camera_display(img))
                
                # Control frame rate
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"Camera loop error: {e}")
                break
        
        # Cleanup
        if self.camera:
            self.camera.release()
    
    def update_camera_display(self, image):
        """Update the camera display label."""
        self.camera_label.config(image=image, text='')
        self.camera_label.image = image  # Keep a reference
    
    def detect_and_process_gesture(self, frame):
        """Detect gesture and process the corresponding action."""
        try:
            # Get current thresholds
            confidence_threshold = self.confidence_var.get()
            stability_threshold = int(self.stability_var.get())
            
            # Predict gesture
            gesture, confidence = self.gesture_recognizer.predict_gesture(frame)
            
            if gesture and confidence >= confidence_threshold:
                # Check for gesture stability
                if gesture == self.last_gesture:
                    self.gesture_count += 1
                else:
                    self.last_gesture = gesture
                    self.gesture_count = 1
                
                # Get corresponding action
                action = self.gesture_recognizer.get_action_for_gesture(gesture)
                
                # Update gesture info
                info_text = f"{gesture} ({confidence:.2f}) -> {action.upper()}"
                if self.gesture_count < stability_threshold:
                    info_text += f" [{self.gesture_count}/{stability_threshold}]"
                
                self.root.after(0, lambda: self.gesture_info_label.config(text=info_text))
                
                # Perform action if stable enough
                if self.gesture_count >= stability_threshold and action != 'unknown':
                    self.root.after(0, lambda: self.image_editor.perform_action(action, from_gesture=True))
                    self.root.after(0, lambda: self.image_editor.update_gesture_status(gesture, confidence, action))
                    
                    # Reset count after performing action
                    self.gesture_count = 0
                    self.last_gesture = None
            
            else:
                # No reliable gesture detected
                if self.gesture_count > 0:
                    self.gesture_count = max(0, self.gesture_count - 1)
                
                if self.gesture_count == 0:
                    self.last_gesture = None
                    self.root.after(0, lambda: self.gesture_info_label.config(text="No gesture detected"))
        
        except Exception as e:
            print(f"Gesture detection error: {e}")
    
    def run(self):
        """Run the application."""
        # Check if model exists
        if not self.gesture_recognizer.model or not self.gesture_recognizer.label_encoder:
            response = messagebox.askyesno(
                "Model Not Found", 
                "Gesture recognition model not found. Would you like to train it now?\n\n"
                "This will use the images in the 'dataset' folder and may take a few minutes."
            )
            if response:
                self.train_model()
        
        # Start the application
        try:
            self.root.mainloop()
        finally:
            # Cleanup
            self.stop_camera()

def main():
    """Main function to run the application."""
    try:
        app = GestureControlledApp()
        app.run()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Application Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
