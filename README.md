# Gesture-Controlled Image Editor

A Python application that uses hand gesture recognition to control image editing operations in real-time. The application uses MediaPipe for hand landmark detection and TensorFlow for gesture classification.

## Features

### Gesture Controls
- **Open Palm**: Reset image to original state
- **Closed Palm**: Pause/unpause gesture control
- **Back Hand**: Navigate to previous image (when folder is loaded)
- **Point Up**: Zoom in
- **Point Down**: Zoom out
- **Point Left**: Rotate image left (15° increments)
- **Point Right**: Rotate image right (15° increments)

### Image Editor Features
- Load single images or entire folders
- Zoom in/out with gesture or manual controls
- Rotate images in 15° increments
- Reset transformations
- Navigate through multiple images
- Save edited images
- Real-time gesture detection with camera feed
- Adjustable confidence and stability thresholds

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam for gesture detection
- Windows, macOS, or Linux

### Setup Instructions

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd gesture_photo_controller
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify dataset structure**
   Make sure your dataset folder has the following structure:
   ```
   dataset/
   ├── train/
   │   ├── back hand/
   │   ├── closed palm/
   │   ├── open palm/
   │   ├── point down/
   │   ├── point left/
   │   ├── point right/
   │   └── point up/
   └── test/
       ├── back hand/
       ├── closed palm/
       ├── open palm/
       ├── point down/
       ├── point left/
       ├── point right/
       └── point up/
   ```

## Usage

### Quick Start

1. **Run the application**
   ```bash
   python main_app.py
   ```

2. **Train the gesture recognition model** (first time only)
   - When you first run the app, it will ask if you want to train the model
   - Click "Yes" or use the "Train Model" button in the interface
   - Training will take a few minutes and uses the images in your dataset folder

3. **Start gesture detection**
   - Click "Start Camera" to begin gesture detection
   - Allow camera access when prompted

4. **Load images**
   - Use "Open Image" to load a single image
   - Use "Open Folder" to load multiple images for navigation

5. **Control with gestures**
   - Make gestures in front of the camera
   - The app will detect and perform corresponding actions
   - Use "Pause" gesture (closed palm) to temporarily disable gesture control

### Manual Controls

If you prefer manual control or want to test without gestures:
- Use the buttons in the left panel for all image operations
- Manual controls work even when camera is not started

### Settings

- **Confidence Threshold**: Minimum confidence level for gesture detection (0.5-0.95)
- **Gesture Stability**: Number of consecutive detections needed before performing action (1-10)

## How It Works

### Architecture

1. **Gesture Recognition** (`gesture_recognizer.py`)
   - Uses MediaPipe to extract hand landmarks from camera frames
   - Trains a neural network on your custom gesture dataset
   - Classifies gestures in real-time with confidence scores

2. **Image Editor** (`image_editor.py`)
   - Tkinter-based GUI for image display and manipulation
   - Supports zoom, rotation, and navigation operations
   - Handles image loading, saving, and transformation history

3. **Main Application** (`main_app.py`)
   - Integrates gesture recognition with image editing
   - Manages camera feed and real-time detection
   - Provides unified interface for all functionality

### Gesture Detection Process

1. Camera captures video frames
2. MediaPipe extracts hand landmarks (21 points per hand)
3. Landmarks are fed to the trained neural network
4. Network predicts gesture with confidence score
5. Stable gestures (detected multiple times) trigger image actions
6. Actions are applied with cooldown period to prevent rapid-fire operations

## Customization

### Adding New Gestures

1. **Collect training data**: Add new gesture images to dataset/train/[gesture_name]/
2. **Update gesture mapping**: Edit the `gesture_actions` dictionary in `gesture_recognizer.py`
3. **Add action handling**: Implement the new action in `image_editor.py`
4. **Retrain model**: Delete existing model files and run training again

### Adjusting Performance

- **Frame rate**: Modify `time.sleep(0.1)` in `camera_loop()` to change detection frequency
- **Model architecture**: Edit the neural network in `train_model()` method
- **Preprocessing**: Adjust image preprocessing in `extract_hand_landmarks()`

## Troubleshooting

### Common Issues

1. **Camera not working**
   - Check if camera is connected and not used by other applications
   - Try changing camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`, etc.

2. **Poor gesture recognition**
   - Ensure good lighting conditions
   - Keep hand clearly visible in camera frame
   - Increase confidence threshold if too sensitive
   - Retrain model with more/better training data

3. **Model training fails**
   - Verify dataset structure matches expected format
   - Ensure training images are valid and accessible
   - Check that you have sufficient disk space

4. **Application crashes**
   - Make sure all dependencies are installed correctly
   - Check Python version compatibility
   - Look for error messages in console output

### Performance Tips

- **Better accuracy**: Use consistent lighting and background during training and detection
- **Faster detection**: Reduce image resolution or detection frequency
- **Stability**: Increase gesture stability threshold for more reliable detection

## File Structure

```
gesture_photo_controller/
├── main_app.py              # Main application entry point
├── gesture_recognizer.py    # Gesture recognition and model training
├── image_editor.py          # Image editing GUI and functionality
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── dataset/                # Training and test data
│   ├── train/             # Training images
│   └── test/              # Test images
├── gesture_model.h5        # Trained model (generated)
└── label_encoder.pkl       # Label encoder (generated)
```

## Dependencies

- **OpenCV**: Camera handling and image processing
- **MediaPipe**: Hand landmark detection
- **TensorFlow**: Neural network training and inference
- **NumPy**: Numerical operations
- **Pillow**: Image handling for GUI
- **Tkinter**: GUI framework (included with Python)
- **Scikit-learn**: Data preprocessing and utilities

## License

This project is open source. Feel free to modify and distribute as needed.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Future Enhancements

- Support for multiple hands
- Additional image editing operations (brightness, contrast, filters)
- Gesture recording and playback
- Voice commands integration
- Mobile app version
- Cloud-based model training
