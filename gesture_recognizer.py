import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from typing import List, Tuple, Optional
import logging

class GestureRecognizer:
    def __init__(self, model_path: str = "gesture_model.h5", label_encoder_path: str = "label_encoder.pkl"):
        """Initialize the gesture recognizer with MediaPipe and TensorFlow model."""
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Model and label encoder
        self.model = None
        self.label_encoder = None
        
        # Gesture mapping to actions
        self.gesture_actions = {
            'open palm': 'reset',           # Reset image to original
            'closed palm': 'pause',         # Pause/stop current action
            'back hand': 'prev_image',      # Previous image
            'point up': 'zoom_in',          # Zoom in
            'point down': 'zoom_out',       # Zoom out
            'point left': 'rotate_left',    # Rotate left
            'point right': 'rotate_right'   # Rotate right
        }
        
        # Load model and label encoder if they exist
        self.load_model()
    
    def extract_hand_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract hand landmarks from an image using MediaPipe."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]  # Use first hand detected
            landmark_array = []
            
            for landmark in landmarks.landmark:
                landmark_array.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmark_array)
        
        return None
    
    def prepare_dataset(self, dataset_path: str = "dataset") -> Tuple[np.ndarray, np.ndarray]:
        """Prepare the dataset by extracting landmarks from all images."""
        train_path = os.path.join(dataset_path, "train")
        
        X = []  # Features (landmarks)
        y = []  # Labels (gesture names)
        
        print("Preparing dataset...")
        
        for gesture_folder in os.listdir(train_path):
            gesture_path = os.path.join(train_path, gesture_folder)
            
            if not os.path.isdir(gesture_path):
                continue
                
            print(f"Processing {gesture_folder}...")
            
            for image_file in os.listdir(gesture_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(gesture_path, image_file)
                    
                    # Load and process image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Extract landmarks
                    landmarks = self.extract_hand_landmarks(image)
                    
                    if landmarks is not None:
                        X.append(landmarks)
                        y.append(gesture_folder)
        
        print(f"Dataset prepared: {len(X)} samples")
        return np.array(X), np.array(y)
    
    def train_model(self, dataset_path: str = "dataset"):
        """Train the gesture recognition model."""
        # Prepare dataset
        X, y = self.prepare_dataset(dataset_path)
        
        if len(X) == 0:
            raise ValueError("No data found in dataset!")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Create model
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Save model and label encoder
        self.save_model()
        
        return history
    
    def save_model(self):
        """Save the trained model and label encoder."""
        if self.model is not None:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")
        
        if self.label_encoder is not None:
            with open(self.label_encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"Label encoder saved to {self.label_encoder_path}")
    
    def load_model(self):
        """Load the trained model and label encoder."""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"Model loaded from {self.model_path}")
            
            if os.path.exists(self.label_encoder_path):
                with open(self.label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"Label encoder loaded from {self.label_encoder_path}")
        
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict_gesture(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Predict gesture from an image."""
        if self.model is None or self.label_encoder is None:
            return None, 0.0
        
        landmarks = self.extract_hand_landmarks(image)
        
        if landmarks is None:
            return None, 0.0
        
        # Reshape for prediction
        landmarks = landmarks.reshape(1, -1)
        
        # Predict
        predictions = self.model.predict(landmarks, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get gesture name
        gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return gesture_name, confidence
    
    def get_action_for_gesture(self, gesture: str) -> str:
        """Get the action associated with a gesture."""
        return self.gesture_actions.get(gesture, 'unknown')
    
    def draw_landmarks(self, image: np.ndarray) -> np.ndarray:
        """Draw hand landmarks on the image."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return image

if __name__ == "__main__":
    # Example usage: Train the model
    recognizer = GestureRecognizer()
    
    try:
        # Train the model if it doesn't exist
        if not os.path.exists("gesture_model.h5"):
            recognizer.train_model()
        else:
            print("Model already exists. Delete 'gesture_model.h5' to retrain.")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the dataset in the correct format!")
