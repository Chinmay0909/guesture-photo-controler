#!/usr/bin/env python3
"""
Test script to verify all components are working correctly.
Run this to test individual components before using the main application.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✓ MediaPipe imported successfully")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
        print(f"  TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import tkinter as tk
        print("✓ Tkinter imported successfully")
    except ImportError as e:
        print(f"✗ Tkinter import failed: {e}")
        return False
    
    try:
        from PIL import Image, ImageTk
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_camera():
    """Test camera functionality."""
    print("\nTesting camera...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Could not open camera")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Could not read frame from camera")
            cap.release()
            return False
        
        print(f"✓ Camera working - Frame size: {frame.shape}")
        cap.release()
        return True
    
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe hand detection."""
    print("\nTesting MediaPipe hand detection...")
    
    try:
        import mediapipe as mp
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Create a test image with a simple hand-like shape
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(test_image, (320, 240), 50, (255, 255, 255), -1)  # Palm
        cv2.rectangle(test_image, (300, 190), (340, 240), (255, 255, 255), -1)  # Finger
        
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        print("✓ MediaPipe hand detection initialized successfully")
        hands.close()
        return True
    
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return False

def test_dataset_structure():
    """Test dataset structure."""
    print("\nTesting dataset structure...")
    
    dataset_path = Path("dataset")
    if not dataset_path.exists():
        print("✗ Dataset folder not found")
        return False
    
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"
    
    if not train_path.exists():
        print("✗ Train folder not found")
        return False
    
    if not test_path.exists():
        print("✗ Test folder not found")
        return False
    
    required_classes = ["back hand", "closed palm", "open palm", "point down", "point left", "point right", "point up"]
    
    train_counts = {}
    test_counts = {}
    
    for class_name in required_classes:
        train_class_path = train_path / class_name
        test_class_path = test_path / class_name
        
        if not train_class_path.exists():
            print(f"✗ Train class '{class_name}' not found")
            return False
        
        if not test_class_path.exists():
            print(f"✗ Test class '{class_name}' not found")
            return False
        
        # Count images
        train_images = list(train_class_path.glob("*.jpg")) + list(train_class_path.glob("*.jpeg")) + list(train_class_path.glob("*.png"))
        test_images = list(test_class_path.glob("*.jpg")) + list(test_class_path.glob("*.jpeg")) + list(test_class_path.glob("*.png"))
        
        train_counts[class_name] = len(train_images)
        test_counts[class_name] = len(test_images)
    
    print("✓ Dataset structure is correct")
    print("\nDataset summary:")
    print("Train set:")
    for class_name, count in train_counts.items():
        print(f"  {class_name}: {count} images")
    print("Test set:")
    for class_name, count in test_counts.items():
        print(f"  {class_name}: {count} images")
    
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    print(f"\nTotal: {total_train} training images, {total_test} test images")
    
    return True

def test_gesture_recognizer():
    """Test gesture recognizer component."""
    print("\nTesting gesture recognizer...")
    
    try:
        from gesture_recognizer import GestureRecognizer
        
        recognizer = GestureRecognizer()
        print("✓ GestureRecognizer initialized successfully")
        
        # Test landmark extraction with dummy image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = recognizer.extract_hand_landmarks(test_image)
        print("✓ Landmark extraction method works")
        
        return True
    
    except Exception as e:
        print(f"✗ GestureRecognizer test failed: {e}")
        return False

def test_image_editor():
    """Test image editor component (without GUI)."""
    print("\nTesting image editor components...")
    
    try:
        # Test if we can import without errors
        import tkinter as tk
        
        # Create a hidden root window for testing
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        from image_editor import ImageEditor
        
        # Test that we can create an ImageEditor instance
        app = ImageEditor(root)
        print("✓ ImageEditor can be imported and instantiated successfully")
        
        root.destroy()
        return True
    
    except Exception as e:
        print(f"✗ ImageEditor test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Component Testing ===\n")
    
    tests = [
        ("Package imports", test_imports),
        ("Camera", test_camera),
        ("MediaPipe", test_mediapipe),
        ("Dataset structure", test_dataset_structure),
        ("Gesture recognizer", test_gesture_recognizer),
        ("Image editor", test_image_editor),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} test...")
        print('='*50)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to run the main application.")
        print("Run: python main_app.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
        print("You may need to install dependencies or check your setup.")

if __name__ == "__main__":
    main()
