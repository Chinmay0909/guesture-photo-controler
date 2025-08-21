#!/usr/bin/env python3
"""
Setup script for Gesture-Controlled Image Editor
This script helps with initial setup and dependency installation.
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def check_dataset():
    """Check if dataset structure is correct."""
    print("\nChecking dataset structure...")
    
    dataset_path = Path("dataset")
    required_folders = ["train", "test"]
    required_classes = ["back hand", "closed palm", "open palm", "point down", "point left", "point right", "point up"]
    
    if not dataset_path.exists():
        print("✗ Dataset folder not found!")
        return False
    
    for folder in required_folders:
        folder_path = dataset_path / folder
        if not folder_path.exists():
            print(f"✗ {folder} folder not found in dataset!")
            return False
        
        for class_name in required_classes:
            class_path = folder_path / class_name
            if not class_path.exists():
                print(f"✗ {class_name} folder not found in {folder}!")
                return False
            
            # Check if there are images
            image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
            if len(image_files) == 0:
                print(f"✗ No images found in {folder}/{class_name}!")
                return False
    
    print("✓ Dataset structure is correct")
    return True

def check_camera():
    """Check if camera is available."""
    print("\nChecking camera availability...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera is available")
            cap.release()
            return True
        else:
            print("✗ Camera not found or not accessible")
            return False
    except ImportError:
        print("! OpenCV not installed yet - camera check will be done after installation")
        return True

def create_shortcuts():
    """Create convenience scripts."""
    print("\nCreating convenience scripts...")
    
    # Create run script for Windows
    if os.name == 'nt':
        with open("run_app.bat", "w") as f:
            f.write("@echo off\n")
            f.write("echo Starting Gesture-Controlled Image Editor...\n")
            f.write("python main_app.py\n")
            f.write("pause\n")
        print("✓ Created run_app.bat")
    
    # Create run script for Unix/Linux/Mac
    else:
        with open("run_app.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo 'Starting Gesture-Controlled Image Editor...'\n")
            f.write("python3 main_app.py\n")
        
        # Make executable
        os.chmod("run_app.sh", 0o755)
        print("✓ Created run_app.sh")

def main():
    """Main setup function."""
    print("=== Gesture-Controlled Image Editor Setup ===\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dataset
    dataset_ok = check_dataset()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check camera (after OpenCV installation)
    camera_ok = check_camera()
    
    # Create shortcuts
    create_shortcuts()
    
    print("\n=== Setup Complete ===")
    
    if dataset_ok and camera_ok:
        print("✓ Everything looks good! You can now run the application.")
        print("\nTo start the application:")
        if os.name == 'nt':
            print("  - Double-click run_app.bat")
            print("  - Or run: python main_app.py")
        else:
            print("  - Run: ./run_app.sh")
            print("  - Or run: python3 main_app.py")
    else:
        print("\n⚠ Some issues were found:")
        if not dataset_ok:
            print("  - Dataset structure needs to be fixed")
        if not camera_ok:
            print("  - Camera access needs to be checked")
        print("\nPlease fix these issues before running the application.")

if __name__ == "__main__":
    main()
