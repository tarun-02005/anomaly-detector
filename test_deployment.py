#!/usr/bin/env python3
"""
Pre-deployment test script
Run this before deploying to ensure everything works
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import flask
        print("✓ Flask imported successfully")
        
        import cv2
        print("✓ OpenCV imported successfully")
        
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO imported successfully")
        
        import torch
        print("✓ PyTorch imported successfully")
        
        import torchvision
        print("✓ Torchvision imported successfully")
        
        import numpy
        print(f"✓ NumPy imported successfully (version: {numpy.__version__})")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_file():
    """Test if model file exists"""
    print("\nTesting model file...")
    model_path = "best_anomaly_model.pt"
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model file found: {model_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"✗ Model file not found: {model_path}")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directories...")
    dirs = ['static', 'static/uploads', 'static/processed', 'templates']
    
    all_exist = True
    for directory in dirs:
        if os.path.exists(directory):
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
            all_exist = False
    
    return all_exist

def test_template_files():
    """Test if template files exist"""
    print("\nTesting template files...")
    templates = ['templates/index.html', 'templates/detector.html']
    
    all_exist = True
    for template in templates:
        if os.path.exists(template):
            print(f"✓ Template exists: {template}")
        else:
            print(f"✗ Template missing: {template}")
            all_exist = False
    
    return all_exist

def test_deployment_files():
    """Test if deployment files exist"""
    print("\nTesting deployment files...")
    files = [
        'requirements.txt',
        'Dockerfile',
        'Procfile',
        'render.yaml',
        'runtime.txt',
        '.dockerignore'
    ]
    
    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"✓ File exists: {file}")
        else:
            print(f"✗ File missing: {file}")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 50)
    print("Pre-Deployment Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model File Test", test_model_file),
        ("Directory Test", test_directories),
        ("Template Test", test_template_files),
        ("Deployment Files Test", test_deployment_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready for deployment!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
