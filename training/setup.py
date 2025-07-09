#!/usr/bin/env python3
"""
Setup script for YOLO Classification Training Environment.

This script automates the setup process:
1. Creates virtual environment
2. Installs dependencies
3. Validates setup

Usage:
    python setup.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, shell=False):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=shell, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def create_virtual_environment():
    """Create virtual environment."""
    print("📦 Creating virtual environment...")
    
    venv_path = Path("training_env")
    
    if venv_path.exists():
        print(f"⚠️  Virtual environment already exists at {venv_path}")
        response = input("Do you want to recreate it? (y/N): ").lower()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_path)
            print("🗑️  Removed existing environment")
        else:
            print("✅ Using existing environment")
            return True
    
    # Create virtual environment
    success, output = run_command([sys.executable, "-m", "venv", str(venv_path)])
    
    if success:
        print("✅ Virtual environment created successfully")
        return True
    else:
        print(f"❌ Failed to create virtual environment: {output}")
        return False


def get_pip_command():
    """Get pip command for current platform."""
    if platform.system() == "Windows":
        return str(Path("training_env") / "Scripts" / "pip")
    else:
        return str(Path("training_env") / "bin" / "pip")


def install_dependencies():
    """Install required dependencies."""
    print("📥 Installing dependencies...")
    
    pip_cmd = get_pip_command()
    
    # Upgrade pip first
    print("⬆️  Upgrading pip...")
    success, output = run_command([pip_cmd, "install", "--upgrade", "pip"])
    if not success:
        print(f"⚠️  Warning: Failed to upgrade pip: {output}")
    
    # Install requirements
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        return False
    
    print("📦 Installing packages from requirements.txt...")
    success, output = run_command([pip_cmd, "install", "-r", "requirements.txt"])
    
    if success:
        print("✅ Dependencies installed successfully")
        return True
    else:
        print(f"❌ Failed to install dependencies: {output}")
        return False


def validate_installation():
    """Validate the installation."""
    print("🔍 Validating installation...")
    
    # Get python command
    if platform.system() == "Windows":
        python_cmd = str(Path("training_env") / "Scripts" / "python")
    else:
        python_cmd = str(Path("training_env") / "bin" / "python")
    
    # Test critical imports
    test_imports = [
        "import cv2; print(f'OpenCV: {cv2.__version__}')",
        "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')",
        "import torch; print(f'PyTorch: {torch.__version__}')",
        "import numpy; print(f'NumPy: {numpy.__version__}')",
        "import PIL; print(f'Pillow: {PIL.__version__}')",
        "import yaml; print('PyYAML: OK')"
    ]
    
    all_passed = True
    
    for test in test_imports:
        success, output = run_command([python_cmd, "-c", test])
        if success:
            print(f"✅ {output.strip()}")
        else:
            print(f"❌ Import failed: {test}")
            all_passed = False
    
    # Test GPU availability
    gpu_test = "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    success, output = run_command([python_cmd, "-c", gpu_test])
    if success:
        print(f"🖥️  {output.strip()}")
    
    return all_passed


def create_activation_scripts():
    """Create convenient activation scripts."""
    print("📝 Creating activation scripts...")
    
    # Windows batch file
    windows_script = """@echo off
echo Activating YOLO Training Environment...
call training_env\\Scripts\\activate.bat
echo ✅ Environment activated!
echo.
echo 🚀 Ready to train! Run: python train.py
cmd /k
"""
    
    # Linux/Mac shell script  
    unix_script = """#!/bin/bash
echo "Activating YOLO Training Environment..."
source training_env/bin/activate
echo "✅ Environment activated!"
echo
echo "🚀 Ready to train! Run: python train.py"
exec bash
"""
    
    try:
        # Create Windows script
        with open("activate.bat", "w") as f:
            f.write(windows_script)
        
        # Create Unix script
        with open("activate.sh", "w") as f:
            f.write(unix_script)
        
        # Make Unix script executable
        if platform.system() != "Windows":
            os.chmod("activate.sh", 0o755)
        
        print("✅ Activation scripts created")
        return True
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to create activation scripts: {e}")
        return False


def print_next_steps():
    """Print next steps for user."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    
    if platform.system() == "Windows":
        print("\n📋 Next steps:")
        print("1. Activate environment:")
        print("   • Double-click: activate.bat")
        print("   • Or manually: training_env\\Scripts\\activate")
        print("\n2. Run training:")
        print("   python train.py")
    else:
        print("\n📋 Next steps:")
        print("1. Activate environment:")
        print("   • Run: ./activate.sh")
        print("   • Or manually: source training_env/bin/activate")
        print("\n2. Run training:")
        print("   python train.py")
    
    print("\n📚 Configuration:")
    print("   • Edit config.yaml to adjust training parameters")
    print("   • See README.md for detailed instructions")
    
    print("\n🔧 Troubleshooting:")
    print("   • GPU issues: Set device: 'cpu' in config.yaml")
    print("   • Memory issues: Reduce batch_size in config.yaml")


def main():
    """Main setup function."""
    print("🚀 YOLO Classification Training Environment Setup")
    print("="*60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Step 1: Create virtual environment
    if not create_virtual_environment():
        print("❌ Setup failed at virtual environment creation")
        sys.exit(1)
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Step 3: Validate installation
    if not validate_installation():
        print("⚠️  Setup completed with warnings - some packages may not work correctly")
    
    # Step 4: Create activation scripts
    create_activation_scripts()
    
    # Step 5: Print next steps
    print_next_steps()


if __name__ == "__main__":
    main() 