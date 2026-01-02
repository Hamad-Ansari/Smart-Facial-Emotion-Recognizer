import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    requirements = [
        'streamlit==1.28.0',
        'opencv-python==4.8.1',
        'deepface==0.0.79',
        'pandas==2.1.3',
        'pillow==10.1.0',
        'matplotlib==3.8.2',
        'numpy==1.24.3',
        'plotly==5.18.0'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\n✅ All packages installed successfully!")
    print("\nTo run the application:")
    print("  streamlit run app.py")

if __name__ == "__main__":
    install_requirements()