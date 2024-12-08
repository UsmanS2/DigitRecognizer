import subprocess
import sys

reqLibraries = [
    "tensorflow",
    "numpy",
    "Pillow",
]

# function that installs libraries
def installLibrary(library):
    subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# Check and Install Libraries
for library in reqLibraries:
    try:
        __import__(library)
        print(f"Imported {library}")
    except ImportError:
        installLibrary(library)
        print(f"Installing {library}")

# Import all necessary libraries
import tkinter as tk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

