import os
import sys
import shutil
import numpy as np

# Get name of current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add parent (source) directory to system path
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)

# Reset the output directory
output_dir = os.path.join(current_dir, "Outputs")
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Set numpy printing options
np.set_printoptions(
    precision=3,
    linewidth=10000,
    suppress=True,
    threshold=10000,
)
