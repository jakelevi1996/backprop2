import os, sys

# Get name of current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add parent (source) directory to system path
source_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(source_dir)

# Create output directory if it doesn't already exist
output_dir = os.path.join(current_dir, "Outputs")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
