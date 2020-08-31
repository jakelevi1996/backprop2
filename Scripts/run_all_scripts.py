"""
This script automates running all of the scripts in this directory ("Scripts"),
and any subdirectories, except for this script ("run_all_scripts.py"), and the
directory initialisation script ("__init__.py")
"""
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

module_path_list = [
    os.path.join(dir_name, filename)
    for (dir_name, _, f_list) in os.walk(current_dir)
    for filename in f_list
    if filename.endswith(".py") and not any([
        filename.endswith("run_all_scripts.py"),
        filename.endswith("__init__.py"),
    ])
]

for module_path in module_path_list:
    display_string = "Running script {}".format(module_path)
    h_line = "*" * len(display_string)
    print(h_line, display_string, h_line, sep="\n")
    os.system("python " + module_path)
