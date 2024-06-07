import bpy
import sys
import os

# Function to get environment variables
def get_env_var(var_name):
    try:
        return os.environ[var_name]
    except KeyError:
        print(f"Error: Environment variable {var_name} not set.")
        sys.exit(1)

# Function to remove BOM from a file
def remove_bom(file_path):
    with open(file_path, 'rb') as file:
        content = file.read()
    
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
        with open(file_path, 'wb') as file:
            file.write(content)
        print(f"BOM removed from {file_path}")
    else:
        print(f"No BOM found in {file_path}")

# Read environment variables for file paths
input_file_path = get_env_var("INPUT_FILE_PATH")
output_file_path = get_env_var("OUTPUT_FILE_PATH")

# Remove BOM from the input GLTF file if present
remove_bom(input_file_path)

# Enable necessary Blender addons
from addon_utils import check, enable
bpy.ops.wm.read_factory_settings(use_empty=True)
for addon in ("io_export_dxf", "io_scene_gltf2"):
    default, enabled = check(addon)
    if not enabled:
        enable(addon, default_set=True, persistent=True)

# Import the GLTF file
bpy.ops.import_scene.gltf(filepath=input_file_path)

# Export to OBJ format
bpy.ops.export_scene.obj(filepath=output_file_path, check_existing=True)
