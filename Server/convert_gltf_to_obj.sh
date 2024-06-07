#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input_file.gltf>"
  exit 1
fi

# Paths to input GLTF file and output OBJ file
INPUT_FILE_PATH="$1"
OUTPUT_FILE_PATH="${INPUT_FILE_PATH%.gltf}.obj"

# Set environment variables
export INPUT_FILE_PATH=$INPUT_FILE_PATH
export OUTPUT_FILE_PATH=$OUTPUT_FILE_PATH

# Run Blender in background mode with the Python script
blender --background --python convert_gltf_to_obj.py
