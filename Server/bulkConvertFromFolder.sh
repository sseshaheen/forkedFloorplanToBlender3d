#!/bin/bash

# Enable debugging
set -x

# Check if the directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

IMAGE_DIR="$1"
OBJECT_DIR="./storage/objects"
BOUNDARY="----WebKitFormBoundary7MA4YWxkTrZu0gW"
SUPPORTED_FORMATS="jpg jpeg png tiff bmp gif"

echo "Starting script to process images in directory: $IMAGE_DIR"
echo "Supported formats: $SUPPORTED_FORMATS"

# Loop through each file in the directory
for file in "$IMAGE_DIR"/*; do
    # Get the file extension
    extension="${file##*.}"
    
    # Check if the file is an image of supported format
    if [[ $SUPPORTED_FORMATS =~ (^|[[:space:]])$extension($|[[:space:]]) ]]; then
        echo "Processing $file"

        # POST request
        response=$(curl -s -X POST "http://localhost:8000/?func=create&debug=1&verbose=1" \
                          -H "Accept: application/json" \
                          -F "file=@${file}")

        echo "POST response: $response"

        # Parse the response
        id=$(echo "$response" | awk -F\' '{print $2}')
        hash=$(echo "$response" | awk -F\' '{print $4}')

        echo "Parsed id: $id, hash: $hash"

        # File name without directory
        filename=$(basename -- "$file")

        # PUT request
        put_response=$(curl -s -X PUT "http://localhost:8000/?func=createandtransform&id=${id}&hash=${hash}&iformat=.${extension}&oformat=.obj&userId=Oq37pxGdFPYnvWbOL94ttbIFqD23&debug=1&verbose=1&session_id=${id}&filename=${filename}" \
                          -H "Accept: application/json" \
                          -H "Content-Type: multipart/form-data; boundary=$BOUNDARY" \
                          --form "file=@${file}")

        echo "PUT response: $put_response"

        # Check if the .obj file is created every 2 seconds
        obj_file="${OBJECT_DIR}/${id}.obj"
        while [ ! -f "$obj_file" ]; do
            echo "Waiting for $obj_file to be created..."
            sleep 2
        done

        echo "$obj_file has been created."

        # Series of actions
        # Add your series of actions here

    else
        echo "Skipping unsupported file format: $file"
    fi
done

echo "All files processed."
