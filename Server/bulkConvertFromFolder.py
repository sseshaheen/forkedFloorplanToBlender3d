import os
import shutil
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def process_images(image_dir):
    processing_dir = os.path.join(image_dir, "processing")
    success_dir = os.path.join(image_dir, "fullSuccess")
    base_dir = "/home/apps/forkedFloorplanToBlender3d/Server/storage"
    object_dir = os.path.join(base_dir, "objects")
    debug_dir = os.path.join(base_dir, "debug")
    final_dir = os.path.join(base_dir, "final")
    supported_formats = ["jpg", "jpeg", "png", "tiff", "bmp", "gif"]

    # Create necessary directories if they do not exist
    os.makedirs(processing_dir, exist_ok=True)
    os.makedirs(success_dir, exist_ok=True)

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))

    # Loop through each file in the directory
    for file in sorted(os.listdir(image_dir)):
        file_path = os.path.join(image_dir, file)
        extension = file.split('.')[-1].lower()

        # Check if the file is an image of supported format
        if extension in supported_formats:
            print(f"Processing {file_path}")

            # Move the file to the processing directory
            processing_file_path = os.path.join(processing_dir, file)
            shutil.move(file_path, processing_file_path)

            # POST request
            post_url = "http://localhost:8000/?func=create&debug=1&verbose=1"
            post_headers = {
                "Accept": "application/json",
                "Content-Length": "0"
            }

            response = session.post(post_url, headers=post_headers)
            print(f"POST response: {response.text}")

            # Parse the response
            try:
                id, hash = response.text.split(",")[0].strip("('"), response.text.split(",")[1].strip(" '")
                print(f"Parsed id: {id}, hash: {hash}")
            except ValueError:
                print(f"Failed to parse POST response: {response.text}")
                continue

            # File name without directory
            filename = os.path.basename(processing_file_path)

            # PUT request
            put_url = (
                f"http://localhost:8000/?func=createandtransform&id={id}"
                f"&hash={hash}"
                f"&iformat=.{extension}"
                f"&oformat=.obj"
                f"&userId=Oq37pxGdFPYnvWbOL94ttbIFqD23"
                f"&debug=1"
                f"&verbose=1"
                f"&session_id={id}"
                f"&filename={filename}"
            )

            put_headers = {
                "Accept": "application/json"
            }
            with open(processing_file_path, 'rb') as file:
                try:
                    put_response = session.put(put_url, headers=put_headers, files={'file': file})
                    print(f"PUT response: {put_response.text}")
                except requests.exceptions.RequestException as e:
                    print(f"PUT request failed: {e}")
                    continue

            # Create the debug directory for this ID if it doesn't exist
            debug_id_dir = os.path.join(debug_dir, id)
            os.makedirs(debug_id_dir, exist_ok=True)

            # Copy the input file to the debug directory
            debug_file_path = os.path.join(debug_id_dir, filename)
            print(f"Copying {processing_file_path} to {debug_file_path}")
            shutil.copy(processing_file_path, debug_file_path)

            # Check if the .obj file is created every 10 seconds, up to 24 retries (4 minutes)
            # TODO: optimize to be based on actual conversion results
            # TODO: when a file is done successfully, move it to a done folder
            obj_file = os.path.join(object_dir, f"{id}.obj")
            retries = 24
            while not os.path.isfile(obj_file) and retries > 0:
                print(f"Waiting for {obj_file} to be created...")
                time.sleep(10)
                retries -= 1

            if os.path.isfile(obj_file):
                print(f"{obj_file} has been created.")

                # Move the debug folder to the final directory
                final_id_dir = os.path.join(final_dir, id)
                print(f"Moving {debug_id_dir} to {final_id_dir}")
                shutil.move(debug_id_dir, final_id_dir)

                # Move all files containing the ID in their filename to the final directory
                for obj_filename in os.listdir(object_dir):
                    if id in obj_filename:
                        src_file = os.path.join(object_dir, obj_filename)
                        dest_file = os.path.join(final_id_dir, obj_filename)
                        print(f"Moving {src_file} to {dest_file}")
                        shutil.move(src_file, dest_file)

                # Move the processed file from processing to success directory
                success_file_path = os.path.join(success_dir, file)
                print(f"Moving {processing_file_path} to {success_file_path}")
                shutil.move(processing_file_path, success_file_path)
            else:
                print(f"{obj_file} was not created after {retries} retries. Moving on to the next file.")

        else:
            print(f"Skipping unsupported file format: {file}")

    print("All files processed.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python process_images.py <directory>")
        sys.exit(1)

    image_dir = sys.argv[1]
    process_images(image_dir)
