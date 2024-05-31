import os
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def process_images(image_dir):
    object_dir = "/home/apps/forkedFloorplanToBlender3d/Server/storage/objects"
    supported_formats = ["jpg", "jpeg", "png", "tiff", "bmp", "gif"]

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))

    # Loop through each file in the directory
    for file in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file)
        extension = file.split('.')[-1].lower()

        # Check if the file is an image of supported format
        if extension in supported_formats:
            print(f"Processing {file_path}")

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
            filename = os.path.basename(file_path)

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
            files = {'file': open(file_path, 'rb')}

            try:
                put_response = session.put(put_url, headers=put_headers, files=files)
                print(f"PUT response: {put_response.text}")
            except requests.exceptions.RequestException as e:
                print(f"PUT request failed: {e}")
                continue

            # Check if the .obj file is created every 4 seconds, up to 5 retries
            obj_file = os.path.join(object_dir, f"{id}.obj")
            retries = 5
            while not os.path.isfile(obj_file) and retries > 0:
                print(f"Waiting for {obj_file} to be created...")
                time.sleep(4)
                retries -= 1

            if os.path.isfile(obj_file):
                print(f"{obj_file} has been created.")
                # Series of actions
                # Add your series of actions here
            else:
                print(f"{obj_file} was not created after 5 retries. Moving on to the next file.")

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
