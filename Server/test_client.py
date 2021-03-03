import cv2
import pywavefront
import os
import requests
import json as jsonlib
import time

"""
The purpose of this file is to test core functionallity of the server.
We also test some common problems.
"""
path_to_test_image="../Images/example.png"
path_to_result_folder="./test-result"

show = False

url="http://127.0.0.1:8000"

if __name__ == "__main__":
    # Create folder to play around with
    if not os.path.exists(path_to_result_folder):
        os.makedirs(path_to_result_folder)
        print("Created test folder.")

    # Load image file
    print("Loading image...")
    image = cv2.imread(path_to_test_image, 0) 

    if show:
        cv2.imshow("inImage", image)
        cv2.waitKey(0)

    # Get some information
    json = {'func': 'info'} # query
    print("----- Get info... ----- ")
    response = requests.get(url, params=json)
    print(response)
    print("Printing Rest Api information:")
    print(response.text)
    print("DONE!")

    # GET entire table
    json = {'func': 'all'} # query
    print("----- Get all files... ----- ")
    response = requests.get(url, params=json)
    print(response)
    print("Printing file array:")
    print(response.text)
    print("DONE!")
    
    # GET and show all images on server
    json = {'func': 'images'}
    response = requests.get(url, params=json)
    print("----- Get Images ----- ")
    print(response)
    print(response.text)
    print("DONE!")

    # GET and show all objects on server
    json = {'func': 'objects'}
    print("----- Get Objects ----- ")
    response = requests.get(url, params=json)
    print(response)
    print(response.text)
    print("DONE!")

    # send POST request to get new file ID
    json = {'func': 'create'}
    print("----- POST create new file ----- ")
    response = requests.post(url, data=jsonlib.dumps(json))
    print(response)
    print(response.text)
    print("DONE!")

    # Handle response
    try:
        pair = list(eval(response.text))
    except Exception as e:
        print("ERROR: " + str(e), " \n With response:" ,response.text)
        exit()
    id = pair[0]
    hash= pair[1]

    # send PUT image file to id
    print("----- PUT upload new file ----- ")
    with open(path_to_test_image, 'rb') as infile:
        json = {'func': 'create', 'id':id, 'hash':hash, 'iformat':'.jpg'}
        headers = {'Content-Type': 'multipart/form-data'}
        response = requests.put(url, data=infile, params=json, headers=headers)
    # TODO test sending to id that already exists
    print(response)
    print(response.text)
    print("DONE!")

    #  Test sending to id that already exists
    print("----- PUT upload same file ----- ")
    with open(path_to_test_image, 'rb') as infile:
        json = {'func': 'create', 'id':id, 'hash':hash, 'iformat':'.jpg'}
        headers = {'Content-Type': 'multipart/form-data'}
        response = requests.put(url, data=infile, params=json, headers=headers)
    print(response)
    print(response.text)
    print("Test Passed!")
    print("DONE!")

    # send POST command to create .obj from image id
    json = {'func': 'transform', 'id':id, 'oformat':'.obj'}
    print("----- POST transform image "+id+" into 3d model of format .obj ----- ")
    response = requests.post(url, data=jsonlib.dumps(json))
    print(response)
    print(response.text)
    print("DONE!")

    # send GET all processes
    json = {'func': 'processes'}
    print("----- GET all processes ----- ")
    response = requests.get(url, params=json)
    print(response)
    print(len(response.json()), "Processes exist!")
    print("DONE!")

    # send GET one process 
    # Get our process pid
    tmp_process = None
    for process in response.json():
        if process["in"] == str(id):
            pid = process["pid"]
            tmp_process = process
    json = {'func': 'process', 'pid':pid}
    print("----- GET "+str(pid)+" process ----- ")
    response = requests.get(url, params=json)
    print(response)
    print(response.text)
    print("DONE!")

    # send POST request to get new file ID
    json = {'func': 'create'}
    response = requests.post(url, data=jsonlib.dumps(json))
    
    try:
        pair = list(eval(response.text))
    except Exception as e:
        print("ERROR: " + str(e), " \n With response:" ,response.text)
        exit()
    id2 = pair[0]
    hash2= pair[1]

    # send PUT and create object
    print("----- PUT "+str(id2)+" process ----- ")
    with open(path_to_test_image, 'rb') as infile:
        json = {'func': 'createandtransform', 'id':id2, 'hash':hash2, 'iformat':'.jpg', 'oformat':'.obj'}
        headers = {'Content-Type': 'multipart/form-data'}
        response = requests.put(url, data=infile, params=json, headers=headers)
    print(response)
    print(response.text)
    print("DONE!")

    # get download and show image
    json = {'func': 'image', 'id': id}
    print("----- GET File process ----- ")
    response = requests.get(url, params=json)

    test_path = path_to_result_folder+"/"+id+".png"

    file = open(test_path, "wb")
    file.write(response.content)
    file.close()

    print("File downloaded! ", test_path) 
    print("DONE!")

    if show:
        print("Showing image!")   
        image = cv2.imread(path_to_result_folder+"/"+id+".png")
        cv2.imshow("gathered image", image)
        cv2.waitKey(0)

    time.sleep(10)
    
    # get download and show .obj file
    json = {'func': 'object', 'id': id, 'oformat':'.obj'} # returns .blend if no format set!
    print("----- GET Object process ----- ")
    response = requests.get(url, params=json)
    print(response)
    obj_path = path_to_result_folder+"/"+id+json["oformat"]

    file = open(obj_path, "wb")
    file.write(response.content)
    file.close()

    json = {'func': 'object', 'id': id, 'oformat':'.mtl'} # returns .blend if no format set!
    print("----- GET Object process ----- ")
    response = requests.get(url, params=json)
    print(response)
    obj_path = path_to_result_folder+"/"+id+json["oformat"]

    file = open(obj_path, "wb")
    file.write(response.content)
    file.close()
    
    print("File downloaded!") 
    print("DONE!")
    
    #if show:
    print("Showing 3d modell!")
    # run a blender script that loads our object and start
        
    exit(0)

    # send post to remove file & object!
    json = {'func': 'remove', 'id': id}
    print("----- Removing our id from server ----- ")
    #response = requests.post(url, params=json)
    #print(response)
    #print(response.text)
    print("File id removed!") 
    print("DONE!")
    

    # send get help for post, put and get!
    json = {'func': 'help'}
    print("----- Sending help request to get,post,put ----- ")
    #response = requests.get(url, params=json)
    #response = requests.post(url, params=json)
    #response = requests.put(url, params=json)
    