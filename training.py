
import cv2
import os
import numpy as np
from PIL import Image

# We will be using Local Binary Patterns Histograms for face recognization since it's quite accurate than the rest
recognizer = cv2.face.LBPHFaceRecognizer_create()

# For detecting the faces in each frame we will use Haarcascade Frontal Face default classifier of OpenCV
detector = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml");


def getImagesAndLabels(path):

    # Getting all file paths
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #if imagePat + '.DS_Store' in imagePaths:
        #imagePaths.remove(imagePat + '.DS_Store')

    #empty face sample initialised
    faceSamples=[]
    
    # IDS for each individual
    ids = []

    # Looping through all the file path
    for imagePath in imagePaths:

        # converting image to grayscale
        PIL_img = Image.open(imagePath).convert('L')

        # converting PIL image to numpy array using array() method of numpy
        img_numpy = np.array(PIL_img,'uint8')

        # Getting the image id
        #1+2.split[+][1]
        id = int(os.path.split(imagePath)[-1].split(".")[0])
        print(id)
        # Getting the face from the training images
        faces = detector.detectMultiScale(img_numpy)

        # Looping for each face and appending it to their respective IDs
        for (x,y,w,h) in faces:

            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Add the ID to IDs
            ids.append(id)

    # Passing the face array and IDs array
    return faceSamples,ids

# Getting the faces and IDs
faces,ids = getImagesAndLabels("faces/")

# Training the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Saving the model into s_model.yml

recognizer.write('saved_model/s_model.yml')
