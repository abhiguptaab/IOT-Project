# Os is required for managing files like directories
import cv2
import os
import requests
import numpy as np
import time

# Replace the URL with your own IPwebcam shot.jpg IP:port
url = 'http://192.168.43.10:8080/shot.jpg'

vid_cam = cv2.VideoCapture(0)
# vid_cam.get(cv2.CAP_PROP_FPS)
# print vid_cam

# For detecting the faces in each frame we will use Haarcascade Frontal Face default classifier of OpenCV
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Set unique id for each individual person
face_id = input("face iddd:")

# Variable for counting the no. of images
count = 0
print "Capturing Images Start"

while (True):

    # -------------------------------------------------
    # vid_cam = requests.get(url)
    # imgNp = np.array(bytearray(vid_cam.content), dtype=np.uint8)
    # image_frame = cv2.imdecode(imgNp, -1)
    # --------------------------------------------------

    # Capturing each video frame from the webcam
    _, image_frame = vid_cam.read()

    # Converting each frame to grayscale image
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detecting different faces
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Increasing the no. of images by 1 since frame we captured
        count += 1

        # Saving the captured image into the training_data folder
        cv2.imwrite("faces/" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # Displaying the frame with rectangular bounded box
        cv2.putText(image_frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', image_frame)


    # press 'q' for at least 100ms to stop this captu/ing process
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # We are taking 100 images for each person for the training data
    # If image taken reach 100, stop taking video
    elif count > 200:
        print "Capturing Images completed"

        break


# Terminate video
vid_cam.release()

# Terminate all started windows
cv2.destroyAllWindows()
import training
