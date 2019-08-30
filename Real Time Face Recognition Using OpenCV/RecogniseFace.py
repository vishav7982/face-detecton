'''Last Script For Face Recognition'''

import cv2
import numpy as np
import os

# creating the recognizer obj
recognizer = cv2.face.LBPHFaceRecognizer_create()

# loading the trained model
recognizer.read('trainer.yml')

# face detection for the face recognition

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

# initiate id counter
id = 0
name = ""
dataset = "Dataset"

# names related to ids: example ==> vishav: id=1,  etc
#names = ['None', 'vishav', 'rohit', 'CHaddi', 'ADnan']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:

    ret, img = cam.read()
    img = cv2.flip(img, 1)  # avoiding the mirror image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting the img into gray scale

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=10,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less than 100 ==> "0" is perfect match
        if confidence < 100:
            #id = names[id]
            for file in os.listdir(dataset):
                face_id = os.path.split(file)[-1].split(".")[1]
                if(id == int(face_id)): # determination of person by the returned id
                    name = os.path.split(file)[-1].split(".")[0]
                    break

            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        # name of the person
        cv2.putText(img=img,text= str(name), org= (x + 5, y - 5),fontFace= font,fontScale= 1,color= (255, 255, 255),thickness= 2)

        # for the confidence of that person
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

#  cleanup stuff !
print("\n Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
