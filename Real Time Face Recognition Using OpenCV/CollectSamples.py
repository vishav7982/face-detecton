'''A script to Creating the Datasets(face samples) of the persons to be recognised Using OpenCv.'''
# importing the libraries

import cv2
import os
# opening the webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640) #  video width
cam.set(4, 480) #  video height

# path to the dataset folder
dataset = "Dataset"
# For each person, enter one numeric face id
#face_id = input('\n enter user id ==>  ')
face_id =int(len(os.listdir(dataset))/60)+1  # bcoz we are taking the 60  face samples
face_name = input("Enter Your Name : ")

print("\n Initializing face captures. Look At the camera and wait ...!!")
# Initialize individual sampling face count
count = 0 # counter to Count The face samples
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Avoiding The Mirror image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert it to gray image
    faces = face_detector.detectMultiScale(gray, 1.3, 5) # detection of face

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)  # put rectangle on The face
        count += 1

        # Save the captured image into the dataset folder
        cv2.imwrite("Dataset//"+face_name+"." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.putText(img,"Capture Samples : "+str(count), (60, 60), fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(255,255,0),thickness=2,lineType=cv2.LINE_AA)

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 60: # Take 60 face sample and stop video
         break

# cleanup Stuff
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
