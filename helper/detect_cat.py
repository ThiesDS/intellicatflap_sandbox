import cv2
import os

def highlight_faces(frame, faces):
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.rectangle(frame, ((0,frame.shape[0] -25)),(270, frame.shape[0]), (255,255,255), -1)

def cv2_cat_detector(img_dir='./data/detected_cats/',detector_file='./config/haarcascade_frontalcatface_extended.xml'):

    # Alternative, faster face detector
    detector = cv2.CascadeClassifier(detector_file)

    # images with detected cats
    files = os.listdir(img_dir)

    for file in files:

        # Read image
        img = cv2.imread(img_dir + file) 

        # Detect cat
        faces = detector.detectMultiScale(img)#, scaleFactor=1.2)#, minNeighbors=3)

        if len(faces)>0:
            # Highlight faces
            highlight_faces(img, faces)

            # Save to file again
            new_file = '.'.join(file.split('.')[0:-1]) + '_face.jpg'
            cv2.imwrite(local_dir + new_file, img)
