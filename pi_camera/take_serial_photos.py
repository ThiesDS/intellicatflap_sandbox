import cv2
import time
from pathlib import Path

# instatiate webcam video capture
videoCaptureObject = cv2.VideoCapture(0)

# Initiate additional varialbes and paramters
result = True
img_no = 0
img_no_max = 10
img_path = "output/"

# Create folder if not exists
Path(img_path).mkdir(parents=True, exist_ok=True)

while(result):

    # Capture webcam video
    ret,frame = videoCaptureObject.read()

    # Save to file
    img_name = "img_" + str(img_no) + ".jpg"
    img_destination = img_path + img_name  
    cv2.imwrite(img_destination,frame)

    # Set bool to false when 30 images are taken
    if img_no==img_no_max:
        result = False

    # Increase no of image
    img_no += 1

    # Sleep for 1 seconds
    time.sleep(1) 

# Release
videoCaptureObject.release()
cv2.destroyAllWindows()