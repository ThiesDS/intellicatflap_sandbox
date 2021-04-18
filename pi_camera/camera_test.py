import cv2

# instatiate webcam video capture
videoCaptureObject = cv2.VideoCapture(0)

result = True

while(result):

    # Capture webcam video
    ret,frame = videoCaptureObject.read()

    # Save to file
    cv2.imwrite("test_image.jpg",frame)

    # Set bool to false
    result = False

# Release
videoCaptureObject.release()
cv2.destroyAllWindows()