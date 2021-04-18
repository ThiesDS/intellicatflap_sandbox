#from mtcnn.mtcnn import MTCNN
import cv2

# Define func to highlight faces in frame
#def highlight_faces(frame, faces):
#    for face in faces:
#        x,y,w,h = face['box']
#        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)    
#        cv2.rectangle(frame, ((0,frame.shape[0] -25)),(270, frame.shape[0]), (255,255,255), -1)

def highlight_faces2(frame, faces):
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.rectangle(frame, ((0,frame.shape[0] -25)),(270, frame.shape[0]), (255,255,255), -1)

# Instantiate face detector
#detector = MTCNN()

# Alternative, faster face detector
detector2 = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
capture = cv2.VideoCapture(0)
 
# Check if camera opened successfully
if (capture.isOpened()== False): 
    print("Error opening video stream or file")

# Define the codec and create VideoWriter object. The output is stored in 'outpy.avi' file.
# Define the fps to be equal to 10. Also frame size is passed.
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
fps = int(capture.get(5))
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
output = cv2.VideoWriter('output_facedetect_cat.avi',fourcc, fps, (frame_width,frame_height))

# Loop infinitely
idx = 0
while(True):

    # Read video stram 
    ret, frame = capture.read()

    if ret == True:

        # If faces are present, hightlight
        if idx%1 == 0:
            # Detect faces in frame
            #faces = detector.detect_faces(frame)
            faces = detector2.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3)

            # Highlight faces
            #highlight_faces(frame, faces)
            highlight_faces2(frame, faces)
        
        # Write the frame into the file 'output.avi'
        output.write(frame)

        # Show
        cv2.imshow('Faces highlighted',frame)
        
        # Break eveything when "ESC" is pressed
        if cv2.waitKey(1) == 27:
            break
    
    # Else, break
    else:
        break
    
    # Increase counter
    idx += 1

# When everything is done, release video caputre and video writer objects
capture.release()
output.release()

# Close all the frames
cv2.destroyAllWindows()
