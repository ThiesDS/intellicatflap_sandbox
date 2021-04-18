import cv2

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
capture = cv2.VideoCapture(0)
 
# Check if camera opened successfully
if (capture.isOpened()== False): 
    print("Error opening video stream or file")

# Loop infinitely
idx = 0
while(True):

    # Read video stram 
    ret, frame = capture.read()

    if ret == True:

        # Draw circle in frame (only every 30th frame)
        if idx%30 == 0:
            cv2.circle(frame,(100, 100), 50, (255,0,0), 3)

        # Show
        cv2.imshow('original video', frame)
        
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

# Close all the frames
cv2.destroyAllWindows()