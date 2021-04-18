import cv2

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
output = cv2.VideoWriter('output.avi',fourcc, fps, (frame_width,frame_height))

# Loop infinitely
while(True):

    # Read video stram 
    ret, frame = capture.read()

    if ret == True:

        # Write the frame into the file 'output.avi'
        output.write(frame)
            
        # convert to grey scale (as an exercise)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Show
        cv2.imshow('original video', frame)
        cv2.imshow('gray video', gray)
        
        # Break eveything when "ESC" is pressed
        if cv2.waitKey(1) == 27:
            break
    
    # Else, break
    else:
        break

# When everything is done, release video caputre and video writer objects
capture.release()
output.release()

# Close all the frames
cv2.destroyAllWindows()