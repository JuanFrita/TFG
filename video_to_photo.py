import cv2
# Open the video file
video = cv2.VideoCapture('videos_playas/soleado_alto.mp4')
# Get the number of frames
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# Loop through the frames and save each one as an image file
for frame_num in range(num_frames):
    # Read the frame
    ret, frame = video.read()
    # Check if the frame was successfully read
    if ret:
        # Save the frame as an image file
        cv2.imwrite(f'/fotos_playas/soleado_alto_{frame_num}.jpg', frame)
    else:
        break
# Release the video file
video.release()