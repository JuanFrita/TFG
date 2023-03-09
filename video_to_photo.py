import cv2
# Open the video file
name = 'soleado_medio'
video = cv2.VideoCapture(f'videos_playas/{name}.mp4')
# Get the number of frames
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(num_frames)
# Get the 3 first frame of the video
for frame_num in range(5):
    # Read the frame
    ret, frame = video.read()
    # Check if the frame was successfully read
    if ret:
        # Save the frame as an image file
        cv2.imwrite(f'fotos_playas/{name}_{frame_num}.jpg', frame)
    else:
        break
# Release the video file
video.release()