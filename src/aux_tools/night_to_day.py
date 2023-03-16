import cv2
import numpy as np

# Load the image
img = cv2.imread('C:/Users/juanf/OneDrive/Escritorio/TFG/resources/fotos_playas/noche/noche_0.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization to improve contrast
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray)

# Convert the image to LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Split the LAB image into channels
L, A, B = cv2.split(lab)

# Apply adaptive thresholding to the L channel
thresh = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply a mask to the A and B channels
A_masked = cv2.bitwise_and(A, thresh)
B_masked = cv2.bitwise_and(B, thresh)

# Merge the masked channels back into the LAB image
lab_masked = cv2.merge((L, A_masked, B_masked))

# Convert the masked LAB image back to BGR color space
final = cv2.cvtColor(lab_masked, cv2.COLOR_LAB2BGR)

# Show the final image
cv2.imshow('Daytime Image', final)
cv2.waitKey(0)
cv2.destroyAllWindows()