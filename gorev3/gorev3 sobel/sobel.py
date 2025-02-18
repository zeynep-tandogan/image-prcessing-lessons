import cv2
import numpy as np

image = cv2.imread('dolma.jpg') 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

sobel_y_image = np.uint8(np.absolute(sobel_y))
cv2.imwrite('sobel_dikey_kenar.jpg', sobel_y_image)
