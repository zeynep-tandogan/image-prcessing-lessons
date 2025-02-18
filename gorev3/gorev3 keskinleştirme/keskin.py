import cv2
import numpy as np

image = cv2.imread('dolma.jpg')  # Görsel dosyasının yolunu belirtin
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

sharpened_image = cv2.filter2D(image, -1, kernel)

cv2.imwrite('keskinleştirilmiş_görsel.jpg', sharpened_image)
