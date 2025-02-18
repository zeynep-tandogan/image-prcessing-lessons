import cv2

image = cv2.imread('dolma.jpg')

blurred_image = cv2.GaussianBlur(image, (15, 15), 0)  

cv2.imwrite('bulanık_görsel.jpg', blurred_image)
