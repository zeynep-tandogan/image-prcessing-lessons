import cv2

image = cv2.imread('dolma.jpg')  

median_blurred_image = cv2.medianBlur(image, 5)  

cv2.imwrite('medyan_filtresi.jpg', median_blurred_image)
