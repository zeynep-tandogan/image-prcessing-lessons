import cv2
import matplotlib.pyplot as plt

# Resmi yükle
img = cv2.imread("dolma.jpg")  

r = img.copy()
r[:,:,0] = 0
r[:,:,1] = 0

g = img.copy()
g[:,:,0] = 0
g[:,:,2] = 0

b = img.copy()
b[:,:,1] = 0
b[:,:,2] = 0

cv2.imshow("original", img)
cv2.imwrite("red_channel.jpg", r)
cv2.imwrite("green_channel.jpg", g)
cv2.imwrite("blue_channel.jpg", b)
# asagıdakiler grayscale gozukuyo bu da kanalları ayırıyor ama

#b, g, r = cv2.split(img)
b = img[:, :, 0]  
g = img[:, :, 1]  
r = img[:, :, 2]

cv2.imwrite("blue_channel_gray.jpg", b)
cv2.imwrite("green_channel_gray.jpg", g)
cv2.imwrite("red_channel_gray.jpg", r)

cv2.waitKey(0)
cv2.destroyAllWindows()