import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = "gorev2\dolma.jpg"  
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image_path = "gray_scale.jpg"
cv2.imwrite(gray_image_path, gray_image)

hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

histogram_path = "gorev2\histogram.png"
plt.savefig(histogram_path)
print(f"Histogram {histogram_path} olarak kaydedildi.")
