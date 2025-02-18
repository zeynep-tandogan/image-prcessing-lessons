import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def load_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))  
        else:
            print(f"Error loading image: {path}")
    return images


def shi_tomasi_corner_detection(image, max_corners=100, quality_level=0.01, min_distance=10):
    corners = cv2.goodFeaturesToTrack(image, max_corners, quality_level, min_distance)
    corners = np.int0(corners)  
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image_color, (x, y), 3, (0, 0, 255), -1)  
    return image_color


def harris_corner_detection(image, block_size=2, ksize=3, k=0.04):
    dst = cv2.cornerHarris(np.float32(image), block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
   
    image_color[dst > 0.01 * dst.max()] = [0, 0, 255]
    return image_color

def modified_corner_detection(image, block_size=2, ksize=3, alpha=0.04):
    gray = np.float32(image)
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    height, width = gray.shape
    offset = block_size // 2
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()
            Sxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()
            Syy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()

            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            r = det - alpha * (trace ** 2)

            if r > 0.01 * det:
                cv2.circle(image_color, (x, y), 3, (0, 0, 255), -1) 

    return image_color

# Görselleri yükleme
image_paths = ["gorev4/chessboard.png", "gorev4/cubes.png", "gorev4/baklava.jpg"]  
images = load_images(image_paths)

for i, img in enumerate(images):
    plt.figure(figsize=(15, 5))

    # Shi-Tomasi
    start = time.time()
    shi_tomasi_result = shi_tomasi_corner_detection(img.copy())
    end = time.time()
    plt.subplot(1, 3, 1)
    plt.title(f"Shi-Tomasi ({end - start:.2f} s)")
    plt.imshow(cv2.cvtColor(shi_tomasi_result, cv2.COLOR_BGR2RGB))

    # Harris
    start = time.time()
    harris_result = harris_corner_detection(img.copy())
    end = time.time()
    plt.subplot(1, 3, 2)
    plt.title(f"Harris ({end - start:.2f} s)")
    plt.imshow(cv2.cvtColor(harris_result, cv2.COLOR_BGR2RGB))

   
    start = time.time()
    modified_result = modified_corner_detection(img.copy())
    end = time.time()
    plt.subplot(1, 3, 3)
    plt.title(f"Modified ({end - start:.2f} s)")
    plt.imshow(cv2.cvtColor(modified_result, cv2.COLOR_BGR2RGB))

    plt.show()
