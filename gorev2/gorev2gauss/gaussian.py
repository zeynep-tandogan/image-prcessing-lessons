import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

input_path = "gorev2/dolma.jpg"
output_path_method1 = "gorev2/gaussian_noisy_image_method1.jpg"
output_path_method2 = "gorev2/gaussian_noisy_image_method2.jpg"

original_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
if original_image is None:
    print("Hatalı bir görüntü yolu girdiniz.")
else:
    mean = 0
    std_dev = 25 
    gaussian_noise1 = np.random.normal(mean, std_dev, original_image.shape)
    noisy_image1 = original_image + gaussian_noise1
    noisy_image1 = np.clip(noisy_image1, 0, 255).astype(np.uint8)

    # Gürültüyü dosyaya kaydet (1. Yöntem)
    cv2.imwrite(output_path_method1, noisy_image1)

    # 2. Yöntem: Daha düşük varyanslı Gaussian Gürültüsü
    var = 0.01  # Gürültünün varyansı
    sigma = np.sqrt(var)  # Standart sapma
    gaussian_noise2 = np.random.normal(loc=mean, scale=sigma, size=original_image.shape)
    noisy_image2 = original_image + gaussian_noise2
    noisy_image2 = np.clip(noisy_image2, 0, 255).astype(np.uint8)

    #2. Yöntem
    cv2.imwrite(output_path_method2, noisy_image2)

    # Gürültü dağılımını göster (2. Yöntem)
    kde = gaussian_kde(gaussian_noise2.ravel()) 
    dist_space = np.linspace(np.min(gaussian_noise2), np.max(gaussian_noise2), 100)
    plt.figure(figsize=(6, 4))
    plt.plot(dist_space, kde(dist_space))
    plt.xlabel("Noise Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Noise Distribution (Method 2)")
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Gaussian Noise (Method 1)")
    plt.imshow(noisy_image1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Gaussian Noise (Method 2)")
    plt.imshow(noisy_image2, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Gürültülü görüntüler kaydedildi:\n1. Yöntem: {output_path_method1}\n2. Yöntem: {output_path_method2}")
