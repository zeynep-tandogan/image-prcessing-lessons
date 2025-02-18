import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


image1_path = "gorev2/dolma.jpg"  
image2_path = "gorev2/dolma2.jpg" 

image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

if image1 is None or image2 is None:
    print("Bir veya iki görüntü yüklenemedi. Lütfen yolları kontrol edin.")
else:
    if image2.shape[0] > image1.shape[0] or image2.shape[1] > image1.shape[1]:
        print("Şablon (image2), ana görüntüden (image1) büyük. Şablon yeniden boyutlandırılıyor.")
        scale_factor = min(image1.shape[0] / image2.shape[0], image1.shape[1] / image2.shape[1])
        new_width = int(image2.shape[1] * scale_factor)
        new_height = int(image2.shape[0] * scale_factor)
        image2 = cv2.resize(image2, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Yeni şablon boyutları: {image2.shape}")

    # Korelasyon hesaplama
    result = cv2.matchTemplate(image1, image2, cv2.TM_CCORR_NORMED)
    result_normalized = (result - np.min(result)) / (np.max(result) - np.min(result))  # Normalizasyon

    # Maksimum korelasyon indeksi
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_normalized)

    h, w = image2.shape  
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    matched_image = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(matched_image, top_left, bottom_right, (0, 255, 0), 2)

    correlation_map_path = "gorev2/correlation_map.jpg"
    cv2.imwrite(correlation_map_path, (result_normalized * 255).astype(np.uint8))

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title("Image 1")
    plt.imshow(image1, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Image 2 (Template)")
    plt.imshow(image2, cmap="gray")
    plt.axis("off")

    # Korelasyon haritasını görselleştirme
    plt.subplot(2, 2, 3)
    plt.title("Correlation Map")
    plt.imshow(result_normalized, cmap="plasma")
    plt.colorbar(label="Correlation Value")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Matched Area on Image 1")
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("gorev2/correlation_visualization.jpg")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(result_normalized, annot=True, cmap="YlOrRd")
    plt.title("Correlation Heatmap")
    plt.savefig("gorev2/correlation_heatmap.jpg")
    plt.show()

    print(f"Maksimum Korelasyon Değeri: {max_val}")
    print(f"Eşleşme Konumu: {max_loc}")
