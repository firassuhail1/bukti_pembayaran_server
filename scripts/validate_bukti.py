import argparse
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

# Fungsi untuk melakukan Error Level Analysis (ELA)
def error_level_analysis(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    temp_path = "temp_ela.jpg"
    
    # Simpan ulang dengan kualitas tertentu
    original.save(temp_path, 'JPEG', quality=quality)
    compressed = Image.open(temp_path)
    
    # Bandingkan perbedaan
    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return np.array(ela_image), max_diff

# Fungsi untuk melakukan analisis noise
def noise_analysis(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noise_map = cv2.Laplacian(image, cv2.CV_64F)  # Hasil noise dalam bentuk array
    noise_variance = noise_map.var()  # Nilai varians (angka)

    # Normalisasi noise agar bisa divisualisasikan dengan lebih baik
    noise_image = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX)
    noise_image = np.uint8(noise_image)

    return noise_variance, noise_image

# Fungsi untuk menyimpan gambar secara terpisah
def save_images_separately(original_path, ela_image, noise_image):
    # print("📢 Menyimpan gambar hasil analisis...")

    original = cv2.imread(original_path)

    if original is None:
        # print("⚠️ Gambar asli tidak ditemukan!")
        return

    try:
        cv2.imwrite(f"gambar_asli.png", original)
        # print("✅ Gambar asli disimpan")

        storage_folder = "../storage/app/public/hasil_ela"
        os.makedirs(storage_folder, exist_ok=True)

        ela_pil = Image.fromarray(ela_image.astype('uint8'))  # Pastikan tipe data uint8
        ela_pil.save(f"ela.png")
        ela_pil.save(os.path.join(storage_folder, "ela.png"))
        # print("✅ Gambar ELA disimpan")

        cv2.imwrite(f"noise.png", noise_image)
        # print("✅ Gambar Noise disimpan")

    except Exception as e:
        print(f"Gagal menyimpan gambar: {e}")

def show_images(original_path, ela_image, noise_image):
    original = cv2.imread(original_path)

    # Tampilkan gambar asli
    cv2.imshow("Gambar Asli", original)

    # Tampilkan gambar ELA
    ela_bgr = cv2.cvtColor(ela_image, cv2.COLOR_RGB2BGR)  # Convert PIL-RGB to OpenCV-BGR
    cv2.imshow("Hasil ELA", ela_bgr)

    # Tampilkan gambar noise
    cv2.imshow("Peta Noise", noise_image)

    cv2.waitKey(0)  # Tunggu sampai tombol ditekan
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path gambar bukti pembayaran")
    args = parser.parse_args()

    ela_result, ela_score = error_level_analysis(args.image_path)
    noise_variance, noise_image = noise_analysis(args.image_path)
    is_authentic = (ela_score < 30)

    # show_images(args.image_path, ela_result, noise_image)

    result = {
        "ela_score": ela_score,
        "noise_variance": noise_variance,
        "is_authentic": is_authentic,
    }

    # Simpan gambar hasil preprocessing
    save_images_separately(args.image_path, ela_result, noise_image)

    print(json.dumps(result, indent=4))

