# import cv2
# import numpy as np
# import piexif
# from PIL import Image, ImageChops, ImageEnhance
# import argparse
# import json
# import os

# def check_metadata(image_path):
#     """Mengecek metadata EXIF gambar"""
#     try:
#         exif_data = piexif.load(image_path)
#         return exif_data != {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
#     except Exception:
#         return False

# def error_level_analysis(image_path):
#     """Menganalisis Error Level Analysis (ELA) untuk mendeteksi editan"""
#     original = Image.open(image_path).convert("RGB")
#     temp_path = "temp_ela.jpg"
#     original.save(temp_path, "JPEG", quality=90)
    
#     recompressed = Image.open(temp_path)
#     ela_image = ImageChops.difference(original, recompressed)
#     extrema = ela_image.getextrema()
    
#     max_diff = max([ex[1] for ex in extrema])
#     scale = 255.0 / max_diff if max_diff > 0 else 1
#     ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
#     ela_array = np.array(ela_image)

#     # Hapus file sementara
#     if os.path.exists(temp_path):
#         os.remove(temp_path)
    
#     return ela_array

# def noise_analysis(image_path):
#     """Menganalisis noise pada gambar untuk mendeteksi manipulasi"""
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     noise = cv2.Laplacian(image, cv2.CV_64F).var()
#     return noise  # Semakin rendah nilai, semakin sedikit noise

# def validate_image(image_path):
#     """Fungsi utama untuk validasi keaslian bukti pembayaran"""
#     result = {
#         "metadata_found": check_metadata(image_path),
#         "ela_analysis": float(np.mean(error_level_analysis(image_path))),
#         "noise_variance": float(noise_analysis(image_path)),
#     }
    
#     # Logika untuk menentukan apakah gambar asli atau editan
#     is_authentic = (
#         result["metadata_found"] and
#         result["ela_analysis"] < 50 and
#         result["noise_variance"] > 50
#     )

#     result["is_authentic"] = is_authentic

#     # Output dalam bentuk JSON
#     print(json.dumps(result, indent=4))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("image_path", help="Path gambar bukti pembayaran")
#     args = parser.parse_args()

#     validate_image(args.image_path)


# import argparse
# import json
# import cv2
# import numpy as np
# from PIL import Image, ImageChops, ImageEnhance

# # Fungsi untuk melakukan Error Level Analysis (ELA)
# def error_level_analysis(image_path, quality=90):
#     original = Image.open(image_path).convert('RGB')
#     temp_path = "temp_ela.jpg"
    
#     # Simpan ulang dengan kualitas tertentu
#     original.save(temp_path, 'JPEG', quality=quality)
#     compressed = Image.open(temp_path)
    
#     # Bandingkan perbedaan
#     ela_image = ImageChops.difference(original, compressed)
#     extrema = ela_image.getextrema()
#     max_diff = max([ex[1] for ex in extrema])
    
#     if max_diff == 0:
#         max_diff = 1
#     scale = 255.0 / max_diff
#     ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
#     return np.array(ela_image), max_diff

# # Fungsi untuk melakukan analisis noise
# def noise_analysis(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     noise = cv2.Laplacian(image, cv2.CV_64F).var()
#     return noise

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("image_path", help="Path gambar bukti pembayaran")
#     args = parser.parse_args()

#     ela_result, ela_score = error_level_analysis(args.image_path)
#     noise_variance = noise_analysis(args.image_path)
#     is_authentic = (ela_score < 30)
#     result = {
#         "ela_score": ela_score,
#         "noise_variance": noise_variance,
#         "is_authentic": is_authentic,
#     }

#     print(json.dumps(result, indent=4))