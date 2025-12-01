# import argparse
# import json
# import os
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
#     noise_map = cv2.Laplacian(image, cv2.CV_64F)  # Hasil noise dalam bentuk array
#     noise_variance = noise_map.var()  # Nilai varians (angka)

#     # Normalisasi noise agar bisa divisualisasikan dengan lebih baik
#     noise_image = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX)
#     noise_image = np.uint8(noise_image)

#     return noise_variance, noise_image

# # Fungsi untuk menyimpan gambar secara terpisah
# def save_images_separately(original_path, ela_image, noise_image):
#     # print("📢 Menyimpan gambar hasil analisis...")

#     original = cv2.imread(original_path)

#     if original is None:
#         # print("⚠️ Gambar asli tidak ditemukan!")
#         return

#     try:
#         cv2.imwrite(f"gambar_asli.png", original)
#         # print("✅ Gambar asli disimpan")

#         storage_folder = "../storage/app/public/hasil_ela"
#         os.makedirs(storage_folder, exist_ok=True)

#         ela_pil = Image.fromarray(ela_image.astype('uint8'))  # Pastikan tipe data uint8
#         ela_pil.save(f"ela.png")
#         ela_pil.save(os.path.join(storage_folder, "ela.png"))
#         # print("✅ Gambar ELA disimpan")

#         cv2.imwrite(f"noise.png", noise_image)
#         # print("✅ Gambar Noise disimpan")

#     except Exception as e:
#         print(f"Gagal menyimpan gambar: {e}")

# def show_images(original_path, ela_image, noise_image):
#     original = cv2.imread(original_path)

#     # Tampilkan gambar asli
#     cv2.imshow("Gambar Asli", original)

#     # Tampilkan gambar ELA
#     ela_bgr = cv2.cvtColor(ela_image, cv2.COLOR_RGB2BGR)  # Convert PIL-RGB to OpenCV-BGR
#     cv2.imshow("Hasil ELA", ela_bgr)

#     # Tampilkan gambar noise
#     cv2.imshow("Peta Noise", noise_image)

#     cv2.waitKey(0)  # Tunggu sampai tombol ditekan
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("image_path", help="Path gambar bukti pembayaran")
#     args = parser.parse_args()

#     ela_result, ela_score = error_level_analysis(args.image_path)
#     noise_variance, noise_image = noise_analysis(args.image_path)
#     is_authentic = (ela_score < 30)

#     # show_images(args.image_path, ela_result, noise_image)

#     result = {
#         "ela_score": ela_score,
#         "noise_variance": noise_variance,
#         "is_authentic": is_authentic,
#     }

#     # Simpan gambar hasil preprocessing
#     save_images_separately(args.image_path, ela_result, noise_image)

#     print(json.dumps(result, indent=4))


# import argparse
# import json
# import os
# import cv2
# import numpy as np
# from PIL import Image, ImageChops, ImageEnhance
# from scipy import ndimage
# from skimage.feature import local_binary_pattern

# # ========== FUNGSI ELA (Error Level Analysis) ==========
# def error_level_analysis(image_path, quality=90):
#     original = Image.open(image_path).convert('RGB')
#     temp_path = "temp_ela.jpg"
    
#     original.save(temp_path, 'JPEG', quality=quality)
#     compressed = Image.open(temp_path)
    
#     ela_image = ImageChops.difference(original, compressed)
#     extrema = ela_image.getextrema()
#     max_diff = max([ex[1] for ex in extrema])
    
#     if max_diff == 0:
#         max_diff = 1
#     scale = 255.0 / max_diff
#     ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
#     ela_array = np.array(ela_image)
    
#     # Hitung ELA score dengan median (lebih robust dari max)
#     ela_median = np.median(ela_array)
    
#     # Cleanup
#     if os.path.exists(temp_path):
#         os.remove(temp_path)
    
#     return ela_array, ela_median, max_diff

# # ========== ANALISIS NOISE ADVANCED ==========
# def noise_analysis_advanced(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # 1. High-frequency noise (Laplacian)
#     laplacian = cv2.Laplacian(image, cv2.CV_64F)
#     noise_variance = laplacian.var()
    
#     # 2. Local noise consistency
#     # Gambar asli cenderung punya noise konsisten di seluruh area
#     # Gambar edit punya inkonsistensi noise
#     local_std = ndimage.generic_filter(image, np.std, size=15)
#     noise_consistency = np.std(local_std)
    
#     # Normalisasi untuk visualisasi
#     noise_image = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
#     noise_image = np.uint8(noise_image)
    
#     return {
#         'variance': noise_variance,
#         'consistency': noise_consistency,
#         'image': noise_image
#     }

# # ========== ANALISIS BLOCK ARTIFACTS (JPEG Compression) ==========
# def detect_block_artifacts(image_path):
#     """
#     Deteksi inkonsistensi block JPEG 8x8.
#     Gambar yang diedit biasanya punya pola block yang tidak natural
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     h, w = image.shape
    
#     # Analisis grid 8x8 (standar JPEG)
#     block_scores = []
    
#     for i in range(0, h - 8, 8):
#         for j in range(0, w - 8, 8):
#             block = image[i:i+8, j:j+8]
            
#             # DCT transform
#             block_float = np.float32(block)
#             dct = cv2.dct(block_float)
            
#             # Hitung energi high-frequency
#             high_freq_energy = np.sum(np.abs(dct[4:, 4:]))
#             block_scores.append(high_freq_energy)
    
#     # Inkonsistensi block
#     block_variance = np.var(block_scores)
    
#     return block_variance

# # ========== ANALISIS TEXTURE LBP (Local Binary Pattern) ==========
# def texture_consistency_analysis(image_path):
#     """
#     Analisis konsistensi texture menggunakan LBP.
#     Area yang diedit biasanya punya texture pattern berbeda
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # LBP parameters
#     radius = 3
#     n_points = 8 * radius
    
#     lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
#     # Hitung histogram LBP di beberapa region
#     h, w = lbp.shape
#     regions = [
#         lbp[0:h//2, 0:w//2],
#         lbp[0:h//2, w//2:w],
#         lbp[h//2:h, 0:w//2],
#         lbp[h//2:h, w//2:w]
#     ]
    
#     histograms = [np.histogram(region, bins=50)[0] for region in regions]
    
#     # Hitung similarity antar region (cosine similarity)
#     similarities = []
#     for i in range(len(histograms)):
#         for j in range(i + 1, len(histograms)):
#             h1 = histograms[i] / (np.linalg.norm(histograms[i]) + 1e-6)
#             h2 = histograms[j] / (np.linalg.norm(histograms[j]) + 1e-6)
#             similarity = np.dot(h1, h2)
#             similarities.append(similarity)
    
#     # Gambar asli cenderung punya texture lebih konsisten
#     texture_consistency = np.mean(similarities)
    
#     return texture_consistency

# # ========== ANALISIS REGIONAL ELA ==========
# def regional_ela_analysis(ela_image):
#     """
#     Analisis ELA per region untuk deteksi manipulasi lokal
#     """
#     h, w = ela_image.shape[:2]
    
#     # Bagi jadi 9 region (3x3 grid)
#     region_scores = []
    
#     for i in range(3):
#         for j in range(3):
#             y1 = i * h // 3
#             y2 = (i + 1) * h // 3
#             x1 = j * w // 3
#             x2 = (j + 1) * w // 3
            
#             region = ela_image[y1:y2, x1:x2]
#             region_mean = np.mean(region)
#             region_scores.append(region_mean)
    
#     # Hitung variance antar region
#     # Gambar yang diedit sebagian akan punya variance tinggi
#     regional_variance = np.var(region_scores)
#     max_region_diff = max(region_scores) - min(region_scores)
    
#     return {
#         'variance': regional_variance,
#         'max_diff': max_region_diff
#     }

# # ========== DETEKSI CLONE/COPY-PASTE ==========
# def detect_cloning(image_path):
#     """
#     Deteksi copy-paste menggunakan feature matching
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # SIFT detector
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(image, None)
    
#     if descriptors is None or len(descriptors) < 2:
#         return 0
    
#     # Self-matching untuk deteksi region duplikat
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(descriptors, descriptors, k=3)
    
#     # Hitung self-similar regions (exclude perfect self-match)
#     similar_count = 0
#     for match_list in matches:
#         if len(match_list) >= 3:
#             # Skip perfect match dengan diri sendiri
#             distances = [m.distance for m in match_list[1:]]
#             if min(distances) < 50:  # Threshold similarity
#                 similar_count += 1
    
#     clone_score = similar_count / max(len(keypoints), 1)
    
#     return clone_score

# # ========== SCORING SYSTEM ==========
# def calculate_authenticity_score(metrics):
#     """
#     Sistem scoring multi-faktor dengan bobot
#     """
#     scores = []
#     weights = []
    
#     # 1. ELA Score (bobot rendah karena prone to false positive)
#     ela_normalized = min(metrics['ela_median'] / 50, 1.0)
#     scores.append(1 - ela_normalized)
#     weights.append(0.15)
    
#     # 2. Noise Consistency (bobot tinggi)
#     noise_norm = min(metrics['noise_consistency'] / 30, 1.0)
#     scores.append(1 - noise_norm)
#     weights.append(0.25)
    
#     # 3. Block Artifacts
#     block_norm = min(metrics['block_variance'] / 10000, 1.0)
#     scores.append(1 - block_norm)
#     weights.append(0.15)
    
#     # 4. Texture Consistency
#     scores.append(metrics['texture_consistency'])
#     weights.append(0.20)
    
#     # 5. Regional ELA Variance
#     regional_norm = min(metrics['regional_variance'] / 1000, 1.0)
#     scores.append(1 - regional_norm)
#     weights.append(0.15)
    
#     # 6. Clone Detection
#     scores.append(1 - min(metrics['clone_score'], 1.0))
#     weights.append(0.10)
    
#     # Weighted average
#     final_score = sum(s * w for s, w in zip(scores, weights)) * 100
    
#     return final_score

# # ========== PENYIMPANAN GAMBAR ==========
# def save_images_separately(original_path, ela_image, noise_image):
#     original = cv2.imread(original_path)
    
#     if original is None:
#         return
    
#     try:
#         cv2.imwrite("gambar_asli.png", original)
        
#         storage_folder = "../storage/app/public/hasil_ela"
#         os.makedirs(storage_folder, exist_ok=True)
        
#         ela_pil = Image.fromarray(ela_image.astype('uint8'))
#         ela_pil.save("ela.png")
#         ela_pil.save(os.path.join(storage_folder, "ela.png"))
        
#         cv2.imwrite("noise.png", noise_image)
        
#     except Exception as e:
#         print(f"Error saving images: {e}")

# # ========== MAIN PROGRAM ==========
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("image_path", help="Path gambar bukti pembayaran")
#     parser.add_argument("--verbose", action="store_true", help="Show detailed metrics")
#     args = parser.parse_args()
    
#     # Jalankan semua analisis
#     ela_result, ela_median, ela_max = error_level_analysis(args.image_path)
#     noise_data = noise_analysis_advanced(args.image_path)
#     block_variance = detect_block_artifacts(args.image_path)
#     texture_consistency = texture_consistency_analysis(args.image_path)
#     regional_ela = regional_ela_analysis(ela_result)
#     clone_score = detect_cloning(args.image_path)
    
#     # Kumpulkan semua metrics
#     metrics = {
#         'ela_median': ela_median,
#         'ela_max': ela_max,
#         'noise_consistency': noise_data['consistency'],
#         'block_variance': block_variance,
#         'texture_consistency': texture_consistency,
#         'regional_variance': regional_ela['variance'],
#         'regional_max_diff': regional_ela['max_diff'],
#         'clone_score': clone_score
#     }
    
#     # Hitung skor akhir
#     authenticity_score = calculate_authenticity_score(metrics)
    
#     # Tentukan keaslian (threshold 60%)
#     is_authentic = authenticity_score >= 60
    
#     # Tentukan confidence level
#     if authenticity_score >= 75:
#         confidence = "High"
#     elif authenticity_score >= 50:
#         confidence = "Medium"
#     else:
#         confidence = "Low"
    
#     # Deteksi suspicious patterns
#     red_flags = []
#     if metrics['regional_variance'] > 500:
#         red_flags.append("Inkonsistensi ELA regional terdeteksi")
#     if metrics['noise_consistency'] > 25:
#         red_flags.append("Pola noise tidak konsisten")
#     if metrics['clone_score'] > 0.3:
#         red_flags.append("Kemungkinan copy-paste terdeteksi")
#     if metrics['texture_consistency'] < 0.5:
#         red_flags.append("Texture pattern tidak natural")
    
#     # Result
#     # result = {
#     #     "authenticity_score": round(authenticity_score, 2),
#     #     "is_authentic": is_authentic,
#     #     "confidence": confidence,
#     #     "red_flags": red_flags,
#     #     "metrics": {
#     #         "ela_score": round(ela_median, 2),
#     #         "noise_consistency": round(noise_data['consistency'], 2),
#     #         "texture_consistency": round(texture_consistency, 2),
#     #         "clone_detection": round(clone_score, 2)
#     #     } if args.verbose else {}
#     # }

#     result = {
#         "authenticity_score": round(float(authenticity_score), 2), # Pastikan float
#         "is_authentic": bool(is_authentic),                         # Pastikan bool
#         # "confidence": float(confidence),                             # Pastikan float
#         # "red_flags": list(red_flags),                                # Pastikan list
#         "metrics": {
#             "ela_score": round(float(ela_median), 2),
#             "noise_consistency": round(float(noise_data['consistency']), 2),
#             "texture_consistency": round(float(texture_consistency), 2),
#             "clone_detection": round(float(clone_score), 2)
#         } if args.verbose else {}
#     }
    
#     # Simpan gambar hasil preprocessing
#     save_images_separately(args.image_path, ela_result, noise_data['image'])
    
#     print(json.dumps(result, indent=4))






# import argparse
# import json
# import os
# import cv2
# import numpy as np
# from PIL import Image, ImageChops, ImageEnhance
# from scipy import ndimage
# from skimage.feature import local_binary_pattern

# # ========== FUNGSI ELA (Error Level Analysis) ==========
# def error_level_analysis(image_path, quality=90):
#     original = Image.open(image_path).convert('RGB')
#     temp_path = "temp_ela.jpg"
    
#     original.save(temp_path, 'JPEG', quality=quality)
#     compressed = Image.open(temp_path)
    
#     ela_image = ImageChops.difference(original, compressed)
#     extrema = ela_image.getextrema()
#     max_diff = max([ex[1] for ex in extrema])
    
#     if max_diff == 0:
#         max_diff = 1
#     scale = 255.0 / max_diff
#     ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
#     ela_array = np.array(ela_image)
    
#     # Gunakan percentile 95 (lebih sensitif dari median)
#     ela_p95 = np.percentile(ela_array, 95)
#     ela_mean = np.mean(ela_array)
    
#     # Cleanup
#     if os.path.exists(temp_path):
#         os.remove(temp_path)
    
#     return ela_array, ela_mean, ela_p95, max_diff

# # ========== ANALISIS NOISE ADVANCED ==========
# def noise_analysis_advanced(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # 1. High-frequency noise (Laplacian)
#     laplacian = cv2.Laplacian(image, cv2.CV_64F)
#     noise_variance = laplacian.var()
    
#     # 2. Local noise consistency - LEBIH SENSITIF
#     # Hitung std deviation di window kecil
#     local_std = ndimage.generic_filter(image, np.std, size=15)
#     noise_consistency = np.std(local_std)
    
#     # 3. Hitung coefficient of variation
#     noise_cv = noise_consistency / (np.mean(local_std) + 1e-6)
    
#     # Normalisasi untuk visualisasi
#     noise_image = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
#     noise_image = np.uint8(noise_image)
    
#     return {
#         'variance': noise_variance,
#         'consistency': noise_consistency,
#         'cv': noise_cv,
#         'image': noise_image
#     }

# # ========== ANALISIS BLOCK ARTIFACTS (JPEG Compression) ==========
# def detect_block_artifacts(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     h, w = image.shape
    
#     block_scores = []
    
#     for i in range(0, h - 8, 8):
#         for j in range(0, w - 8, 8):
#             block = image[i:i+8, j:j+8]
#             block_float = np.float32(block)
#             dct = cv2.dct(block_float)
            
#             # Hitung energi high-frequency
#             high_freq_energy = np.sum(np.abs(dct[4:, 4:]))
#             block_scores.append(high_freq_energy)
    
#     # Inkonsistensi block (lebih tinggi = lebih mencurigakan)
#     block_variance = np.var(block_scores)
#     block_cv = np.std(block_scores) / (np.mean(block_scores) + 1e-6)
    
#     return {
#         'variance': block_variance,
#         'cv': block_cv
#     }

# # ========== ANALISIS TEXTURE LBP ==========
# def texture_consistency_analysis(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     radius = 3
#     n_points = 8 * radius
    
#     lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
#     # Bagi jadi 9 region (3x3)
#     h, w = lbp.shape
#     regions = []
#     for i in range(3):
#         for j in range(3):
#             y1 = i * h // 3
#             y2 = (i + 1) * h // 3
#             x1 = j * w // 3
#             x2 = (j + 1) * w // 3
#             regions.append(lbp[y1:y2, x1:x2])
    
#     histograms = [np.histogram(region, bins=50, range=(0, 50))[0] for region in regions]
    
#     # Hitung similarity dengan metode lebih strict
#     similarities = []
#     for i in range(len(histograms)):
#         for j in range(i + 1, len(histograms)):
#             h1 = histograms[i] / (np.linalg.norm(histograms[i]) + 1e-6)
#             h2 = histograms[j] / (np.linalg.norm(histograms[j]) + 1e-6)
#             similarity = np.dot(h1, h2)
#             similarities.append(similarity)
    
#     texture_consistency = np.mean(similarities)
#     texture_variance = np.var(similarities)
    
#     return {
#         'consistency': texture_consistency,
#         'variance': texture_variance
#     }

# # ========== ANALISIS REGIONAL ELA ==========
# def regional_ela_analysis(ela_image):
#     h, w = ela_image.shape[:2]
    
#     # Bagi jadi 16 region (4x4) - lebih detail
#     region_scores = []
    
#     for i in range(4):
#         for j in range(4):
#             y1 = i * h // 4
#             y2 = (i + 1) * h // 4
#             x1 = j * w // 4
#             x2 = (j + 1) * w // 4
            
#             region = ela_image[y1:y2, x1:x2]
#             region_mean = np.mean(region)
#             region_scores.append(region_mean)
    
#     regional_variance = np.var(region_scores)
#     max_region_diff = max(region_scores) - min(region_scores)
#     regional_cv = np.std(region_scores) / (np.mean(region_scores) + 1e-6)
    
#     return {
#         'variance': regional_variance,
#         'max_diff': max_region_diff,
#         'cv': regional_cv
#     }

# # ========== DETEKSI EDGE INCONSISTENCY ==========
# def detect_edge_inconsistency(image_path):
#     """
#     Gambar asli punya edge yang konsisten.
#     Gambar edit sering punya edge yang tidak natural di area manipulasi.
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Deteksi edges
#     edges = cv2.Canny(image, 50, 150)
    
#     # Hitung edge density per region
#     h, w = edges.shape
#     edge_densities = []
    
#     for i in range(4):
#         for j in range(4):
#             y1 = i * h // 4
#             y2 = (i + 1) * h // 4
#             x1 = j * w // 4
#             x2 = (j + 1) * w // 4
            
#             region = edges[y1:y2, x1:x2]
#             density = np.sum(region > 0) / region.size
#             edge_densities.append(density)
    
#     edge_variance = np.var(edge_densities)
#     edge_cv = np.std(edge_densities) / (np.mean(edge_densities) + 1e-6)
    
#     return {
#         'variance': edge_variance,
#         'cv': edge_cv
#     }

# # ========== DETEKSI CLONE/COPY-PASTE ==========
# def detect_cloning(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(image, None)
    
#     if descriptors is None or len(descriptors) < 2:
#         return 0
    
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(descriptors, descriptors, k=3)
    
#     similar_count = 0
#     for match_list in matches:
#         if len(match_list) >= 3:
#             distances = [m.distance for m in match_list[1:]]
#             if min(distances) < 40:  # Threshold lebih ketat
#                 similar_count += 1
    
#     clone_score = similar_count / max(len(keypoints), 1)
    
#     return clone_score

# # ========== SCORING SYSTEM BARU (LEBIH STRICT) ==========
# def calculate_authenticity_score(metrics):
#     """
#     Sistem scoring yang LEBIH KETAT dan SENSITIF terhadap anomali
#     """
    
#     # PENALTY SYSTEM - Setiap anomali mengurangi score
#     base_score = 100.0
    
#     # 1. ELA Analysis (threshold lebih ketat)
#     if metrics['ela_mean'] > 15:  # Dulu 50
#         penalty = min((metrics['ela_mean'] - 15) / 35 * 20, 20)
#         base_score -= penalty
    
#     if metrics['ela_p95'] > 40:  # P95 tinggi = sangat mencurigakan
#         penalty = min((metrics['ela_p95'] - 40) / 60 * 15, 15)
#         base_score -= penalty
    
#     # 2. Noise Consistency (PENTING)
#     if metrics['noise_consistency'] > 15:  # Dulu 30
#         penalty = min((metrics['noise_consistency'] - 15) / 15 * 25, 25)
#         base_score -= penalty
    
#     if metrics['noise_cv'] > 0.3:  # Coefficient of variation tinggi
#         penalty = min((metrics['noise_cv'] - 0.3) / 0.4 * 10, 10)
#         base_score -= penalty
    
#     # 3. Regional ELA (SANGAT PENTING untuk deteksi edit lokal)
#     if metrics['regional_variance'] > 300:  # Dulu 500
#         penalty = min((metrics['regional_variance'] - 300) / 700 * 20, 20)
#         base_score -= penalty
    
#     if metrics['regional_cv'] > 0.5:
#         penalty = min((metrics['regional_cv'] - 0.5) / 0.5 * 15, 15)
#         base_score -= penalty
    
#     # 4. Block Artifacts
#     if metrics['block_cv'] > 0.8:
#         penalty = min((metrics['block_cv'] - 0.8) / 0.5 * 12, 12)
#         base_score -= penalty
    
#     # 5. Texture Consistency (rendah = masalah)
#     if metrics['texture_consistency'] < 0.65:  # Dulu 0.5
#         penalty = (0.65 - metrics['texture_consistency']) / 0.65 * 15
#         base_score -= penalty
    
#     if metrics['texture_variance'] > 0.02:
#         penalty = min((metrics['texture_variance'] - 0.02) / 0.03 * 10, 10)
#         base_score -= penalty
    
#     # 6. Edge Inconsistency
#     if metrics['edge_cv'] > 0.6:
#         penalty = min((metrics['edge_cv'] - 0.6) / 0.5 * 12, 12)
#         base_score -= penalty
    
#     # 7. Clone Detection (STRICT)
#     if metrics['clone_score'] > 0.15:  # Dulu 0.3
#         penalty = min((metrics['clone_score'] - 0.15) / 0.35 * 18, 18)
#         base_score -= penalty
    
#     # Pastikan score tidak negatif
#     final_score = max(base_score, 0)
    
#     return final_score

# # ========== PENYIMPANAN GAMBAR ==========
# def save_images_separately(original_path, ela_image, noise_image):
#     original = cv2.imread(original_path)
    
#     if original is None:
#         return
    
#     try:
#         cv2.imwrite("gambar_asli.png", original)
        
#         storage_folder = "../storage/app/public/hasil_ela"
#         os.makedirs(storage_folder, exist_ok=True)
        
#         ela_pil = Image.fromarray(ela_image.astype('uint8'))
#         ela_pil.save("ela.png")
#         ela_pil.save(os.path.join(storage_folder, "ela.png"))
        
#         cv2.imwrite("noise.png", noise_image)
        
#     except Exception as e:
#         print(f"Error saving images: {e}")

# # ========== MAIN PROGRAM ==========
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("image_path", help="Path gambar bukti pembayaran")
#     parser.add_argument("--verbose", action="store_true", help="Show detailed metrics")
#     args = parser.parse_args()
    
#     # Jalankan semua analisis
#     ela_result, ela_mean, ela_p95, ela_max = error_level_analysis(args.image_path)
#     noise_data = noise_analysis_advanced(args.image_path)
#     block_data = detect_block_artifacts(args.image_path)
#     texture_data = texture_consistency_analysis(args.image_path)
#     regional_ela = regional_ela_analysis(ela_result)
#     edge_data = detect_edge_inconsistency(args.image_path)
#     clone_score = detect_cloning(args.image_path)
    
#     # Kumpulkan semua metrics
#     metrics = {
#         'ela_mean': ela_mean,
#         'ela_p95': ela_p95,
#         'ela_max': ela_max,
#         'noise_consistency': noise_data['consistency'],
#         'noise_cv': noise_data['cv'],
#         'block_variance': block_data['variance'],
#         'block_cv': block_data['cv'],
#         'texture_consistency': texture_data['consistency'],
#         'texture_variance': texture_data['variance'],
#         'regional_variance': regional_ela['variance'],
#         'regional_cv': regional_ela['cv'],
#         'edge_variance': edge_data['variance'],
#         'edge_cv': edge_data['cv'],
#         'clone_score': clone_score
#     }
    
#     # Hitung skor akhir dengan sistem BARU
#     authenticity_score = calculate_authenticity_score(metrics)
    
#     # THRESHOLD BARU - LEBIH KETAT
#     if authenticity_score >= 80:
#         status = "ASLI"
#         confidence = "High"
#         is_authentic = True
#     elif authenticity_score >= 60:
#         status = "KEMUNGKINAN ASLI"
#         confidence = "Medium"
#         is_authentic = True
#     elif authenticity_score >= 45:
#         status = "MENCURIGAKAN"
#         confidence = "Low"
#         is_authentic = False
#     else:
#         status = "PALSU"
#         confidence = "High"
#         is_authentic = False
    
#     # Deteksi suspicious patterns dengan threshold lebih ketat
#     red_flags = []
    
#     if metrics['regional_variance'] > 300:
#         red_flags.append(f"⚠️ Inkonsistensi ELA regional (variance: {metrics['regional_variance']:.1f})")
    
#     if metrics['noise_consistency'] > 15:
#         red_flags.append(f"⚠️ Pola noise tidak konsisten ({metrics['noise_consistency']:.1f})")
    
#     if metrics['clone_score'] > 0.15:
#         red_flags.append(f"⚠️ Kemungkinan copy-paste terdeteksi ({metrics['clone_score']:.3f})")
    
#     if metrics['texture_consistency'] < 0.65:
#         red_flags.append(f"⚠️ Texture pattern tidak natural ({metrics['texture_consistency']:.3f})")
    
#     if metrics['edge_cv'] > 0.6:
#         red_flags.append(f"⚠️ Edge inconsistency terdeteksi ({metrics['edge_cv']:.3f})")
    
#     if metrics['ela_p95'] > 40:
#         red_flags.append(f"⚠️ ELA P95 tinggi ({metrics['ela_p95']:.1f})")
    
#     if metrics['regional_cv'] > 0.5:
#         red_flags.append(f"⚠️ Variasi regional tinggi ({metrics['regional_cv']:.3f})")
    
#     # Result
#     result = {
#         "authenticity_score": round(authenticity_score, 2),
#         # "status": status,
#         "is_authentic": is_authentic,
#         # "confidence": confidence,
#         # "red_flags": red_flags,
#         "metrics": {
#             "ela_mean": round(ela_mean, 2),
#             "ela_p95": round(ela_p95, 2),
#             "noise_consistency": round(noise_data['consistency'], 2),
#             "regional_variance": round(regional_ela['variance'], 2),
#             "texture_consistency": round(texture_data['consistency'], 3),
#             "edge_cv": round(edge_data['cv'], 3),
#             "clone_score": round(clone_score, 3)
#         } if args.verbose else {}
#     }
    
#     # Simpan gambar hasil preprocessing
#     save_images_separately(args.image_path, ela_result, noise_data['image'])
    
#     print(json.dumps(result, indent=4))


# import argparse
# import json
# import os
# import cv2
# import numpy as np
# from PIL import Image, ImageChops, ImageEnhance
# import pytesseract
# import re

# # ========== DETEKSI AREA TEKS/ANGKA ==========
# def detect_text_regions(image_path):
#     """
#     Deteksi area yang mengandung teks dan angka (terutama nominal)
#     """
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # OCR untuk deteksi posisi teks
#     try:
#         # Deteksi semua teks dengan bounding box
#         ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='ind+eng')
        
#         text_regions = []
#         number_regions = []
        
#         for i in range(len(ocr_data['text'])):
#             text = ocr_data['text'][i].strip()
#             conf = int(ocr_data['conf'][i])
            
#             # Hanya ambil text dengan confidence > 30
#             if conf > 30 and text:
#                 x = ocr_data['left'][i]
#                 y = ocr_data['top'][i]
#                 w = ocr_data['width'][i]
#                 h = ocr_data['height'][i]
                
#                 # Expand bounding box sedikit (buffer 5px)
#                 x = max(0, x - 5)
#                 y = max(0, y - 5)
#                 w = w + 10
#                 h = h + 10
                
#                 # Cek apakah mengandung angka (NOMINAL)
#                 has_digit = bool(re.search(r'\d', text))
                
#                 if has_digit:
#                     # Prioritas tinggi untuk nominal
#                     number_regions.append({
#                         'box': (x, y, w, h),
#                         'text': text,
#                         'type': 'number'
#                     })
#                 else:
#                     # Teks biasa (nama, tanggal, dll)
#                     text_regions.append({
#                         'box': (x, y, w, h),
#                         'text': text,
#                         'type': 'text'
#                     })
        
#         return number_regions, text_regions
        
#     except Exception as e:
#         print(f"OCR Error: {e}")
#         return [], []

# # ========== BUAT MASK UNTUK AREA TEKS ==========
# def create_text_mask(image_shape, regions, expand_margin=20):
#     """
#     Buat mask untuk fokus analisis pada area teks saja
#     """
#     mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    
#     for region in regions:
#         x, y, w, h = region['box']
        
#         # Expand area untuk menangkap efek sekitar teks
#         x = max(0, x - expand_margin)
#         y = max(0, y - expand_margin)
#         w = min(image_shape[1] - x, w + 2 * expand_margin)
#         h = min(image_shape[0] - y, h + 2 * expand_margin)
        
#         # Set area teks jadi putih (255)
#         mask[y:y+h, x:x+w] = 255
    
#     return mask

# # ========== ELA FOKUS PADA TEKS ==========
# def error_level_analysis_focused(image_path, text_mask, quality=90):
#     """
#     ELA yang hanya fokus pada area teks/nominal
#     """
#     original = Image.open(image_path).convert('RGB')
#     temp_path = "temp_ela.jpg"
    
#     original.save(temp_path, 'JPEG', quality=quality)
#     compressed = Image.open(temp_path)
    
#     ela_image = ImageChops.difference(original, compressed)
#     extrema = ela_image.getextrema()
#     max_diff = max([ex[1] for ex in extrema])
    
#     if max_diff == 0:
#         max_diff = 1
#     scale = 255.0 / max_diff
#     ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
#     ela_array = np.array(ela_image)
    
#     # Hanya analisis area teks (dimana mask = 255)
#     if np.sum(text_mask) > 0:
#         masked_ela = ela_array[text_mask > 0]
#         ela_mean = np.mean(masked_ela)
#         ela_p95 = np.percentile(masked_ela, 95)
#         ela_std = np.std(masked_ela)
#     else:
#         # Fallback jika tidak ada teks terdeteksi
#         ela_mean = np.mean(ela_array)
#         ela_p95 = np.percentile(ela_array, 95)
#         ela_std = np.std(ela_array)
    
#     # Cleanup
#     if os.path.exists(temp_path):
#         os.remove(temp_path)
    
#     return ela_array, ela_mean, ela_p95, ela_std, max_diff

# # ========== NOISE ANALYSIS FOKUS TEKS ==========
# def noise_analysis_focused(image_path, text_mask):
#     """
#     Analisis noise hanya pada area teks
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Laplacian untuk deteksi noise
#     laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
#     # Hanya analisis area teks
#     if np.sum(text_mask) > 0:
#         masked_laplacian = laplacian[text_mask > 0]
#         noise_variance = masked_laplacian.var()
#         noise_mean = np.mean(np.abs(masked_laplacian))
#     else:
#         noise_variance = laplacian.var()
#         noise_mean = np.mean(np.abs(laplacian))
    
#     # Visualisasi
#     noise_image = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
#     noise_image = np.uint8(noise_image)
    
#     return {
#         'variance': noise_variance,
#         'mean': noise_mean,
#         'image': noise_image
#     }

# # ========== ANALISIS PER TEXT REGION ==========
# def analyze_individual_regions(ela_array, regions):
#     """
#     Analisis setiap region teks secara individual
#     Untuk deteksi mana yang paling mencurigakan
#     """
#     suspicious_regions = []
    
#     for region in regions:
#         x, y, w, h = region['box']
        
#         # Crop ELA di region ini
#         region_ela = ela_array[y:y+h, x:x+w]
        
#         region_mean = np.mean(region_ela)
#         region_max = np.max(region_ela)
#         region_std = np.std(region_ela)
        
#         # Hitung suspicion score
#         suspicion = 0
        
#         if region_mean > 20:
#             suspicion += (region_mean - 20) / 30 * 30
        
#         if region_max > 60:
#             suspicion += (region_max - 60) / 90 * 30
        
#         if region_std > 15:
#             suspicion += (region_std - 15) / 20 * 20
        
#         if suspicion > 25:  # Threshold mencurigakan
#             suspicious_regions.append({
#                 'text': region['text'],
#                 'type': region['type'],
#                 'suspicion_score': round(suspicion, 2),
#                 'ela_mean': round(region_mean, 2),
#                 'position': (x, y)
#             })
    
#     # Sort by suspicion score
#     suspicious_regions.sort(key=lambda x: x['suspicion_score'], reverse=True)
    
#     return suspicious_regions

# # ========== DETEKSI FONT INCONSISTENCY ==========
# def detect_font_inconsistency(image_path, regions):
#     """
#     Deteksi inkonsistensi font pada nominal/teks
#     Text yang diedit biasanya punya karakteristik font berbeda
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     font_features = []
    
#     for region in regions:
#         x, y, w, h = region['box']
        
#         if w < 10 or h < 10:  # Skip region terlalu kecil
#             continue
        
#         roi = image[y:y+h, x:x+w]
        
#         # Hitung karakteristik font
#         # 1. Stroke width variation
#         edges = cv2.Canny(roi, 50, 150)
#         stroke_density = np.sum(edges > 0) / (w * h)
        
#         # 2. Texture pattern
#         mean_intensity = np.mean(roi)
#         std_intensity = np.std(roi)
        
#         font_features.append({
#             'stroke_density': stroke_density,
#             'mean_intensity': mean_intensity,
#             'std_intensity': std_intensity,
#             'text': region['text']
#         })
    
#     if len(font_features) < 2:
#         return 0
    
#     # Hitung variance antar region
#     stroke_vars = [f['stroke_density'] for f in font_features]
#     intensity_vars = [f['mean_intensity'] for f in font_features]
    
#     stroke_cv = np.std(stroke_vars) / (np.mean(stroke_vars) + 1e-6)
#     intensity_cv = np.std(intensity_vars) / (np.mean(intensity_vars) + 1e-6)
    
#     inconsistency_score = (stroke_cv + intensity_cv) / 2
    
#     return inconsistency_score

# # ========== DETEKSI COPY-PASTE PADA ANGKA ==========
# def detect_number_cloning(image_path, number_regions):
#     """
#     Deteksi apakah ada angka yang di-copy-paste
#     """
#     if len(number_regions) < 2:
#         return 0
    
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Extract semua region angka
#     number_crops = []
#     for region in number_regions:
#         x, y, w, h = region['box']
#         if w > 10 and h > 10:
#             crop = image[y:y+h, x:x+w]
#             number_crops.append(crop)
    
#     if len(number_crops) < 2:
#         return 0
    
#     # Compare similarity antar region
#     similarities = []
    
#     for i in range(len(number_crops)):
#         for j in range(i + 1, len(number_crops)):
#             crop1 = number_crops[i]
#             crop2 = number_crops[j]
            
#             # Resize ke ukuran sama
#             h = min(crop1.shape[0], crop2.shape[0])
#             w = min(crop1.shape[1], crop2.shape[1])
            
#             crop1_resized = cv2.resize(crop1, (w, h))
#             crop2_resized = cv2.resize(crop2, (w, h))
            
#             # Hitung similarity
#             diff = cv2.absdiff(crop1_resized, crop2_resized)
#             similarity = 1 - (np.mean(diff) / 255)
            
#             if similarity > 0.8:  # Sangat mirip
#                 similarities.append(similarity)
    
#     if similarities:
#         return max(similarities)
    
#     return 0

# # ========== SCORING SYSTEM ==========
# def calculate_authenticity_score(metrics, has_text_regions):
#     """
#     Scoring fokus pada manipulasi teks/nominal
#     """
#     if not has_text_regions:
#         # Jika tidak ada teks terdeteksi, kembalikan score rendah
#         return 40.0, ["⚠️ Tidak ada teks/nominal yang terdeteksi"]
    
#     base_score = 100.0
#     penalties = []
    
#     # 1. ELA pada area teks (PRIORITAS TERTINGGI)
#     if metrics['ela_mean'] > 18:
#         penalty = min((metrics['ela_mean'] - 18) / 32 * 30, 30)
#         base_score -= penalty
#         penalties.append(f"ELA tinggi pada teks ({metrics['ela_mean']:.1f})")
    
#     if metrics['ela_p95'] > 45:
#         penalty = min((metrics['ela_p95'] - 45) / 55 * 20, 20)
#         base_score -= penalty
#         penalties.append(f"ELA P95 tinggi ({metrics['ela_p95']:.1f})")
    
#     if metrics['ela_std'] > 20:
#         penalty = min((metrics['ela_std'] - 20) / 30 * 15, 15)
#         base_score -= penalty
#         penalties.append(f"Variasi ELA tinggi ({metrics['ela_std']:.1f})")
    
#     # 2. Noise pada area teks
#     if metrics['noise_variance'] > 800:
#         penalty = min((metrics['noise_variance'] - 800) / 1200 * 15, 15)
#         base_score -= penalty
#         penalties.append(f"Noise tidak natural ({metrics['noise_variance']:.1f})")
    
#     # 3. Font inconsistency (SANGAT PENTING)
#     if metrics['font_inconsistency'] > 0.25:
#         penalty = min((metrics['font_inconsistency'] - 0.25) / 0.35 * 25, 25)
#         base_score -= penalty
#         penalties.append(f"Inkonsistensi font terdeteksi ({metrics['font_inconsistency']:.3f})")
    
#     # 4. Clone detection pada angka
#     if metrics['number_clone_score'] > 0.85:
#         penalty = min((metrics['number_clone_score'] - 0.85) / 0.15 * 20, 20)
#         base_score -= penalty
#         penalties.append(f"Copy-paste angka terdeteksi ({metrics['number_clone_score']:.3f})")
    
#     # 5. Suspicious regions
#     if metrics['suspicious_count'] > 0:
#         penalty = min(metrics['suspicious_count'] * 8, 20)
#         base_score -= penalty
#         penalties.append(f"{metrics['suspicious_count']} area teks mencurigakan")
    
#     return max(base_score, 0), penalties

# # ========== PENYIMPANAN GAMBAR ==========
# def save_images_with_highlights(original_path, ela_image, text_mask, suspicious_regions):
#     """
#     Simpan gambar dengan highlight pada area mencurigakan
#     """
#     original = cv2.imread(original_path)
    
#     if original is None:
#         return
    
#     try:
#         # Gambar asli
#         cv2.imwrite("gambar_asli.png", original)
        
#         # Gambar dengan highlight area teks
#         highlighted = original.copy()
        
#         # Tambahkan mask teks (semi-transparent)
#         mask_colored = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)
#         mask_colored[:, :, 1] = text_mask  # Green channel
#         highlighted = cv2.addWeighted(highlighted, 0.7, mask_colored, 0.3, 0)
        
#         # Highlight suspicious regions dengan kotak merah
#         for region in suspicious_regions:
#             x, y = region['position']
#             cv2.rectangle(highlighted, (x-5, y-5), (x+50, y+30), (0, 0, 255), 2)
#             cv2.putText(highlighted, f"{region['suspicion_score']:.0f}", 
#                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
#         cv2.imwrite("highlighted_text_regions.png", highlighted)
        
#         # ELA image
#         storage_folder = "../storage/app/public/hasil_ela"
#         os.makedirs(storage_folder, exist_ok=True)
        
#         ela_pil = Image.fromarray(ela_image.astype('uint8'))
#         ela_pil.save("ela.png")
#         ela_pil.save(os.path.join(storage_folder, "ela.png"))
        
#     except Exception as e:
#         print(f"Error saving images: {e}")

# # ========== MAIN PROGRAM ==========
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("image_path", help="Path gambar bukti pembayaran")
#     parser.add_argument("--verbose", action="store_true", help="Show detailed metrics")
#     args = parser.parse_args()
    
#     # 1. Deteksi area teks dan nominal
#     number_regions, text_regions = detect_text_regions(args.image_path)
#     all_regions = number_regions + text_regions
    
#     has_text = len(all_regions) > 0
    
#     # 2. Buat mask fokus pada teks
#     image = cv2.imread(args.image_path)
#     text_mask = create_text_mask(image.shape, all_regions, expand_margin=15)
    
#     # 3. Analisis fokus pada area teks
#     ela_result, ela_mean, ela_p95, ela_std, ela_max = error_level_analysis_focused(
#         args.image_path, text_mask
#     )
    
#     noise_data = noise_analysis_focused(args.image_path, text_mask)
    
#     # 4. Analisis per region
#     suspicious_regions = analyze_individual_regions(ela_result, all_regions)
    
#     # 5. Deteksi font inconsistency
#     font_inconsistency = detect_font_inconsistency(args.image_path, all_regions)
    
#     # 6. Deteksi cloning pada angka
#     number_clone_score = detect_number_cloning(args.image_path, number_regions)
    
#     # Kumpulkan metrics
#     metrics = {
#         'ela_mean': ela_mean,
#         'ela_p95': ela_p95,
#         'ela_std': ela_std,
#         'noise_variance': noise_data['variance'],
#         'font_inconsistency': font_inconsistency,
#         'number_clone_score': number_clone_score,
#         'suspicious_count': len(suspicious_regions)
#     }
    
#     # Hitung authenticity score
#     authenticity_score, penalties = calculate_authenticity_score(metrics, has_text)
    
#     # Determine status
#     if authenticity_score >= 80:
#         status = "ASLI"
#         confidence = "High"
#         is_authentic = True
#     elif authenticity_score >= 65:
#         status = "KEMUNGKINAN ASLI"
#         confidence = "Medium"
#         is_authentic = True
#     elif authenticity_score >= 45:
#         status = "MENCURIGAKAN"
#         confidence = "Low"
#         is_authentic = False
#     else:
#         status = "PALSU"
#         confidence = "High"
#         is_authentic = False
    
#     # Compile red flags
#     red_flags = []
    
#     for penalty in penalties:
#         red_flags.append(f"⚠️ {penalty}")
    
#     if suspicious_regions:
#         top_suspicious = suspicious_regions[:3]
#         for sr in top_suspicious:
#             red_flags.append(f"🔴 '{sr['text']}' sangat mencurigakan (score: {sr['suspicion_score']})")
    
#     # Result
#     result = {
#         "authenticity_score": round(authenticity_score, 2),
#         "status": status,
#         "is_authentic": is_authentic,
#         "confidence": confidence,
#         "text_regions_detected": len(all_regions),
#         "number_regions_detected": len(number_regions),
#         "suspicious_regions_count": len(suspicious_regions),
#         "red_flags": red_flags,
#         "suspicious_texts": [
#             {"text": sr['text'], "score": sr['suspicion_score']} 
#             for sr in suspicious_regions[:5]
#         ],
#         "metrics": {
#             "ela_mean": round(ela_mean, 2),
#             "ela_p95": round(ela_p95, 2),
#             "noise_variance": round(noise_data['variance'], 2),
#             "font_inconsistency": round(font_inconsistency, 3),
#             "number_clone_score": round(number_clone_score, 3)
#         } if args.verbose else {}
#     }
    
#     # Simpan gambar dengan highlight
#     save_images_with_highlights(args.image_path, ela_result, text_mask, suspicious_regions)
    
#     print(json.dumps(result, indent=4))

    
# # apakah anda juga bisa mendeteksi kerapatan teks dengan lainnya? seperti apakah jarak antar teks itu ada yg berbeda, karna kadang editan itu jarak antar teks nya ada berubah walaupun dikit, jadi harus detil. lalu kekecilan teks font size nya itu dilihat, apakah bisa? lalu fokuskan mendeteksi score dari teks yang diedit itu di bagian bagian teks yang krusial, seperti nominal, nama bank, tujuan transfer dll, yg krusial, cek juga inkonsistensi nominal juga jadi salah satu cara mengecek, contoh nominal utama sudah di edit dari 59.000 menjadi 99.000, dan di bawah ada nominal informasi lagi ternyata masih 59.000, nah itu juga bisa dicek juga. apakah bisa melakukan itu semua? yg sekarang sudah bagus, tetapi masih kurang. tolong perbaiki dan sempurnakan lagi yaa.
# # diakun Firas Suhail std (improving image authenticity detection with ELA analysis)


import argparse
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import pytesseract
import re
from difflib import SequenceMatcher

# ========== KONFIGURASI FIELD KRUSIAL ==========
CRITICAL_KEYWORDS = {
    'nominal': {
        'keywords': ['rp', 'total', 'jumlah', 'amount', 'bayar', 'tagihan', 'biaya'],
        'priority': 10,
        'must_have_number': True
    },
    'bank': {
        'keywords': ['bca', 'mandiri', 'bri', 'bni', 'cimb', 'permata', 'danamon', 
                     'bank', 'rekening', 'account'],
        'priority': 8,
        'must_have_number': False
    },
    'tujuan': {
        'keywords': ['tujuan', 'penerima', 'dari', 'ke', 'kepada', 'transfer', 
                     'nama', 'beneficiary', 'recipient'],
        'priority': 9,
        'must_have_number': False
    },
    'tanggal': {
        'keywords': ['tanggal', 'waktu', 'date', 'time', 'jam', 'pukul'],
        'priority': 7,
        'must_have_number': True
    },
    'referensi': {
        'keywords': ['ref', 'referensi', 'no', 'id', 'trx', 'transaksi'],
        'priority': 6,
        'must_have_number': True
    }
}

# ========== DETEKSI AREA TEKS DENGAN METADATA ==========
def detect_text_regions_advanced(image_path):
    """
    Deteksi teks dengan informasi detail: posisi, ukuran, spacing
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    try:
        ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, 
                                            lang='ind+eng', config='--psm 6')
        
        all_regions = []
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            if conf > 30 and text:
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Deteksi tipe field
                field_type, priority = classify_text_field(text)
                
                # Extract numbers dari text
                numbers = extract_numbers(text)
                
                region = {
                    'box': (x, y, w, h),
                    'text': text,
                    'confidence': conf,
                    'field_type': field_type,
                    'priority': priority,
                    'font_height': h,
                    'font_width': w / max(len(text), 1),  # Rata-rata lebar per karakter
                    'has_number': len(numbers) > 0,
                    'numbers': numbers,
                    'position_y': y,
                    'position_x': x
                }
                
                all_regions.append(region)
        
        # Sort by vertical position (top to bottom)
        all_regions.sort(key=lambda x: x['position_y'])
        
        return all_regions
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return []

# ========== KLASIFIKASI FIELD KRUSIAL ==========
def classify_text_field(text):
    """
    Klasifikasi apakah text termasuk field krusial
    """
    text_lower = text.lower()
    
    for field_type, config in CRITICAL_KEYWORDS.items():
        for keyword in config['keywords']:
            if keyword in text_lower:
                return field_type, config['priority']
    
    # Cek apakah pure number (kemungkinan nominal)
    if re.match(r'^[\d\.,\s]+$', text.replace('Rp', '').strip()):
        if len(text.replace('.', '').replace(',', '').replace(' ', '')) >= 4:
            return 'nominal', 10
    
    return 'other', 3

# ========== EXTRACT ANGKA ==========
def extract_numbers(text):
    """
    Extract semua angka dari text
    """
    # Remove currency symbols
    cleaned = text.replace('Rp', '').replace('IDR', '').strip()
    
    # Extract numbers
    numbers = re.findall(r'\d+(?:[.,]\d+)*', cleaned)
    
    # Convert to actual numbers
    result = []
    for num in numbers:
        try:
            # Remove thousand separators, replace comma with dot
            normalized = num.replace('.', '').replace(',', '.')
            result.append(float(normalized))
        except:
            pass
    
    return result

# ========== ANALISIS SPACING ANTAR TEKS ==========
def analyze_text_spacing(regions):
    """
    Analisis konsistensi jarak antar teks (vertical & horizontal)
    """
    if len(regions) < 3:
        return {'vertical_cv': 0, 'horizontal_cv': 0, 'irregularities': []}
    
    # Group regions by approximate Y position (line detection)
    lines = []
    current_line = [regions[0]]
    
    for i in range(1, len(regions)):
        y_diff = abs(regions[i]['position_y'] - current_line[-1]['position_y'])
        
        if y_diff < 15:  # Same line threshold
            current_line.append(regions[i])
        else:
            if len(current_line) > 1:
                lines.append(current_line)
            current_line = [regions[i]]
    
    if len(current_line) > 1:
        lines.append(current_line)
    
    # Analisis vertical spacing (antar baris)
    vertical_gaps = []
    for i in range(len(regions) - 1):
        y_gap = regions[i+1]['position_y'] - (regions[i]['position_y'] + regions[i]['box'][3])
        if y_gap > 0:  # Valid gap
            vertical_gaps.append(y_gap)
    
    vertical_cv = np.std(vertical_gaps) / (np.mean(vertical_gaps) + 1e-6) if vertical_gaps else 0
    
    # Analisis horizontal spacing (dalam satu baris)
    horizontal_gaps = []
    for line in lines:
        line.sort(key=lambda x: x['position_x'])
        for i in range(len(line) - 1):
            x_gap = line[i+1]['position_x'] - (line[i]['position_x'] + line[i]['box'][2])
            if x_gap > 0:
                horizontal_gaps.append(x_gap)
    
    horizontal_cv = np.std(horizontal_gaps) / (np.mean(horizontal_gaps) + 1e-6) if horizontal_gaps else 0
    
    # Deteksi irregularities (outliers)
    irregularities = []
    
    if vertical_gaps:
        v_mean = np.mean(vertical_gaps)
        v_std = np.std(vertical_gaps)
        
        for i, gap in enumerate(vertical_gaps):
            if abs(gap - v_mean) > 2 * v_std:  # Outlier
                irregularities.append({
                    'type': 'vertical_spacing',
                    'between': f"{regions[i]['text']} → {regions[i+1]['text']}",
                    'expected': round(v_mean, 1),
                    'actual': round(gap, 1),
                    'deviation': round(abs(gap - v_mean), 1)
                })
    
    return {
        'vertical_cv': vertical_cv,
        'horizontal_cv': horizontal_cv,
        'irregularities': irregularities
    }

# ========== ANALISIS FONT SIZE CONSISTENCY ==========
def analyze_font_consistency(regions):
    """
    Deteksi inkonsistensi ukuran font, terutama pada field krusial
    """
    if len(regions) < 3:
        return {'font_height_cv': 0, 'font_width_cv': 0, 'inconsistencies': []}
    
    # Group by field type
    field_groups = {}
    for region in regions:
        field_type = region['field_type']
        if field_type not in field_groups:
            field_groups[field_type] = []
        field_groups[field_type].append(region)
    
    inconsistencies = []
    
    # Analisis per group
    for field_type, group in field_groups.items():
        if len(group) < 2:
            continue
        
        heights = [r['font_height'] for r in group]
        widths = [r['font_width'] for r in group]
        
        h_mean = np.mean(heights)
        h_std = np.std(heights)
        w_mean = np.mean(widths)
        w_std = np.std(widths)
        
        # Deteksi outlier dalam group
        for region in group:
            h_dev = abs(region['font_height'] - h_mean)
            w_dev = abs(region['font_width'] - w_mean)
            
            if h_dev > 1.5 * h_std or w_dev > 1.5 * w_std:
                inconsistencies.append({
                    'text': region['text'],
                    'field_type': field_type,
                    'height_deviation': round(h_dev, 1),
                    'width_deviation': round(w_dev, 1),
                    'priority': region['priority']
                })
    
    # Overall font consistency
    all_heights = [r['font_height'] for r in regions]
    all_widths = [r['font_width'] for r in regions]
    
    font_height_cv = np.std(all_heights) / (np.mean(all_heights) + 1e-6)
    font_width_cv = np.std(all_widths) / (np.mean(all_widths) + 1e-6)
    
    return {
        'font_height_cv': font_height_cv,
        'font_width_cv': font_width_cv,
        'inconsistencies': sorted(inconsistencies, key=lambda x: x['priority'], reverse=True)
    }

# ========== CROSS-VALIDATION NOMINAL ==========
def cross_validate_nominals(regions):
    """
    Cek konsistensi nominal di berbagai tempat
    Contoh: Total transfer vs Jumlah dibayar harus sama
    """
    # Extract semua region yang ada nominal
    nominal_regions = [r for r in regions if r['has_number'] and r['numbers']]
    
    if len(nominal_regions) < 2:
        return {'conflicts': [], 'consistency_score': 1.0}
    
    # Group nominal by value
    nominal_values = {}
    for region in nominal_regions:
        for num in region['numbers']:
            if num >= 1000:  # Hanya nominal besar (kemungkinan uang)
                key = int(num)
                if key not in nominal_values:
                    nominal_values[key] = []
                nominal_values[key].append(region)
    
    conflicts = []
    
    # Cek apakah ada nominal yang seharusnya sama tapi beda
    # Strategi: cari nominal yang "mirip" tapi tidak sama persis
    all_nominals = list(nominal_values.keys())
    
    for i in range(len(all_nominals)):
        for j in range(i + 1, len(all_nominals)):
            num1 = all_nominals[i]
            num2 = all_nominals[j]
            
            # Cek apakah "mirip" (beda digit pertama atau terakhir)
            # Contoh: 59000 vs 99000 (beda digit pertama)
            if is_suspicious_nominal_pair(num1, num2):
                regions1 = nominal_values[num1]
                regions2 = nominal_values[num2]
                
                conflicts.append({
                    'nominal1': num1,
                    'nominal2': num2,
                    'text1': regions1[0]['text'],
                    'text2': regions2[0]['text'],
                    'field1': regions1[0]['field_type'],
                    'field2': regions2[0]['field_type'],
                    'suspicion': 'Nominal berbeda dengan pola edit'
                })
    
    # Hitung consistency score
    consistency_score = 1.0 if not conflicts else max(0, 1.0 - len(conflicts) * 0.3)
    
    return {
        'conflicts': conflicts,
        'consistency_score': consistency_score
    }

def is_suspicious_nominal_pair(num1, num2):
    """
    Deteksi apakah dua nominal mencurigakan (kemungkinan hasil edit)
    """
    str1 = str(int(num1))
    str2 = str(int(num2))
    
    if len(str1) != len(str2):
        return False
    
    # Hitung berapa digit yang berbeda
    diff_count = sum(c1 != c2 for c1, c2 in zip(str1, str2))
    
    # Jika hanya 1-2 digit yang beda, dan nilainya signifikan, mencurigakan
    if diff_count <= 2:
        ratio = max(num1, num2) / min(num1, num2)
        if ratio > 1.2:  # Beda lebih dari 20%
            return True
    
    return False

# ========== ELA FOKUS FIELD KRUSIAL ==========
def error_level_analysis_critical_fields(image_path, regions, quality=90):
    """
    ELA dengan prioritas pada field krusial
    """
    original = Image.open(image_path).convert('RGB')
    temp_path = "temp_ela.jpg"
    
    original.save(temp_path, 'JPEG', quality=quality)
    compressed = Image.open(temp_path)
    
    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    ela_array = np.array(ela_image)
    
    # Analisis per field dengan weighted priority
    field_scores = []
    
    for region in regions:
        x, y, w, h = region['box']
        
        # Expand slightly
        x = max(0, x - 10)
        y = max(0, y - 10)
        w = min(ela_array.shape[1] - x, w + 20)
        h = min(ela_array.shape[0] - y, h + 20)
        
        roi = ela_array[y:y+h, x:x+w]
        
        roi_mean = np.mean(roi)
        roi_p95 = np.percentile(roi, 95)
        roi_std = np.std(roi)
        roi_max = np.max(roi)
        
        # Weighted score berdasarkan priority
        weighted_score = (roi_mean * 0.3 + roi_p95 * 0.4 + roi_std * 0.2 + roi_max * 0.1) * region['priority'] / 10
        
        field_scores.append({
            'region': region,
            'ela_mean': roi_mean,
            'ela_p95': roi_p95,
            'ela_std': roi_std,
            'ela_max': roi_max,
            'weighted_score': weighted_score
        })
    
    # Overall weighted ELA
    if field_scores:
        critical_fields = [f for f in field_scores if f['region']['priority'] >= 7]
        
        if critical_fields:
            weighted_ela = np.average(
                [f['ela_mean'] for f in critical_fields],
                weights=[f['region']['priority'] for f in critical_fields]
            )
        else:
            weighted_ela = np.mean([f['ela_mean'] for f in field_scores])
    else:
        weighted_ela = np.mean(ela_array)
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return ela_array, weighted_ela, field_scores

# ========== LAYOUT PATTERN ANALYSIS ==========
def analyze_layout_pattern(regions):
    """
    Deteksi apakah layout terlihat natural atau ada yang janggal
    """
    if len(regions) < 5:
        return {'alignment_score': 1.0, 'issues': []}
    
    issues = []
    
    # 1. Alignment analysis (left/right/center)
    x_positions = [r['position_x'] for r in regions]
    x_std = np.std(x_positions)
    
    # Cari left-aligned texts
    left_aligned = [x for x in x_positions if x < np.median(x_positions)]
    
    if len(left_aligned) > 2:
        left_std = np.std(left_aligned)
        if left_std > 10:  # Misalignment
            issues.append({
                'type': 'misalignment',
                'description': f'Text alignment tidak konsisten (std: {left_std:.1f}px)'
            })
    
    # 2. Grid pattern detection
    # Bukti pembayaran asli biasanya follow grid pattern
    y_positions = [r['position_y'] for r in regions]
    y_diffs = np.diff(sorted(y_positions))
    
    if len(y_diffs) > 3:
        y_diff_std = np.std(y_diffs)
        y_diff_mean = np.mean(y_diffs)
        
        if y_diff_std / (y_diff_mean + 1e-6) > 0.5:
            issues.append({
                'type': 'irregular_spacing',
                'description': 'Spacing vertikal tidak mengikuti pola grid'
            })
    
    alignment_score = max(0, 1.0 - len(issues) * 0.2)
    
    return {
        'alignment_score': alignment_score,
        'issues': issues
    }

# ========== SCORING SYSTEM ADVANCED ==========
def calculate_authenticity_score_advanced(metrics, regions_count):
    """
    Sistem scoring super detail
    """
    if regions_count == 0:
        return 20.0, ["⚠️ Tidak ada teks yang terdeteksi"]
    
    base_score = 100.0
    penalties = []
    critical_issues = []
    
    # === KATEGORI 1: ELA ANALYSIS (30%) ===
    if metrics['weighted_ela'] > 16:
        penalty = min((metrics['weighted_ela'] - 16) / 34 * 30, 30)
        base_score -= penalty
        if penalty > 20:
            critical_issues.append(f"🔴 ELA sangat tinggi pada field krusial ({metrics['weighted_ela']:.1f})")
        else:
            penalties.append(f"ELA tinggi ({metrics['weighted_ela']:.1f})")
    
    # === KATEGORI 2: NOMINAL CROSS-VALIDATION (25%) ===
    if metrics['nominal_conflicts'] > 0:
        penalty = min(metrics['nominal_conflicts'] * 15, 25)
        base_score -= penalty
        critical_issues.append(f"🔴 {metrics['nominal_conflicts']} konflik nominal terdeteksi")
    
    if metrics['nominal_consistency'] < 0.8:
        penalty = (1.0 - metrics['nominal_consistency']) * 15
        base_score -= penalty
        penalties.append(f"Konsistensi nominal rendah ({metrics['nominal_consistency']:.2f})")
    
    # === KATEGORI 3: FONT CONSISTENCY (20%) ===
    if metrics['font_inconsistencies'] > 0:
        penalty = min(metrics['font_inconsistencies'] * 8, 20)
        base_score -= penalty
        if metrics['font_inconsistencies'] >= 2:
            critical_issues.append(f"🔴 {metrics['font_inconsistencies']} inkonsistensi font")
        else:
            penalties.append(f"Inkonsistensi font terdeteksi")
    
    if metrics['font_height_cv'] > 0.3:
        penalty = min((metrics['font_height_cv'] - 0.3) / 0.4 * 12, 12)
        base_score -= penalty
        penalties.append(f"Variasi ukuran font ({metrics['font_height_cv']:.2f})")
    
    # === KATEGORI 4: SPACING ANALYSIS (15%) ===
    if metrics['spacing_irregularities'] > 0:
        penalty = min(metrics['spacing_irregularities'] * 6, 15)
        base_score -= penalty
        if metrics['spacing_irregularities'] >= 2:
            critical_issues.append(f"🔴 {metrics['spacing_irregularities']} spacing tidak natural")
        else:
            penalties.append(f"Spacing irregularities")
    
    if metrics['vertical_spacing_cv'] > 0.4:
        penalty = min((metrics['vertical_spacing_cv'] - 0.4) / 0.5 * 10, 10)
        base_score -= penalty
        penalties.append(f"Spacing vertikal tidak konsisten")
    
    # === KATEGORI 5: LAYOUT PATTERN (10%) ===
    if metrics['alignment_score'] < 0.7:
        penalty = (1.0 - metrics['alignment_score']) * 10
        base_score -= penalty
        penalties.append(f"Layout pattern tidak natural")
    
    return max(base_score, 0), penalties, critical_issues

# ========== PENYIMPANAN GAMBAR ==========
def save_detailed_analysis(original_path, ela_image, regions, field_scores, conflicts):
    """
    Simpan gambar dengan analisis detail
    """
    original = cv2.imread(original_path)
    
    if original is None:
        return
    
    try:
        # Gambar 1: Highlight field krusial
        highlighted = original.copy()
        
        # Sort by priority
        sorted_scores = sorted(field_scores, key=lambda x: x['weighted_score'], reverse=True)
        
        for i, score_data in enumerate(sorted_scores[:10]):
            region = score_data['region']
            x, y, w, h = region['box']
            
            # Color based on priority and score
            if score_data['weighted_score'] > 50:
                color = (0, 0, 255)  # Red - very suspicious
            elif score_data['weighted_score'] > 30:
                color = (0, 165, 255)  # Orange - suspicious
            elif region['priority'] >= 8:
                color = (0, 255, 255)  # Yellow - critical field
            else:
                color = (0, 255, 0)  # Green - normal
            
            thickness = 3 if region['priority'] >= 8 else 2
            
            cv2.rectangle(highlighted, (x-5, y-5), (x+w+5, y+h+5), color, thickness)
            
            # Label
            label = f"{region['field_type'][:3]}: {score_data['weighted_score']:.0f}"
            cv2.putText(highlighted, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Mark conflicts
        for conflict in conflicts:
            # Find regions
            for region in regions:
                if conflict['text1'] in region['text'] or conflict['text2'] in region['text']:
                    x, y, w, h = region['box']
                    cv2.rectangle(highlighted, (x-8, y-8), (x+w+8, y+h+8), (255, 0, 255), 3)
        
        cv2.imwrite("analysis_detailed.png", highlighted)
        
        # Gambar 2: ELA
        storage_folder = "../storage/app/public/hasil_ela"
        os.makedirs(storage_folder, exist_ok=True)
        
        ela_pil = Image.fromarray(ela_image.astype('uint8'))
        ela_pil.save("ela.png")
        ela_pil.save(os.path.join(storage_folder, "ela.png"))
        
    except Exception as e:
        print(f"Error saving images: {e}")

# ========== MAIN PROGRAM ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path gambar bukti pembayaran")
    parser.add_argument("--verbose", action="store_true", help="Show detailed metrics")
    args = parser.parse_args()
    
    # 1. Deteksi semua text regions dengan metadata
    # print("Mendeteksi teks dan field krusial...")
    regions = detect_text_regions_advanced(args.image_path)
    
    if not regions:
        result = {
            "authenticity_score": 20.0,
            "status": "ERROR",
            "is_authentic": False,
            "confidence": "Low",
            "red_flags": ["⚠️ Tidak dapat mendeteksi teks pada gambar"]
        }
        print(json.dumps(result, indent=4, ensure_ascii=False))
        exit()
    
    # print(f"✅ Terdeteksi {len(regions)} text regions")
    
    # 2. Analisis spacing
    # print("📏 Menganalisis spacing antar teks...")
    spacing_analysis = analyze_text_spacing(regions)
    
    # 3. Analisis font consistency
    # print("🔤 Menganalisis konsistensi font...")
    font_analysis = analyze_font_consistency(regions)
    
    # 4. Cross-validate nominals
    # print("💰 Memvalidasi konsistensi nominal...")
    nominal_validation = cross_validate_nominals(regions)
    
    # 5. ELA analysis fokus field krusial
    # print("🔬 Melakukan Error Level Analysis...")
    ela_result, weighted_ela, field_scores = error_level_analysis_critical_fields(args.image_path, regions)
    
    # 6. Layout pattern analysis
    # print("📐 Menganalisis pola layout...")
    layout_analysis = analyze_layout_pattern(regions)
    
    # Compile metrics
    metrics = {
        'weighted_ela': weighted_ela,
        'nominal_conflicts': len(nominal_validation['conflicts']),
        'nominal_consistency': nominal_validation['consistency_score'],
        'font_inconsistencies': len(font_analysis['inconsistencies']),
        'font_height_cv': font_analysis['font_height_cv'],
        'font_width_cv': font_analysis['font_width_cv'],
        'spacing_irregularities': len(spacing_analysis['irregularities']),
        'vertical_spacing_cv': spacing_analysis['vertical_cv'],
        'horizontal_spacing_cv': spacing_analysis['horizontal_cv'],
        'alignment_score': layout_analysis['alignment_score']
    }
    
    # Calculate score
    # print("🎯 Menghitung authenticity score...")
    authenticity_score, penalties, critical_issues = calculate_authenticity_score_advanced(
        metrics, len(regions)
    )
    
    # Determine status
    if authenticity_score >= 85:
        status = "ASLI"
        confidence = "Very High"
        is_authentic = True
    elif authenticity_score >= 70:
        status = "KEMUNGKINAN ASLI"
        confidence = "High"
        is_authentic = True
    elif authenticity_score >= 55:
        status = "RAGU-RAGU"
        confidence = "Medium"
        is_authentic = False
    elif authenticity_score >= 40:
        status = "MENCURIGAKAN"
        confidence = "High"
        is_authentic = False
    else:
        status = "PALSU"
        confidence = "Very High"
        is_authentic = False
    
    # Compile red flags
    all_red_flags = critical_issues + [f"⚠️ {p}" for p in penalties]
    
    # Add specific issues
    for conflict in nominal_validation['conflicts']:
        all_red_flags.append(
            f" Konflik nominal: '{conflict['text1']}' vs '{conflict['text2']}' "
            f"({conflict['nominal1']:,.0f} vs {conflict['nominal2']:,.0f})"
        )
    
    for irregularity in spacing_analysis['irregularities'][:3]:
        all_red_flags.append(
            f"📏 Spacing tidak normal: {irregularity['between']} "
            f"(expected: {irregularity['expected']}px, actual: {irregularity['actual']}px)"
        )
    
    for incons in font_analysis['inconsistencies'][:3]:
        all_red_flags.append(
            f" Font '{incons['text']}' tidak konsisten (field: {incons['field_type']})"
        )
    
    # Top suspicious fields
    top_suspicious = sorted(field_scores, key=lambda x: x['weighted_score'], reverse=True)[:5]
    
    # Result
    result = {
        "authenticity_score": round(authenticity_score, 2),
        "status": status,
        "is_authentic": is_authentic,
        "confidence": confidence,
        "summary": {
            "total_text_regions": len(regions),
            "critical_fields_detected": len([r for r in regions if r['priority'] >= 8]),
            "nominal_conflicts": len(nominal_validation['conflicts']),
            "font_inconsistencies": len(font_analysis['inconsistencies']),
            "spacing_irregularities": len(spacing_analysis['irregularities'])
        },
        # "red_flags": all_red_flags,
        "most_suspicious_fields": [
            {
                "text": s['region']['text'],
                "field_type": s['region']['field_type'],
                "suspicion_score": round(s['weighted_score'], 2),
                "ela_mean": round(s['ela_mean'], 2),
                "priority": s['region']['priority']
            }
            for s in top_suspicious
        ],
        "nominal_analysis": {
            "conflicts": nominal_validation['conflicts'],
            "consistency_score": round(nominal_validation['consistency_score'], 2)
        } if args.verbose else {},
        "detailed_metrics": {
            "weighted_ela": round(weighted_ela, 2),
            "font_height_cv": round(font_analysis['font_height_cv'], 3),
            "font_width_cv": round(font_analysis['font_width_cv'], 3),
            "vertical_spacing_cv": round(spacing_analysis['vertical_cv'], 3),
            "horizontal_spacing_cv": round(spacing_analysis['horizontal_cv'], 3),
            "alignment_score": round(layout_analysis['alignment_score'], 2)
        } if args.verbose else {}
    }
    
    # Simpan hasil analisis visual
    # print("💾 Menyimpan hasil analisis...")
    save_detailed_analysis(
        args.image_path, 
        ela_result, 
        regions, 
        field_scores,
        nominal_validation['conflicts']
    )
    
    # print("\n" + "="*60)
    print(json.dumps(result, indent=4))
    # print("="*60)
