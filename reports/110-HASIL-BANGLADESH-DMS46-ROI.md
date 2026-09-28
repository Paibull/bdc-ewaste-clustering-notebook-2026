# Validasi Eksternal DMS46 pada ROI YOLO Bangladesh

## Pertanyaan eksperimen

Apakah proporsi material DMS46 di dalam union bounding box YOLO diperkaya secara konsisten pada empat kelas material Bangladesh, dan bagaimana hasilnya dibanding evaluasi `rembg` pada key gambar yang sama?

## Perbaikan desain

DMS46 tetap frozen. Primary score adalah fraction material kontinu di dalam ROI bbox tanpa padding; bbox axis-aligned untuk anotasi polygon YOLO diturunkan dari min/max titik polygon. Tidak dilakukan argmax untuk primary metric dan tidak digunakan `rembg`. Output enam fraksi juga diringkas secara deskriptif pada 12 kelas. Bangladesh menyediakan label kelas objek, bukan pixel-level material ground truth.

## Metric utama

Cakupan anotasi: 2157 file gambar; 2153 anotasi tidak kosong; 2153 gambar dengan ROI valid. Subset material: 1018/1.018 key dinilai; 0 dikecualikan. 1407 anotasi polygon dikonversi menjadi bbox enclosing; 0 bbox invalid; 0 bbox kosong.

| Material | N positif | AUROC (95% CI) | AP | Prevalensi (AP chance) | AP lift | Median positif | Median negatif |
|---|---:|---:|---:|---:|---:|---:|---:|
| glass | 127 | 0.780 (0.736–0.823) | 0.391 | 0.125 | 3.13× | 0.135 | 0.000 |
| metal | 293 | 0.683 (0.648–0.718) | 0.543 | 0.288 | 1.89× | 0.003 | 0.000 |
| paper_cardboard | 200 | 0.757 (0.717–0.796) | 0.473 | 0.196 | 2.41× | 0.370 | 0.009 |
| plastic | 398 | 0.707 (0.674–0.738) | 0.598 | 0.391 | 1.53× | 0.305 | 0.078 |

Macro AUROC **0.732** (bootstrap 95% CI 0.710–0.752); macro AP **0.501**; macro AP lift **2.24×**; permutation p=0.0010 (999 permutasi, seed 20260927).

## ROI versus `rembg`

Pada 1018 key yang sama: macro AUROC ROI 0.732 vs `rembg` 0.739, paired Δ=-0.008 (95% CI -0.015–+0.000); macro AP ROI 0.501 vs 0.528, paired Δ=-0.027 (95% CI -0.047–-0.004). Argmax metrics hanya sensitivity: Macro-F1 0.490, balanced accuracy 0.501, accuracy 0.520, Top-2 0.722, MRR 0.704.

## Profil material deskriptif

Rata-rata fraction terbesar per kelas (bukan prediksi label atau accuracy):

- Battery Waste (N=148): other_unknown (0.450)
- Glass Waste (N=127): other_unknown (0.496)
- Keyboard (N=126): plastic (0.608)
- Light Bulb (N=43): other_unknown (0.472)
- Medical Waste (N=282): other_unknown (0.831)
- Metal Waste (N=293): other_unknown (0.557)
- Mobile (N=151): plastic (0.481)
- Mouse (N=80): plastic (0.523)
- Organic Waste (N=100): other_unknown (0.886)
- Paper Waste (N=200): other_unknown (0.465)
- PCB (N=205): plastic (0.424)
- Plastic Waste (N=398): other_unknown (0.390)

## Deteksi material target tanpa pemaksaan argmax

| Material | Target ≥1% ROI | Target ≥5% ROI | Target ≥10% ROI |
|---|---:|---:|---:|
| Glass | 67.7% | 60.6% | 53.5% |
| Metal | 45.7% | 37.5% | 31.7% |
| Paper/cardboard | 81.5% | 77.5% | 73.5% |
| Plastic | 90.7% | 83.9% | 77.4% |

Macro rate: ≥1% **71.4%**, ≥5% **64.9%**, ≥10% **59.0%**. Overall rate pada seluruh 1018 gambar: ≥1% **73.1%**, ≥5% **66.4%**, ≥10% **60.5%**.
Top-2 **0.722** dan MRR **0.704** memakai ranking empat kandidat glass, metal, paper_cardboard, plastic; keduanya secondary metric. AUROC/AP tetap primary metric. Detection rate menunjukkan material relevan ditemukan di ROI tanpa harus menjadi argmax; confusion matrix dan Macro-F1 hanya sensitivity analysis.
Fraction kecil dapat berasal dari material nyata atau segmentation noise. Bangladesh tidak menyediakan pixel-level material ground truth maupun daftar lengkap material setiap objek.

Verdict tetap PARTIAL.
DMS46 menunjukkan bukti eksternal moderat sebagai material descriptor. Material target diperkaya di atas chance dan sering muncul dalam komposisi ROI meskipun tidak selalu dominan. Hasil mendukung penggunaan DMS46 sebagai descriptor eksploratif, bukan classifier material tunggal atau estimator komposisi yang tervalidasi penuh.


## Verdict dan batas

Verdict sesuai decision rule terkunci: **PARTIAL**. Ini menguji enrichment fraction DMS46 terhadap label kelas kasar pada dataset Bangladesh; bukan validasi segmentasi pixel atau ground truth komposisi material per objek.

Inferensi: 2153 gambar, GPU NVIDIA A10, batch size 1, runtime 122.29 detik.

Pengecualian anotasi:
- `empty_annotation`: 4
