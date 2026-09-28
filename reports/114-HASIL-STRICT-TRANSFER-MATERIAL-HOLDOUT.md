# Strict transfer dan material holdout

> Batas partisi dari review final: K12 pada laporan ini berasal dari `semantic_g0_partitions.npz`. Ia berbeda dari medoid K12 pada `k_stability_100/assignments.npz` (ARI discovery 0,974794; urutan key sama). Angka transfer/material berikut memvalidasi partisi terdahulu, belum medoid K12 baru secara persis. Lihat laporan 115 untuk hasil multi-resolution dan batas klaim.

## A. Strict frozen external transfer

Nearest cluster dicari di seluruh centroid; cluster tanpa mapping dihitung sebagai salah. Mapping BDC tetap dibekukan.
K12 menggunakan transform beku `run_semantic_g0`; label K14 berasal dari sweep lama dengan constructor PCA/fusion berbeda dan direkonstruksi melalui transform sensitivitas K14 yang tersimpan. Keduanya lolos ARI 1.0000, tetapi selisih K14−K12 tidak dapat diatribusikan ke jumlah cluster saja.

| Estimator | Strict micro | Strict macro | 95% CI macro | Coverage mapped | Unmapped | Conditional accuracy | Closed-set macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| fused_k12 | 0.890 | 0.888 | [0.865, 0.913] | 0.958 | 0.042 | 0.929 | 0.905 |
| radio_k12 | 0.852 | 0.838 | [0.813, 0.864] | 0.941 | 0.059 | 0.906 | 0.867 |
| fused_k14 | 0.904 | 0.903 | [0.880, 0.926] | 0.965 | 0.035 | 0.937 | 0.927 |
Recall fused_k12 (battery / mobile / PCB / input peripheral): 0.878 / 0.868 / 0.951 / 0.854.
Recall radio_k12 (battery / mobile / PCB / input peripheral): 0.628 / 0.874 / 0.956 / 0.893.
Recall fused_k14 (battery / mobile / PCB / input peripheral): 0.926 / 0.861 / 0.971 / 0.854.

Strict macro null q95: 0.264; fused K12 permutation p=0.0010.
Paired delta fused K12−RADIO K12: [0.029, 0.070]; fused K14−K12: [0.004, 0.026].
Verdict A: **STRICT_TRANSFER_SUPPORTED**. Fusion strict gain: **True**. CI delta K14−K12 tidak memuat nol, tetapi karena constructor space berbeda, ini bukan bukti bahwa perubahan K saja meningkatkan transfer.

## B. Continuous material pada locked holdout

Target ditransformasi sebagai sqrt(fraction) lalu dinormalisasi L2. Baseline memakai source dan statistik akuisisi/background yang distandardisasi dari discovery; full menambahkan indikator cluster. CI memakai bootstrap web/field dan p memakai permutasi cluster dalam source.

| Partisi | Target | Incremental R² (95% CI) | Δ Hellinger (95% CI) | Δ cosine (95% CI) | Permutation p | N holdout |
|---|---|---:|---:|---:|---:|---:|
| fused_k12 | six_materials | 0.1145 [0.0889, 0.1409] | 0.0295 [0.0238, 0.0357] | 0.0190 [0.0145, 0.0235] | 0.0010 | 600 |
| fused_k12 | five_materials | 0.1528 [0.1210, 0.1845] | 0.0463 [0.0378, 0.0545] | 0.0310 [0.0245, 0.0377] | 0.0010 | 598 |
| fused_k14 | six_materials | 0.1190 [0.0915, 0.1463] | 0.0300 [0.0236, 0.0364] | 0.0196 [0.0150, 0.0242] | 0.0010 | 600 |
| fused_k14 | five_materials | 0.1571 [0.1259, 0.1887] | 0.0466 [0.0384, 0.0553] | 0.0317 [0.0251, 0.0383] | 0.0010 | 598 |

Baris holdout tanpa mass pada lima material known: 2. Verdict B: **ROBUST_MATERIAL_SUPPORT**.

## Kesimpulan untuk paper

Reconstruction ARI: fused_k12=1.0000, radio_k12=1.0000, fused_k14=1.0000. Klaim transfer harus mengikuti hasil nearest centroid global dan tingkat cluster unmapped; skor closed-set saja tidak membuktikan setiap gambar dapat dipetakan oleh taxonomy beku.
Klaim aman: cluster membawa informasi tambahan tentang komposisi DMS pada locked holdout setelah source dan covariates akuisisi/background dikontrol. Klaim harus dipersempit: ini bukan validasi pixel-level material ground truth atau bukti bahwa cluster adalah taxonomy material; efek K12/K14 tidak dapat dipisahkan dari perbedaan constructor fused yang diwarisi.

Runtime scoring: 0.5 menit. 
