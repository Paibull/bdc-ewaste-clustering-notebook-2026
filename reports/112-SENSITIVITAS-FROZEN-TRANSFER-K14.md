# Sensitivitas Frozen External Transfer pada K 14

Tanggal evaluasi: 27 September 2026  
Status: `SENSITIVITY_ONLY`  
Inference baru: tidak ada; evaluasi memakai embedding Bangladesh yang sudah tersimpan.

## Tujuan

Sweep resolusi internal memilih fused K-Means K=14, sedangkan frozen external transfer awal memakai K=12 yang telah dibekukan sebelumnya. Evaluasi ini memeriksa apakah perbedaan K mengubah kesimpulan transfer tanpa memilih parameter atau mapping dari label Bangladesh.

## Protokol

Fused space K=14 direkonstruksi dari 3.579 citra discovery BDC dengan preprocessing yang sama seperti sweep. Rekonstruksi terhadap assignment tersimpan menghasilkan ARI 1,000. Mapping family ditentukan dari semantic profile BDC sebelum scoring eksternal:

- cluster 0: battery;
- cluster 5: mobile;
- cluster 12: PCB melalui node `electronic_component`;
- cluster 6 dan 9: input peripheral.

Evaluasi memakai 710 citra known-set dan 1.443 citra out-of-taxonomy yang sama dengan laporan 111. Bootstrap paired distratifikasi menurut family, 2.000 draw, seed 20260927.

## Hasil known-set

| Estimator | Micro Top-1 | Macro Top-1 | Top-2 | MRR |
|---|---:|---:|---:|---:|
| Fused K=14 selected-K | 0,9070 | 0,9031 | 0,9592 | 0,9464 |
| Fused K=12 frozen | 0,9085 | 0,9051 | 0,9606 | 0,9477 |
| RADIO K=12 selected-K | 0,8803 | 0,8667 | 0,9380 | 0,9297 |

Paired delta K=14 dikurangi K=12 untuk Macro Top-1 adalah -0,0020 dengan CI 95% [-0,0154; +0,0129]. Perbedaannya tidak terukur secara meyakinkan. Paired delta K=14 dikurangi RADIO untuk Macro Top-1 adalah +0,0364 dengan CI 95% [+0,0172; +0,0573]. Kesimpulan bahwa fusi meningkatkan transfer agregat tetap bertahan pada selected-K internal.

Per-family recall K=14 adalah Battery 0,8176, Mobile 0,9404, PCB 1,0000, dan Input peripheral 0,8544. Dibanding K=12, K=14 menukar sebagian recall Battery dengan recall Mobile; agregatnya tetap hampir identik.

## Hasil novelty

| Estimator | Nearest-distance AUROC / AUPRC | Negative-margin AUROC / AUPRC |
|---|---:|---:|
| Fused K=14 | 0,9271 / 0,9484 | 0,8734 / 0,8881 |
| Fused K=12 | 0,9049 / 0,9342 | 0,8476 / 0,8738 |
| RADIO K=12 | 0,9197 / 0,9430 | 0,8937 / 0,9133 |

K=14 memperbaiki nearest-family distance untuk novelty, tetapi RADIO tetap terbaik pada negative margin. Karena tidak ada threshold yang dituning, hasil ini tetap dibaca sebagai kemampuan ranking unknown.

## Keputusan

Tidak diperlukan inference atau validasi ulang lain untuk menyelesaikan inkonsistensi K. Gunakan:

- K=14 sebagai baseline selected-K internal dan atlas multi-resolusi;
- K=12 sebagai frozen external protocol utama karena telah dibekukan sebelum evaluasi Bangladesh;
- K=14 sebagai sensitivity analysis yang menunjukkan kesimpulan transfer tidak bergantung pada K=12.

Jangan mengganti tabel utama laporan 111 secara retroaktif. Laporkan sensitivitas ini setelah hasil frozen K=12 atau dalam lampiran.

Artefak: `results/bangladesh_frozen_transfer_k14_sensitivity/metrics.json`. Implementasi: `scripts/evaluate_bangladesh_k14_sensitivity.py`.
