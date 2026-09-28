# Hasil AMF-DINOv2 dan perbandingan framework

Tanggal run: 25 September 2026  
Status: selesai dan diaudit dengan evaluator bersama.

## Provenance AMF

Notebook `kaggle_amf_dinov2.ipynb` dan varian divisive direplikasi pada
4.179 citra canonical melalui Modal. Encoder tetap frozen DINOv2 ViT-L/14;
empat view diambil dari blok 6, 12, 18, dan 24, masing-masing berupa gabungan
`[CLS; mean patch]` berukuran 2.048. AMF memakai `k=30`, `c=12`, seed 42, dan
maksimum 15 iterasi. Divisive memakai arm literal-adaptive, split `c=2`, gate
modularity `Q >= 0,30`, minimum anak 40, maksimum 20 daun.

Run Modal:

- App: `bdc-amf-dinov2`, App ID `ap-2CQKlVt3SG7w3SEvgQOVhw`;
- run: `full-1790306969`;
- FunctionCall: `fc-01M3B9RY5S4KBRE4JQW7AE8116`;
- device: CUDA, Torch `2.5.1+cu124`;
- waktu remote: 215,7 detik;
- manifest SHA-256: `48f197459235414cc4d063c58d98e40f875cc61d9c5d5f2b1088a0ef4343c84d`;
- `final_partition_modified=false`.

Artifact utama berada di
`semifinal/results/amf_dinov2/full-1790306969/full-1790306969/`.
Hash artifact utama: `metrics.csv`
`fe603a1f191dd49b3bc6f87e25ee9c02223733791f992fe19c9b2494bdf658e5`,
`assignments.csv`
`5d3e1c1770d03edb7eafc9b0d65f35b8303bc2724aa32c185460a21371ed280d`, dan
`divisive_summary.json`
`b9f6bc70f637c56b588256365fc99eca3a2f4bf90a71fdc4688a80e3995d9905`.

## Hasil arm AMF

| arm | silhouette internal | source NMI | stability ARI mean | ukuran min–maks |
|---|---:|---:|---:|---:|
| literal-adaptive | 0,0871 | 0,1905 | 0,9986 | 113–690 |
| literal-uniform | 0,1000 | 0,1987 | 0,9983 | 199–628 |
| balanced-adaptive | 0,0324 | 0,0876 | 0,8817 | 47–1.606 |
| balanced-uniform | −0,0091 | 0,1656 | 1,0000 | 55–1.723 |

Silhouette internal di atas adalah reference PCA-64 milik runner AMF. Untuk
perbandingan lintas framework, semua candidate kemudian dinilai ulang pada
reference `data_driven` 100-D yang sama dan pada discovery mask 3.579 citra;
600 citra holdout dikeluarkan agar sama dengan evaluasi G0.

## Comparison bersama

Angka berikut berasal dari
`semifinal/results/framework_comparison.csv`. `semantic_*` adalah proxy dari
concept bank SigLIP2, bukan ground truth manusia. Untuk HDBSCAN, metric yang
dipakai adalah DBCV; silhouette tidak dihitung.

| candidate | method | cluster/noise | silhouette/DBCV | semantic concentration | margin | coverage |
|---|---|---:|---:|---:|---:|---:|
| canonical data-driven | K-Means | 12 / 0% | 0,3051 | 2,4748 | 0,9891 | 0,9167 |
| K-Means fused | K-Means | 12 / 0% | 0,3052 | 2,4707 | 0,9853 | 0,9167 |
| Ward fused | Ward | 12 / 0% | 0,2951 | 2,4795 | 1,0202 | 0,9167 |
| K-Medoids PAM fused | K-Medoids | 12 / 0% | 0,3017 | 2,5147 | 1,0464 | 0,9167 |
| Leiden resolution 0,5 | Leiden | 13 / 0% | 0,2947 | 2,5040 | 1,0585 | 0,9231 |
| evidence accumulation | consensus | 12 / 0% | 0,3004 | 2,5076 | 1,0498 | 0,9167 |
| AMF literal-uniform | AMF | 12 / 0% | 0,2718 | 2,3403 | 0,9169 | 0,9167 |
| AMF literal-adaptive | AMF | 12 / 0% | 0,2248 | 2,1529 | 0,8134 | 0,9167 |
| AMF divisive | AMF top-down | 20 / 0% | 0,0720 | 1,9038 | 0,5390 | 0,5500 |
| HDBSCAN mcs=40 | HDBSCAN | 15 / 29,4% | DBCV 0,2763 | 2,7028 | 1,1307 | 0,8667 |

HDBSCAN mcs=40 terlihat paling tinggi pada semantic proxy karena ia menyisihkan
29,4% data sebagai noise. Itu bukan kemenangan yang apple-to-apple untuk task
yang membutuhkan interpretasi populasi utama; hasilnya dicatat sebagai kandidat
substructure, bukan pengganti partisi penuh.

## Keputusan hypothesis

1. **H1 — AMF lebih baik dari K-Means:** negatif untuk klaim improvement.
   AMF sangat stabil, tetapi arm terbaiknya (`literal-uniform`) masih di bawah
   K-Means fused pada silhouette reference (0,2718 vs 0,3052) dan semantic
   concentration (2,3403 vs 2,4707).
2. **H2 — weighting dan balancing penting:** positif sebagai diagnosis.
   Raw uniform lebih baik daripada raw adaptive; L2 balancing menurunkan
   compactness dan, pada arm adaptive, menurunkan stability. Jadi feature-scale
   AMF bukan detail kosmetik.
3. **H3 — divisive membuat substructure lebih jelas:** negatif pada konfigurasi
   notebook. Dua puluh daun menghasilkan silhouette 0,0720, coverage 0,55, dan
   semantic profile lebih lemah daripada flat AMF.
4. **H4 — framework lain memberi semantic clarity:** ada kandidat menarik,
   tetapi tidak ada pengganti yang menang secara keseluruhan. K-Medoids,
   evidence accumulation, dan Leiden 0,5 sedikit lebih tinggi pada semantic
   margin; K-Means fused tetap paling kompak dan tidak membuang citra.

## Apa yang sebenarnya ditemukan

Top concept untuk hampir semua framework didominasi function/device: printer,
laptop, audio, washing machine, phone/tablet, display, dan input peripheral.
Material dan condition hanya muncul sebagai kantong tertentu, misalnya PCB,
material baterai, dan bulk scrap. AMF tidak mengubah temuan ini menjadi cluster
material atau condition yang bersih.

Kesimpulan metodologisnya: AMF adalah framework yang valid dan defensible untuk
dimasukkan sebagai eksperimen pembanding karena mekanismenya jelas dan stabil,
tetapi pada dataset ini ia tidak mengalahkan baseline fused dan tidak mengatasi
dominasi function. Untuk lensa material, representation khusus seperti DMS46
lebih tepat dipertahankan sebagai eksperimen terpisah. Untuk partisi utama
data-driven, K-Means fused tetap pilihan paling sederhana dengan evidence paling
seimbang.

## Artefak dan reproducibility

- Preregistrasi: `semifinal/docs/98-PREREGISTRASI-AMF-DINOV2.md`;
- register autoresearch: `semifinal/docs/99-RESEARCH-REGISTER-FRAMEWORKS.md`;
- runner: `semifinal/scripts/run_amf_dinov2.py`;
- evaluator bersama: `semifinal/scripts/compare_frameworks.py`;
- comparison: `semifinal/results/framework_comparison.csv`;
- semantic profiles: `semifinal/results/framework_semantic_profiles.csv`;
- pairwise structure: `semifinal/results/framework_pairwise.csv`;
- visual gallery medoid/core: `semifinal/results/framework_semantic_gallery.pdf`
  (SHA-256 `395284140e5c8ac424988103a25d4054d61ac7aa8afbe9d488644e90fc71ff13`);
- metadata dan hash: `semifinal/results/framework_comparison_metadata.json`.

Hash comparison saat laporan ini dibuat: `framework_comparison.csv`
`a4da5376d2fe25e76614b3cf10392e9f17c69d3b09e9f1a83fdbcd7489f449f4`,
`framework_semantic_profiles.csv`
`58d4825a09f3340932326e49e0d6101ccd041b78449d0a6918a74e4018f1dde1`, dan
`framework_pairwise.csv`
`19fa1088fc99e94f005a478430bc497eefa010de69db5fb6a270f37411e960a5`.
