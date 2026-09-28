# 110 — Hasil K-resolution semantic sweep

## Pertanyaan dan protokol aktual

Eksperimen menguji apakah hasil yang biasa-biasa saja terutama muncul karena semua estimator dipaksa memakai `k=12`. Fit hanya menggunakan 3.579 baris discovery; 600 holdout tidak dipakai untuk memilih `k`. Grid coarse `[4,6,8,10,12,14,16,18,20,22,24]` diikuti tetangga `k*±1`; pemilihan menggunakan skor terkunci pada evaluator. Sebanyak 976 assignment valid dari delapan family menghasilkan 485 kandidat evaluasi dan 34 arm terpilih. Evaluasi final memakai common data-driven space. WordNet-SigLIP2 cache hanya dipakai untuk diagnostic vocabulary post-hoc.

Semua family selesai: classical G0, AMF, TURTLE project adapter, FGW prototype, MGE reimplementation, MCSF adapter, TEMI adapter, dan TAC adapter. Tidak ada Modal job yang perlu diulang. Finalist null test memakai 199 within-source permutations dan maximum-over-K. P-value terkoreksi untuk 34 arm terpilih semuanya `0,005`; hasil refresh menghitung ulang tahap ini dari checkpoint lokal setelah perbaikan group-key evaluator.

## Selected `k`

| Family | Arm dan selected `k` |
|---|---|
| Classical G0 | Ward fused 12; PAM fused 14; K-Means SigLIP 12, Radio 12, fused 14, EVA02 12, DINOv3 12, DINOv2 14, ConvNeXt 12, AIMv2 12; diagonal GMM fused 14; evidence accumulation 14 |
| AMF | MB8 uniform 12, MB8 adaptive 12, MB4 uniform 12, MB4 adaptive 12 |
| TURTLE adapter | MB8 12; MB4 12 |
| FGW prototype | MB8 12; MB4 12 |
| MGE reimplementation | single-link 16; average-link sensitivity 12 |
| MCSF adapter | MB3 14 |
| TEMI adapter | SigLIP2 14; Radio 12; MSN 10; Franca 10; FashionSigLIP 16; DINOv2 8 |
| TAC adapter | SigLIP2 head 12, SigLIP2 concat 12; Radio+SigLIP2G initial/head/concat 14 |

## Pembanding dan keputusan

| Partisi | `k` | Silhouette | Stability ARI | Source NMI | Function AMI | Min–max ukuran klaster |
|---|---:|---:|---:|---:|---:|---:|
| Fused 8-view K-Means existing | 12 | 0,305210 | tidak dihitung ulang | 0,164801 | 0,783534 | 237–457 |
| Fused 8-view K-Means fair, grid | 12 | 0,304772 | 0,999584 | 0,159973 | 0,788228 | 238–461 |
| Fused 8-view K-Means selected-K | 14 | 0,312245 | 0,994943 | 0,158439 | 0,799485 | 130–461 |

Untuk fused K-Means, `k=14` memberi silhouette `+0,00747` dan function AMI `+0,01126` dibanding run fair `k=12`, dengan source NMI hampir sama. Jadi mematok `k=12` memang sedikit membatasi baseline ini, tetapi perubahan `k` saja tidak memberi dasar untuk mempromosikan estimator advanced. Tak satu pun dari 34 arm lolos seluruh promotion gate; p-value signifikan saja tidak cukup, dan syarat gain semantik, stability, kompaksi, source, serta holdout tetap berlaku.

Tiga arm TAC yang dipilih (`siglip2_concat` k=12, `radio_siglip2g_initial` k=14, `radio_siglip2g_concat` k=14) tidak memiliki seed-stability ARI. Ketiganya ditandai tidak valid untuk promotion; selected `k`-nya hanya dilaporkan sebagai keluaran tahap seleksi, bukan kandidat yang memenuhi hard gate.

Enam arm TEMI terpilih juga tidak lolos hard gate stability: mean seed ARI berada pada `0,210–0,356`, di bawah ambang `0,60`. Jadi angka `k` TEMI tetap dicatat untuk menjawab sensitivitas resolusi, tetapi partisinya tidak eligible untuk klaim kandidat stabil.

Verdict: **`MULTI_RESOLUTION_ATLAS_SUPPORTED`**. Top-concept node yang bertahan pada `k` berdekatan dan muncul di minimal dua family: `phone_tablet`, `battery`, `laptop`, `washing_machine`, `desktop`, `printer`, `input_peripheral`, `audio`, `display`, `cooking_appliance`, `electronic_component`, dan `camera`. Ini mendukung atlas lintas-resolusi, bukan kemenangan satu metode. Fused K-Means terpilih-K tetap baseline estimator; belum ada advanced method yang dipromosikan.

Diagnostic cached vocabulary bersifat exploratory. Rata-rata top-50 capture / normalized entropy: outlet/socket `0,503 / 0,558`; plug `0,375 / 0,629`; power strip `0,372 / 0,646`. Nilai enrichment permutation `p=0,005–0,009`, tetapi fragmentasi outlet/plug/power-strip masih lebih tinggi daripada keyboard/mouse. Bukti ini belum menunjukkan stop-kontak, plug, dan power strip sebagai node yang terpisah dan stabil.

## Material support, runtime, dan biaya

DMS46 dihitung post-hoc saja, tidak memengaruhi pemilihan. Untuk 8 representative (satu per family) dan existing fused 8-view baseline, AMI material berada pada `0,068–0,114`, partial R² material setelah source `0,082–0,146` (`p=0,005`), dan within-minus-between cosine gap `0,025–0,050` (`p=0,005`). Ini mendukung adanya koherensi material moderat, bukan validasi semantik independen.

Jumlah runtime worker yang tercatat `10.527` detik; biaya Modal kasar **USD 4,55**, dihitung dari runtime worker dan tarif CPU/GPU/RAM akun. Estimasi mengecualikan startup, waktu idle container, serta transfer Volume. Total wall time dari mulai kerja sampai final null refresh sekitar **177 menit**, termasuk kegagalan evaluator lokal dan resume; semua compute Modal sendiri selesai.

## App dan FunctionCall

Status semua App: selesai; tidak ada crash-loop yang tersisa. App alias dan semua launcher FunctionCall ID:

| App | Coarse shards | Parity/benchmark | Refinement |
|---|---|---|---|
| `bdc-kroll-classic-20260927` | `fc-01M3GWQ02XJH7H0R3HFC55XP14`, `fc-01M3GWQ0B8HR2DP0GJBXTXBRB4`, `fc-01M3GWQ007PTH87Z4EKHZQQ4TE`, `fc-01M3GWQ1KCCNBAD4PDMBA8DS7S` | — | `fc-01M3GYY62MB3KDTNX36RD7W99R` |
| `bdc-kroll-mge-20260927` | `fc-01M3GWQNQ61J3PKVN8WT0XS79S`, `fc-01M3GWQNME9T8J817VMSE1C7SH`, `fc-01M3GWQNRXFT3R8037SKEZJ3B0`, `fc-01M3GWQNYVZNJAR51405M3RYCF` | — | `fc-01M3GYY65FRK3Y6QKB9PKYZ498` |
| `bdc-kroll-mcsf-20260927` | `fc-01M3GWGRTWBN01G73MDR0HNGD2`, `fc-01M3GWQ0VX5A4JNPCE2VT3FCKA`, `fc-01M3GWQ0VKA36Q55P86QVK7TAM`, `fc-01M3GWQ0AE5ZWJB01SE81QZS9X` | — | `fc-01M3GYY6M4SFX6EE646JWT5WW6` |
| `bdc-kroll-amf-20260927` | `fc-01M3GWQNV13AWRCA3D99PKGPFV`, `fc-01M3GWQNZ5B354CYTZK95CJQM0`, `fc-01M3GWQP5F8N4D8SDAXTPSPTPX`, `fc-01M3GWQP5ER7PR1GPX4RDBYK1C` | `fc-01M3GVM1Q1XSGSKMRND9HT25HX` | `fc-01M3GYY6N7HAK7Z9E0G82C5YPK` |
| `bdc-kroll-fgw-20260927` | `fc-01M3GWQPJXC5WQYZ61MZ1HCE7K`, `fc-01M3GWQPN38ZGXSCXZ7BK2RS7M`, `fc-01M3GWQPS3SK1N64X9TQEQZPFS`, `fc-01M3GWQP9WBYCM0C9XS9R7PYRB` | `fc-01M3GVM5FYDH96DTVJZ03WYTWQ` | `fc-01M3GYY6MXZBY8EKTTM4WX33K8` |
| `bdc-kroll-turtle-20260927` | `fc-01M3GW28Y3B7KAZDB60Y3NMPFB`, `fc-01M3GW29FS70ZCN96VX1QX3BT4`, `fc-01M3GW2A4GGPHDPRK3HRGG8AKY`, `fc-01M3GW2AQVQDF68R5P56GMZXN5` | `fc-01M3GVNR87MM340WGWZM80F0VZ` | `fc-01M3GYY6MSVQNQCYKBMW6PCBT3` |
| `bdc-kroll-temi-20260927` | `fc-01M3GWRG08KZMZGV46QPBF8F6R`, `fc-01M3GWRGNWGV3XGBQ7FKNEXFMC`, `fc-01M3GWRG7TNC56B63GCKNDT39X`, `fc-01M3GWRGJA2MHD5RGW4H8RNHT8` | `fc-01M3GVWVP2K7W85R6V55PVG148`, `fc-01M3GVWWANWSRQ11V828TAREJG`, `fc-01M3GVWWY0NJ3YXZK8R2B7NE0B`, `fc-01M3GVWXFYQ2QJNHVYS5ZJT4RE` | `fc-01M3GYY6FVRVJ06SAAW9A9NQG6` |
| `bdc-kroll-tac-20260927` | `fc-01M3GWRH2GC0W3BK766XNYX8PR`, `fc-01M3GWRH68M07TWQFP8K5RRJB7`, `fc-01M3GWRGPQ7VB7KJGQGY7N5MMB`, `fc-01M3GWRGTZSBMMETGXG06ZZ4W4` | `fc-01M3GVWVA1SF5SA6MDHQYQKF7V` | `fc-01M3GYY6DNE7H3SEST6T9Z4EV3` |

Per-launcher App/FunctionCall metadata juga ada di `results/k_resolution_sweep/20260927/launcher_*.json`.

## Batas bukti

Label konsep berasal dari frozen anchor scores, bukan anotasi manusia; label tersebut tidak membuktikan kategori reuse/recycle atau kondisi perangkat. Gap vocabulary dipilih post-hoc dan tidak masuk ranking. TURTLE, FGW, MGE, TEMI, TAC, dan MCSF yang dijalankan merupakan adapter/reimplementation proyek sesuai kontrak runner, bukan seluruhnya reproduksi resmi end-to-end. Promotion gate konservatif tidak dilonggarkan setelah melihat hasil.

Data lengkap: [selected-k dan gate](../results/k_resolution_sweep/20260927/selected_k_by_method.csv), [semua kandidat](../results/k_resolution_sweep/20260927/all_candidates.csv), [Pareto front](../results/k_resolution_sweep/20260927/pareto_front.csv), [semantic nodes](../results/k_resolution_sweep/20260927/semantic_nodes.csv), [gap vocabulary](../results/k_resolution_sweep/20260927/gap_vocabulary_diagnostic.csv), [material support](../results/k_resolution_sweep/20260927/material_support.csv), [pairwise partitions](../results/k_resolution_sweep/20260927/pairwise_selected_partitions.csv), [summary](../results/k_resolution_sweep/20260927/summary.json).
