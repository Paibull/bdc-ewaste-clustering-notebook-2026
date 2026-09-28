# Hasil Frozen Transfer dan Dukungan Material pada Fused K-Means

## Rekonstruksi dan mapping beku

Transform PCA direkonstruksi hanya pada 3.579 citra discovery BDC; K-Means dijalankan ulang semata untuk merekonstruksi centroid dengan `K=12`, `n_init=20`, dan `random_state=0`. Tidak ada PCA, centroid, atau clustering yang di-fit pada Bangladesh. Reconstruction menghasilkan **ARI=1,0000** untuk fused dan **ARI=1,0000** untuk RADIO; alignment Hungarian menunjukkan ID cluster identik dengan partisi tersimpan.

| Family eksternal | Fused cluster | RADIO cluster |
|---|---:|---:|
| Battery | 1 | 4 |
| Mobile | 5 | 3 |
| PCB | 8 | 10 |
| Input peripheral | 0, 7 | 2, 11 |

Cluster lain diperlakukan sebagai `out_of_taxonomy`. Mapping ini ditetapkan dari profil semantic BDC dan tidak diubah setelah scoring. Keyboard dan Mouse digabung sebagai `input_peripheral`.

## Frozen external transfer

Embedding Bangladesh memakai transform harmonisasi BDC yang sama: center crop persegi, resize 150×150 dengan LANCZOS, lalu JPEG quality 75 dan subsampling 4:2:0. Terdapat 2.153 key anotasi unik: 710 known-set dan 1.443 out-of-taxonomy; empat gambar tanpa anotasi dikecualikan. Bootstrap paired dan distratifikasi menurut family, 2.000 draw, seed 20260927. Top-2 memakai empat family frozen.

| Estimator | Micro Top-1 (95% CI) | Macro Top-1 (95% CI) | Top-2 (95% CI) | MRR (95% CI) | Mean rank (95% CI) |
|---|---:|---:|---:|---:|---:|
| Fused | 0,908 (0,889–0,928) | 0,905 (0,883–0,926) | 0,961 (0,946–0,973) | 0,948 (0,936–0,959) | 1,131 (1,100–1,163) |
| RADIO | 0,880 (0,859–0,901) | 0,867 (0,843–0,890) | 0,938 (0,921–0,954) | 0,930 (0,917–0,942) | 1,183 (1,149–1,218) |

| Family | N | Fused recall (95% CI) | RADIO recall (95% CI) |
|---|---:|---:|---:|
| Battery | 148 | 0,885 (0,831–0,932) | 0,628 (0,547–0,703) |
| Mobile | 151 | 0,881 (0,828–0,927) | 0,940 (0,901–0,974) |
| PCB | 205 | 1,000 (1,000–1,000) | 1,000 (1,000–1,000) |
| Input peripheral | 206 | 0,854 (0,806–0,898) | 0,898 (0,859–0,937) |

Paired delta fused−RADIO: Micro Top-1 **+0,0282** (95% CI +0,0099 hingga +0,0493), Macro Top-1 **+0,0384** (+0,0172 hingga +0,0619), dan MRR **+0,0180** (+0,0070 hingga +0,0298). Fused lebih baik secara agregat, terutama pada Battery; RADIO lebih baik pada Mobile dan Input peripheral. Kedua estimator mencapai 100% recall pada PCB di sampel ini.

Novelty memakai unknown sebagai kelas positif dan tidak memilih threshold. Prevalensi unknown—juga AP chance baseline—adalah 0,670.

| Estimator | Sinyal novelty | AUROC | AUPRC |
|---|---|---:|---:|
| Fused | Nearest-known-family distance | 0,905 | 0,934 |
| Fused | Negatif margin family peringkat 1–2 | 0,848 | 0,874 |
| RADIO | Nearest-known-family distance | 0,920 | 0,943 |
| RADIO | Negatif margin family peringkat 1–2 | 0,894 | 0,913 |

Kedua sinyal memisahkan known dan out-of-taxonomy dengan baik pada set ini, tetapi RADIO lebih kuat untuk novelty; hasil itu tidak mengubah verdict transfer known-set.

## Hubungan material dengan fused K-Means

Fraction DMS46 dan partisi fused join tepat pada **4.179 key unik**. Enam fraction berjumlah satu dengan galat absolut maksimum `8,3×10⁻⁸`. Pada 3.579 discovery rows, `sqrt(fraction)` yang dinormalisasi L2 memberi between-cluster **R²=0,1374**. Permutasi 999 kali di dalam strata sumber `web`/`field` menghasilkan **p=0,001** (null 95th percentile 0,0075). Jadi profil material berbeda menurut cluster perangkat di luar variasi antar-sumber; besar asosiasinya terbatas dan bukan ukuran akurasi material.

Contoh profil yang paling informatif:

| Cluster (nama semantic) | N | Fraction mean terbesar | Enrichment terhadap mean global | Entropy mean (nat) |
|---|---:|---|---:|---:|
| 9 (printer) | 295 | Plastic 0,542 | 1,64× | 0,703 |
| 0 (input_peripheral) | 316 | Plastic 0,452 | 1,37× | 0,655 |
| 4 (laptop) | 376 | Plastic 0,425 | 1,29× | 0,929 |
| 1 (battery) | 340 | other_unknown 0,489 | 1,20× | 0,655 |
| 8 (pcb) | 371 | other_unknown 0,386 | 0,95× | 0,692 |

Global mean `other_unknown` adalah 0,407, plastic 0,330, glass 0,100, paper/cardboard 0,098, metal 0,062, dan rubber 0,004. Komposisi paling mirip menurut Jensen–Shannon distance (base 2) adalah cluster 7 `input_peripheral` dengan 11 `audio` (0,080), lalu cluster 2 `bulk_scrap` dengan 3 `washing_machine` (0,090). Ini menunjukkan sebagian cluster perangkat berbeda memiliki campuran material serupa; sebagian besar profil juga tetap campuran dan `other_unknown` besar.

Analisis ini memberi dukungan semantic internal bahwa cluster fungsi/perangkat berkaitan dengan komposisi material DMS46. Ini **bukan external validation taxonomy**, bukan pixel-level material ground truth, dan bukan bukti bahwa fraction DMS46 adalah estimasi komposisi objek yang tervalidasi.

## Keputusan dan batas

Verdict frozen transfer: **`FUSION_IMPROVES_TRANSFER`**. CI fused Macro Top-1 berada jauh di atas chance 0,25 dan lower CI paired delta Macro Top-1 positif. Kesimpulan dibatasi pada 710 gambar berlabel YOLO dalam empat family yang mapping-nya sudah dibekukan; keyboard dan mouse sengaja digabung. Novelty dinilai tanpa threshold, pada dataset dengan proporsi unknown 67,0%. Material support mengukur asosiasi pada discovery BDC dan tidak boleh disebut validasi external taxonomy.

Embedding dijalankan pada A10G; runtime fungsi 302,63 detik (GPU teridentifikasi NVIDIA A10). Modal profile `faizakbar2301`, Volume `bdc-data`, App `bdc-bangladesh-frozen-transfer-20260927`, App ID `ap-RCtFcSrFdGNZUY5q12JU96`, FunctionCall ID `fc-01M3H2GFB3DYJPXH2RSK3PE7BP`.
