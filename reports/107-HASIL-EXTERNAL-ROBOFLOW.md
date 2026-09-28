# Hasil validasi external facet

Tanggal: 27 September 2026  
Status: **Roboflow function transfer `EXPLORATORY_POSITIVE`; condition representation `NEGATIVE`; external DMS46 material mapping `BLOCKED`.**

## Kesimpulan singkat

Pada subset Roboflow v44 yang dipilih sebelum scoring, assignment ke 11 fixed
function clusters memiliki agreement yang kuat dengan 14 UNU-KEY (`AMI 0,6256`,
999-permutation `p=0,001`). Hasil tetap serupa pada balanced sensitivity set dan
ketiga split resmi. Ini mendukung transfer struktur function/device pada subset
ini; bukan bukti bahwa target clusters merupakan taxonomy final atau bahwa
material dan condition sudah tervalidasi external.

Material dan condition memberi jawaban berbeda. Apple DMS46 tidak dapat
membedakan lima label material TECNALIA melalui taxonomy output resminya, jadi
external material score tidak dihitung. Ablation condition tidak menemukan
representation baru yang melewati gate peningkatan `+0,05`; condition tetap
diagnostic facet, bukan standalone taxonomy.

## Roboflow: external function-taxonomy validation

### Protokol yang dijalankan

Dataset official Roboflow “E-Waste Dataset”, version 44, COCO, CC BY 4.0,
diunduh langsung ke Modal Volume `bdc-data` dan tidak disalin ke local machine.
Versi, mapping, model, preprocessing, split, null, metric, dan gate telah
dibekukan dalam `106A-PREREGISTRASI-ROBOFLOW-EXTERNAL.md` sebelum embedding.

Subset mencakup 18 label perangkat utuh yang dipetakan eksplisit ke 14 UNU-KEY.
Pemilihan mensyaratkan sedikitnya 100 anotasi sumber dan direct UNU mapping;
kelas komponen/material tidak dimasukkan. Gambar berlabel ambigu dikeluarkan,
exact duplicate dihapus, dan duplikat dengan label bertentangan akan seluruhnya
dikeluarkan. Hasil preflight: 4.974 gambar unik eligible, seluruh 14 UNU-KEY
terwakili, 1.199 gambar multi-UNU ambigu dikeluarkan, 293 duplikat label sama
dihapus, dan tidak ditemukan duplikat konflik. Eligible set terdiri dari 3.377
train, 1.026 valid, dan 571 test image; split hanya digunakan sebagai kelompok
evaluasi.

Representation adalah frozen `ViT-SO400M-16-SigLIP2-384`. Input mengikuti
harmonisasi target: center crop, resize 150 px dengan LANCZOS, JPEG quality 75
dengan subsampling 4:2:0, lalu transform resmi model. Embedding external
diproyeksikan memakai mean dan span basis target yang tetap. Assignment memakai
centroid dari 3.579 target discovery keys pada 11 function clusters; 600 locked
audit keys tidak digunakan untuk membentuk centroid. Tidak ada training,
fine-tuning, clustering ulang, atau perubahan target partition.

### Hasil

| Scope | N | UNU coverage | AMI | AMI null q97,5 | AMI permutation p | ARI | NMI | Weighted cluster purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary, imbalanced | 4.974 | 14/14 | **0,6256** | 0,0015 | **0,001** | 0,4948 | 0,6279 | 0,6892 |
| Balanced sensitivity, 79/key | 1.106 | 14/14 | **0,6170** | 0,0066 | **0,001** | 0,4105 | 0,6270 | 0,5280 |
| Official train | 3.377 | 14/14 | 0,6103 | — | — | 0,4843 | 0,6138 | 0,6544 |
| Official valid | 1.026 | 14/14 | 0,6694 | — | — | 0,6025 | 0,6800 | 0,7729 |
| Official test | 571 | 14/14 | 0,6526 | — | — | 0,5110 | 0,6710 | 0,7636 |

Null mengacak label UNU dengan frekuensi tetap dan mempertahankan assignment
cluster; 999 replikasi, seed `20260926`. Gate primary preregistrasi lulus:
`AMI > null q97,5` dan `p <= 0,025`. Balanced sensitivity juga berada jauh di
atas null. Weighted purity turun dari 0,6892 menjadi 0,5280 pada sampling
seimbang; karena itu purity dibaca bersama AMI/ARI/NMI, bukan sebagai satu-satunya
skor.

Stability terhadap 100 bootstrap centroid target:

| Metric | Mean | SD | Bootstrap 95% range |
|---|---:|---:|---:|
| AMI | 0,9712 | 0,0050 | 0,9611–0,9796 |
| ARI | 0,9807 | 0,0044 | 0,9731–0,9878 |

Ini mengukur sensitivitas assignment external terhadap perubahan kecil pada
centroid target, bukan ketidakpastian populasi atau semantic correctness.

### Interpretasi dan batas klaim

Hasil menunjukkan bahwa pada 18 label terpilih dan 14 UNU-KEY ini, geometri
function cluster target selaras dengan taxonomy perangkat external lebih baik
daripada null permutation. Keselarasan muncul pada seluruh split yang dilaporkan
dan tidak bergantung hanya pada kelas terbesar. Statusnya tetap
`EXPLORATORY_POSITIVE` karena validasi dibatasi oleh subset eligible (25,37%
image sumber), pemetaan label-ke-UNU, dan satu versi dataset.

Klaim yang diizinkan: assignment function out-of-sample pada subset dan protokol
ini menunjukkan agreement dengan UNU-KEY di atas null permutation. Tidak boleh
disebut Macro-F1/accuracy classifier, bukti untuk seluruh 77 kelas, taxonomy
final, atau validasi material/condition. Confusion matrix dan daftar prediction
menyediakan audit label per gambar.

## Apple DMS46: material external gate

### Hasil gate

External benchmark TECNALIA RGB berisi 1.088 crop, 13 scene, dan lima material
class. DMS46 taxonomy tidak menyediakan pemisahan yang diperlukan untuk mapping
valid ke seluruh lima label: kategori official yang tersedia menggabungkan
kelas alloy ke `Metal`. Karena mapping tidak dapat dibekukan tanpa mengarang
padanan, external DMS46 inference dan score dihentikan sesuai gate. Status
branch: `BLOCKED`; tidak ada metric external yang boleh dilaporkan.

Ini tidak membatalkan diagnostic DMS46 full-target yang sudah selesai
sebelumnya. Descriptor fraction 46 kelas membentuk composition clusters dengan
dispersion 0,3370 dibanding null 0,5488–0,5510 (`p=0,0099`) dan stability ARI
0,9925, tetapi source masih dapat diprediksi (AUROC 0,7894) dan belum ada human
semantic validation. Jadi DMS46 tetap candidate composition descriptor,
bukan model material yang terbukti superior atau ground truth.

## Visible-condition representation ablation

DefectAtlas-20 memakai 13.233 image, 41 camera groups, group-aware folds yang
sudah ada, dan locked target audit hanya 23 image. Model atau hyperparameter
tidak ditune pada target audit. Baseline SigLIP2 group-CV AUROC adalah 0,9271;
hasil representation yang diuji:

| Representation | Group-CV AUROC (95% CI) | Fold AUROC range | Average precision | Balanced accuracy | Macro-F1 | Target-audit AUROC (n=23) | Target-audit balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| SigLIP2 existing pipeline | 0,9269 (0,9185–0,9356) | 0,9163–0,9414 | 0,9421 | 0,8518 | 0,8500 | 0,9821 | 0,8661 |
| DINOv2 frozen | 0,8404 (0,8247–0,8533) | 0,8133–0,8645 | 0,8714 | 0,7562 | 0,7535 | 0,8750 | 0,8661 |
| Low-level edge/texture/image-quality | 0,5560 (0,5434–0,5663) | 0,5220–0,5679 | 0,5683 | 0,5420 | 0,5393 | 0,4286 | 0,5134 |

Gate mensyaratkan candidate baru menaikkan AUROC sekurangnya `+0,05` terhadap
baseline SigLIP2 pada group-CV dan sekurangnya `+0,05` terhadap concept baseline
pada target audit. Existing SigLIP2 adalah baseline, bukan candidate baru;
DINOv2 dan low-level representation tidak memenuhi gate. Target audit hanya 23
image, sehingga nilainya dibaca bersama group-CV dan fold spread, bukan sebagai
bukti mandiri. Status ablation: `NEGATIVE` untuk promosi representation baru.
Condition tetap diagnostic overlay, bukan standalone taxonomy yang sudah
terverifikasi.

## Kaitannya dengan facet coherence

Analisis preregistrasi 101/102 menjawab pertanyaan berbeda dari ablation di atas:
apakah fixed data-driven clusters berasosiasi dengan facet setelah
`function × source` dikontrol. Hasilnya:

| Facet | Partial R² | Permutation p | Discovery–holdout profile correlation | Status gate |
|---|---:|---:|---:|---|
| DMS46 material composition | 0,0339 | 0,001 | 0,6818 | Lulus |
| Visible-condition proxy | 0,2434 | 0,001 | 0,9692 | Lulus, secondary diagnostic |
| Joint | 0,0556 | 0,001 | 0,7875 | Lulus |

Ini adalah bukti residual association dari descriptor/proxy yang ada; bukan
material accuracy, condition accuracy, atau causal effect. Karena condition
representation ablation tidak melampaui promotion gate, condition tetap
secondary diagnostic. Material composition dapat disebut facet pendukung,
dengan keterangan bahwa DMS46 external ground-truth comparison masih blocked.

## Keputusan untuk main thesis

Hasil baru memperkuat, tetapi tidak mengganti, narasi utama:

1. Akuisisi dapat menghasilkan struktur semu pada generic clustering; input
   harmonization tetap merupakan kontribusi metodologis utama.
2. Setelah harmonisasi, fixed function/device partition memiliki transfer
   agreement external dengan UNU-KEY yang melampaui null pada subset Roboflow.
3. Material composition dan visible condition lebih tepat diperlakukan sebagai
   cross-cutting facets/diagnostic overlays setelah function dan source
   dikontrol, bukan sebagai taxonomy hard yang diklaim sudah benar.
4. Validasi semantic manusia serta external material mapping tetap membatasi
   klaim komposisi material.

| Branch | Evidence / gate | Status | Permitted claim | Prohibited claim |
|---|---|---|---|---|
| Roboflow UNU-aligned | Primary AMI 0,6256 > null q97,5 0,0015; p 0,001; balanced sensitivity juga lulus | `EXPLORATORY_POSITIVE` | Function-cluster assignment selaras dengan subset UNU taxonomy di atas null | Taxonomy final; semua 77 kelas; material/condition validation |
| DMS46 external material | Official mapping menggabungkan alloy ke `Metal`; five-class mapping tidak valid | `BLOCKED` | Tidak ada external DMS score | DMS46 superior pada material ground truth |
| DMS46 full composition (sebelumnya) | Dispersion 0,3370 vs null; belum human-validated | `EXPLORATORY_POSITIVE` / `DIAGNOSTIC_ONLY` | Composition descriptor menunjukkan struktur yang berbeda dari null | Kandungan material benar atau ground-truth accuracy |
| Condition representation ablation | Tidak ada candidate melewati gate `+0,05` | `NEGATIVE` untuk promosi | Existing condition score dipakai sebagai diagnostic facet | Condition standalone taxonomy telah tervalidasi |
| Facet coherence (sebelumnya) | Material/condition residual gate lulus pada fixed clusters | `EXPLORATORY_POSITIVE` | Residual association di luar function × source pada descriptor/proxy | Causal effect atau accuracy seluruh target |

## Provenance dan reproducibility

- Dataset: Roboflow `electronic-waste-detection/e-waste-dataset-r0ojc`, v44,
  COCO, CC BY 4.0; 19.604 image, 28.917 annotation, 77 source classes.
- Archive SHA-256: `5b174ba1a5c576f8d888e6f567e77540748a71577c67f3dcd5dfb922062c82b8`.
- Acquisition metadata SHA-256:
  `32b25cd83296c6b9e4e270b7839f3566a820e90d879f24007f1317164408a1bc`.
- Fixed class mapping SHA-256:
  `56230ac9481107f7de2fa6a4e0bf779f8d83959433c88de16e3c1c3cefb8f1ed`.
- Roboflow preregistration SHA-256:
  `b5f6339a82fa08dae47cf29dc2e4aeba0dda9b1fd1bb312e878b3388a6beb16f`.
- Target transfer bundle SHA-256:
  `6b0a2d66c049de0e215078920de2d1bb14b3c8424f0315ca7f5db1f97c2b14a3`;
  function-span reproduction max absolute error `0`.
- Command: `python -m modal run --profile faizakbar2301 --detach semifinal/scripts/run_roboflow_external_validation.py`.
- Modal App `ap-VqIElbTs0gGbgumqPUGL7V`; FunctionCall
  `fc-01M3F8P3ABZX8XTA5D2T7RT8ZG`; start 26 Sep 2026 23:27:30 WIB; finish
  27 Sep 2026 00:01:58 WIB; duration 2.055,25 s.
- Dataset path: `/data/externals/roboflow_e_waste_v44_verified`; outputs:
  `/data/externals/roboflow_e_waste_v44_verified/function_validation/attempt_02`.
- Primary local audit artifacts: `results/roboflow_external_validation/`;
  prediction and manifest each have 4.974 unique matching keys. Metrics, both
  confusion CSVs, preflight, prediction, manifest, and final status checksums
  match the Volume metadata. The status-checksum write order was corrected in
  the runner, and corrected metadata was uploaded back to the Volume.
- Metrics SHA-256:
  `6402e68ce09770e3be49a23c4ce49286195e72ddc9d9abb1e7e632af844739ac`.
- Metadata SHA-256:
  `bb8c0ac7d60674adaf8276f2535435d8bb8d65e765f189f0a9cefefdab230cce`.
- Prediction SHA-256:
  `b69f88c4dc8df55d7777c64262207b326aeeab0b40c2642e12bc4d731fa95d23`.

Tidak ada perubahan pada dataset target, harmonization, locked holdout, split,
seed historis, cluster names, scorer, atau final partition. Perbandingan metric
antarbranch tetap dipisahkan karena mengukur pertanyaan yang berbeda.
