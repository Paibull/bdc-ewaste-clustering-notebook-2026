# Hasil Eksekusi Acquisition-Aware Multiview Clustering

Tanggal laporan: 27 September 2026 (WIB)  
Preregistrasi: [`107-PREREGISTRASI-ACQUISITION-MULTIVIEW-PARALLEL.md`](107-PREREGISTRASI-ACQUISITION-MULTIVIEW-PARALLEL.md)  
Run group: `20260926T180402Z`  
Status: **seluruh job final berhasil; artefak diunduh dan checksum diverifikasi; evaluator bersama selesai**.

## Ringkasan

Eksperimen menggunakan tujuh Modal App pada profile `faizakbar2301`, Volume
`bdc-data`, dan 4.179 key citra. Semua fitting hanya memakai 3.579 baris
discovery; 600 baris holdout tidak ikut fitting dan tetap diberi sentinel `-2`.
Tidak ada baseline lama yang dijalankan ulang dan CoMVC tidak dijalankan.

Hasil utama **tidak melewati gate**. Multiplex-4 native+bridge pada parameter
primer hanya menghasilkan 5 cluster, silhouette common-space 0,0908, dan mean
ARI antarseed 0,6462. Target validitas yang dikunci adalah 10–16 cluster,
silhouette sekurangnya 0,285210, dan mean ARI sekurangnya 0,80. Source NMI yang
sangat rendah (0,0073) tidak cukup menjadi bukti integrasi yang baik: cluster
terlalu sedikit, sementara cross-source kNN rate hanya 0,0361. Karena itu,
fused K-Means beku tetap menjadi estimator utama; Multiplex dilaporkan sebagai
hasil negatif/ablation, bukan pemenang.

## Data, provenance, dan validasi

| Artefak | SHA-256 |
|---|---|
| `semifinal/artifacts/ewaste_emb.npz` | `f18a4f4705f05aaa1124921063c8a61dcb9e3b9d5171e0bdead693f731947d8b` |
| `semifinal/artifacts/ewaste_emb_harm_resjpeg.npz` | `90453ebea61d6752a08c04f3600e55ee4256ddf4992ccef20969522dc955c767` |
| `semifinal/artifacts/ewaste_keys.npz` | `48f197459235414cc4d063c58d98e40f875cc61d9c5d5f2b1088a0ef4343c84d` |
| `semifinal/artifacts/semantic_g0_partitions.npz` | `9b9fd7bfb8cd692a262b7cec7e27aa777cdb6caadd020c0adaf004f1e8852922` |
| `semifinal/results/imgstats.csv` | `9c47b2f8a428fdbf9df4fa1a10f7c93fc89c7fc8caf695a89a7afd75b687f221` |
| `semifinal/artifacts/lens_spaces.npz` | `6242a4266f10d22e878568514785dddadb25fa85646c705c809a0cd258eaaa2a` |
| `semifinal/artifacts/concept_axes.npz` | `ffe7cfa2bc7f12400bfe3c70e01e70ba9fc47dd352fc133b2fe04f91895d9407` |
| `semifinal/results/clusters_lenses_final.csv` | `f09e72daace2d3a8f4635d54e388cb0adf4553e57e5f31e7a985cc741023b60a` |

| Provenance | SHA-256 |
|---|---|
| Preregistrasi 107 (tidak diubah) | `2a5e81dfb478bd056d2379dd71ca15cd6b2a005a23eda8029a081b15b819ca08` |
| Input manifest | `cd2f05ea8ee527d4b19f43aa04c6de86de4f5ebe20ce7b073b9b583489059b9b` |
| Runner final, termasuk koreksi resume/checksum | `a6c0aa7a047d0c49d7ff4ad795196cf5dc48e5d4338b9796a4352eb2ab2d14bb` |
| Evaluator bersama | `9ddcca2fd32bd7aa75631584e7e44cd5be90d281c4fcaa89039ffff223feb3f8` |
| Baseline beku `framework_comparison.csv` | `a4da5376d2fe25e76614b3cf10392e9f17c69d3b09e9f1a83fdbcd7489f449f4` |

Evaluator selesai pada 27 September pukul 03:21:27 WIB. Ia memvalidasi 130
kandidat baru dan memuat 31 kandidat baseline. Metadata menyatakan
`unmodified_baseline_outputs: true`; SHA-256 baseline tetap sama dengan
preregistrasi. Setiap checksum artefak kandidat, urutan key/split, mask
discovery/holdout, panjang label, dan sentinel holdout lolos pemeriksaan.

## Eksekusi Modal

Perintah full yang digunakan untuk setiap entrypoint:

```powershell
python -m modal run --detach --profile faizakbar2301 `
  semifinal/scripts/run_parallel_frameworks.py::<entrypoint> `
  --stage full --run-group 20260926T180402Z
```

Enam job awal berjalan tumpang tindih pada sekitar 01:34 WIB; Multiplex-4
selesai lebih awal, sedangkan TURTLE menjadi job terlama. Jadi eksekusi akhir
memang paralel, tetapi kartu crash-loop pada dashboard berasal dari App ID
upaya pertama, bukan App ID sukses di tabel berikut.

| Metode | App sukses / FunctionCall sukses | Mulai–selesai WIB | Runtime | Kandidat awal → final | Resource |
|---|---|---:|---:|---:|---|
| Multiplex-4 | [`ap-CEtP0nYg9rJdIi0xIYINGi`](https://modal.com/apps/faizakbar2301/main/ap-CEtP0nYg9rJdIi0xIYINGi) / `fc-01M3FFR14W7CC7TXAARD1TGYHB` | 01:30:56–01:31:54 | 57,84 dtk | 9 → 23 | 8 CPU, 16 GiB |
| Multiplex-8 | [`ap-JZ1n6z55VnHYPee9pHW34Y`](https://modal.com/apps/faizakbar2301/main/ap-JZ1n6z55VnHYPee9pHW34Y) / `fc-01M3FFSZ64VRTWNZB09JTBGWSX` | 01:32:02–01:34:29 | 143,52 dtk | 9 → 23 | 8 CPU, 24 GiB |
| Harmony → Leiden | [`ap-5yh1Du7t1k7KNTyQpjmvfy`](https://modal.com/apps/faizakbar2301/main/ap-5yh1Du7t1k7KNTyQpjmvfy) / `fc-01M3FFTPSMHK0XE6TXJPDDYPW3` | 01:32:27–01:35:30 | 179,86 dtk | 28 | 8 CPU, 16 GiB |
| BBKNN → Leiden | [`ap-1GbWT0DJA7A8Phl4iw92tt`](https://modal.com/apps/faizakbar2301/main/ap-1GbWT0DJA7A8Phl4iw92tt) / `fc-01M3FFVGJ59C2RDYZ206YV6V5P` | 01:32:52–01:34:29 | 95,97 dtk | 28 | 8 CPU, 24 GiB |
| AMF multi-backbone | [`ap-79nCSfwgnQhFM0zf5opnDk`](https://modal.com/apps/faizakbar2301/main/ap-79nCSfwgnQhFM0zf5opnDk) / `fc-01M3FFW6PV2X40APZX1Y5ZWFG0` | 01:33:14–01:34:38 | 79,56 dtk | 12 | 8 CPU, 24 GiB |
| TURTLE multi-backbone | [`ap-RoF2BfHkHQNKzybLRij3l2`](https://modal.com/apps/faizakbar2301/main/ap-RoF2BfHkHQNKzybLRij3l2) / `fc-01M3FFX1GRH4TTAZ0T59KXMAGK` | 01:33:42–02:08:27 | 2.083,84 dtk | 6 | 8 CPU, 24 GiB |
| FGW prototype | [`ap-gikUczseiOIwdApITzNEwx`](https://modal.com/apps/faizakbar2301/main/ap-gikUczseiOIwdApITzNEwx) / `fc-01M3FFXV2BKM4X2Y33F4TJP07C` | 01:34:08–01:34:28 | 18,46 dtk | 10 | 8 CPU, 16 GiB |

Runtime aktual dari `environment.json`:

| App | Python / dependency aktual |
|---|---|
| Core (Multiplex, AMF, Harmony, FGW) | Python 3.11.12; NumPy 1.26.4; SciPy 1.15.3; pandas 2.3.3; scikit-learn 1.7.2; igraph 1.0.0; leidenalg 0.12.0 |
| Harmony | tambahan `harmonypy` 0.0.10 |
| BBKNN | `bbknn` 1.6.0; Scanpy 1.11.4; AnnData 0.12.3; NumPy aktual 2.4.6 (deviasi dari versi core yang direncanakan) |
| FGW prototype | tambahan POT 0.9.5 |
| TURTLE multi-backbone | PyTorch 2.5.1; tqdm 4.67.1; repo `mlbio-epfl/turtle` pada commit `4674efab38e8f3cd0a4cd1f25d3ffc88be3dcabc` |

### Resume dan koreksi implementasi

Pemeriksaan artefak menemukan bahwa runner awal membentuk arm `native_only` dan
`harmonized_only`, tetapi melewatkannya pada filter full. Karena kedua arm sudah
terkunci dalam §6.1 preregistrasi, runner diperbaiki dan hanya dua App Multiplex
di-resume; kandidat lama diverifikasi checksum-nya lalu dipakai ulang. Setiap
App Multiplex menjadi 23 kandidat: 7 native-only, 7 harmonized-only, dan 9
native+bridge. FunctionCall resume penambahan arm:

| Metode | App ID / FunctionCall ID | Mulai–selesai WIB | Runtime |
|---|---|---:|---:|
| Multiplex-4 | `ap-rfyGc7nYUqyjnO4tlCUWER` / `fc-01M3FN1B9Q7R3NXA5234VTW01V` | 03:03:33–03:04:41 | 68,32 dtk |
| Multiplex-8 | `ap-1RGERh04OA66nbcm0zGvDL` / `fc-01M3FN1AZQF2JA4XXD3093PK4E` | 03:03:29–03:05:05 | 94,77 dtk |

Unduhan pertama arm tambahan mendeteksi checksum `status.json` yang stale:
status mutable lama sempat masuk ke manifest sebelum ditulis ulang. Runner
kemudian mengecualikan `status.json` dan `status_running.json` dari checksums
artefak, memverifikasi semua checksums lain, dan melakukan resume kedua tanpa
refit kandidat. Call validasi checksum:

| Metode | App ID / FunctionCall ID | Mulai–selesai WIB | Runtime |
|---|---|---:|---:|
| Multiplex-4 | `ap-GCYKUQenUJe19gYQpoaxNg` / `fc-01M3FNG6CJ20FNM4D9W51T7FGX` | 03:11:39–03:11:50 | 9,26 dtk |
| Multiplex-8 | `ap-mIBzjMQjI8jV4jUxqc2BZ1` / `fc-01M3FNG6D5W1SHEKGYKP8DRD26` | 03:11:36–03:11:54 | 16,54 dtk |

Call final tersebut memakai runner SHA-256 di atas. Resume tidak mengubah
parameter atau baseline; checksum kandidat yang tersimpan dipertahankan.
Seluruh tujuh status full terakhir adalah `success`. Jejak versi runner:
full awal `3e564c7572b65714dea0831d8cc7f257788dbd1b243094e7e9c09f8875e18c92`,
perbaikan arm `a3380689776ff64f559add0e12f7f1e48a32693bab5acbc948e8e0bc15c4daf6`,
dan perbaikan checksum akhir `a6c0aa7a047d0c49d7ff4ad795196cf5dc48e5d4338b9796a4352eb2ab2d14bb`.
Call terakhir hanya memuat cache kandidat, memverifikasi hash, dan memperbarui
manifest/status; tidak ada clustering yang di-refit.

Upaya full pertama mengalami crash-loop dan tidak dipakai sebagai hasil.
FunctionCall/App ID yang ditolak: Multiplex-4
`fc-01M3FEXWRJMDFMXHR3YVFKHEW5` (`ap-9SsRPnuO3fjxJdtXZ0HqQu`); Multiplex-8
`fc-01M3FEY40PZ8RCQWZT80GSM6DY` (`ap-RpTUWteuJHKzhPnJlgSqyZ`); Harmony
`fc-01M3FEYCQK7RE5K9275KE1418J` (`ap-yqTFc6KbPYoJQzzkgX4H7T`); BBKNN
`fc-01M3FEYNCQBE34MFH50NEBJ018` (`ap-WQ4nh0eEtpBSLvWH29B5jS`); AMF
`fc-01M3FEZ0JHY4R2MYVFCZNCC2HA` (`ap-LEYkuy4ueB9eT4dhWBQ8k1`); TURTLE
`fc-01M3FEZAWF8NDB685YE5YX9Z5F` (`ap-bD3nDGCO4YpfKU9cM0GCQv`); dan FGW
`fc-01M3FEZJD5DAB62B5NK5D0WYDS` (`ap-pND0eng3xy9Gj5tsBkGX3A`). App ID-nya
berbeda dari App sukses pada tabel; log launch dan status akhir tersimpan lokal.

Semua job yang diterima memakai CPU; tidak ada GPU yang dialokasikan. Resume
TURTLE `ap-E62fTpjpSaGDnTpyMFXfkh` / `fc-01M3FJ260A3CB18YRSC6A4HMV2` setelah
job CPU selesai menemukan output sukses yang sudah ada dan no-op, bukan run GPU.
Estimasi biaya preregistrasi adalah US$0,40–1,20; billing Modal aktual tidak
dieksport sehingga angka itu bukan tagihan terverifikasi.

## Hasil metrik

Semua metrik cluster di bawah dihitung evaluator pada 3.579 discovery rows;
silhouette memakai common space data-driven 100-D dan fixed sample 3.000
(seed 42). Source NMI lebih rendah dan source entropy lebih tinggi menunjukkan
mixing yang lebih baik, tetapi harus dibaca bersama struktur cluster. Semantic
margin adalah proxy dari concept bank, bukan ground truth.

| Kandidat seed 42 | k (min cluster) | Silhouette | Source NMI | Source entropy | Fraksi cluster mixed | ARI antarseed | Semantic margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fused K-Means beku | 12 (237) | 0,3052 | 0,1648 | 0,5396 | 0,583 | 0,9745 | 0,9853 |
| Multiplex-4 native-only | 14 (70) | 0,2018 | 0,3933 | 0,0000 | 0,000 | 0,9408 | 0,9151 |
| Multiplex-4 harmonized-only | 11 (206) | 0,2874 | 0,1406 | 0,6081 | 0,455 | 0,9473 | 0,9503 |
| **Multiplex-4 native+bridge (primer)** | **5 (597)** | **0,0908** | **0,0073** | **0,8962** | **1,000** | **0,6462** | **0,3375** |
| Multiplex-8 native-only | 14 (70) | 0,2020 | 0,3933 | 0,0000 | 0,000 | 0,9293 | 0,9153 |
| Multiplex-8 harmonized-only | 11 (194) | 0,2874 | 0,1388 | 0,6127 | 0,455 | 0,9560 | 0,9487 |
| Multiplex-8 native+bridge (primer) | 6 (287) | 0,1051 | 0,0204 | 0,8742 | 1,000 | 0,7030 | 0,4814 |
| Harmony MB4 native fused | 15 (48) | 0,2631 | 0,0855 | 0,7102 | 0,733 | 0,8546 | 0,8635 |
| BBKNN MB4 native fused | 15 (47) | 0,2872 | 0,1637 | 0,5255 | 0,400 | 0,9207 | 1,0307 |
| AMF MB4 harmonized | 12 (47) | 0,2784 | 0,1387 | 0,6108 | 0,417 | 1,0000 | 0,9082 |
| TURTLE MB4 harmonized | 12 (219) | 0,2831 | 0,1562 | 0,5587 | 0,500 | 0,8017 | 0,9824 |
| FGW prototype MB4, α=0,5 | 12 (149) | 0,2383 | 0,0087 | 0,8885 | 1,000 | 0,8496 | 0,7855 |

Nilai source NMI Multiplex native-only yang tinggi dan source entropy nol
menunjukkan pemisahan web–field, bukan pencampuran. Sebaliknya, nilai NMI
native+bridge yang mendekati nol terjadi bersama k=5/6, silhouette rendah, dan
cross-source kNN rate 0,0361/0,0347 (di bawah baseline 0,1115). Jadi angka NMI
tersebut tidak boleh ditafsirkan sendiri sebagai keberhasilan integrasi.

## Keputusan hipotesis preregistrasi

| ID | Hasil | Interpretasi terbatas pada bukti |
|---|---|---|
| H1 | **Tidak lolos** | Primer Multiplex-4 gagal gate jumlah cluster, silhouette, dan stabilitas. ARI null percentile 1,00, tetapi mean ARI 0,6462 tetap di bawah 0,80; mixing proxy yang tinggi tidak mengatasi collapse. |
| H2 | **Tidak didukung oleh panel** | Pada MB4/MB8, native+bridge memberi silhouette dan stabilitas lebih rendah daripada kedua arm kontrol. Native-only memisahkan source; harmonized-only lebih seimbang dan jauh lebih compact daripada bridge. |
| H3 | **Keuntungan relatif parsial, bukan promosi** | MB8 bridge naik dibanding MB4 bridge pada silhouette (0,1051 vs 0,0908), ARI (0,7030 vs 0,6462), dan semantic margin (0,4814 vs 0,3375), tetapi keduanya tetap di bawah gate primer dan hanya 6/5 cluster. |
| H4 | **Trade-off; tidak ada pemenang tunggal** | Harmony menurunkan source NMI tetapi silhouette turun; BBKNN mendekati baseline pada silhouette, namun mixing/entropy tidak membaik terhadap K-Means secara konsisten. BBKNN juga memiliki deviasi versi yang dicatat di bawah. |
| H5 | **Tidak ada promosi multi-backbone** | AMF stabil tetapi silhouette/margin lebih rendah dari K-Means; TURTLE MB4 paling dekat pada silhouette/margin namun tidak melewati silhouette floor, dan MB8 lebih rendah. TURTLE RADIO+SigLIP2 existing tidak diulang. |
| H6 | **Multiresolution saja; bukan hierarki** | Containment antar-resolusi MB4 bridge: 0,9927; 0,9436; 0,8544; 0,8785. Hanya 2/4 transisi mencapai 0,90, kurang dari 3/4 yang disyaratkan. |

H2–H5 ditandai evaluator sebagai `see_metric_panel`; tidak ada skor komposit
pasca-hasil. `pareto.csv` memuat 62 titik Pareto dari keseluruhan 161 baris
(130 kandidat baru + 31 beku). Tidak satu pun angka tersebut dipakai untuk
memilih ulang konfigurasi setelah hasil terlihat. Audit galeri manual tidak
dijalankan; karena H1 sudah gagal beberapa gate numerik, klaim promosi juga
tidak dibuat.

## Batas, deviasi, dan interpretasi

- Runner full awal melewatkan dua arm Multiplex yang sudah dipreregistrasi.
  Kekurangan ini diperbaiki dengan resume berbasis candidate cache dan
  checksum; hanya 28 kandidat kontrol baru yang dihitung (14 per App). Tidak
  ada baseline lama, TURTLE RADIO+SigLIP2, atau CoMVC yang dihitung ulang.
- Environment BBKNN melaporkan NumPy `2.4.6`, sedangkan image core yang
  direncanakan memakai `1.26.4`; BBKNN, Scanpy `1.11.4`, dan AnnData `0.12.3`
  tetap menghasilkan artefak yang lolos alignment/checksum. Karena package
  resolution menyimpang dari pin core, hasil BBKNN diperlakukan sebagai
  comparator eksploratif, bukan bukti konfirmatori.
- Tujuh App memakai 8 CPU dan 16/24 GiB sesuai spesifikasi masing-masing; GPU
  tidak dipakai. Algoritme graph/Leiden dalam runner ini CPU-only. Semua job
  memakai cached embedding; tidak ada inference atau fine-tuning.
- Source label adalah proxy acquisition dari ukuran citra (`max(W,H)>150`),
  bukan semantic class. Text/concept hanya dipakai evaluator setelah clustering.
  Evaluasi semantik adalah validasi proxy dan tidak menggantikan anotasi buta.
- Estimator utama semifinal tetap fused K-Means beku; hasil ini tidak
  membuktikan bahwa harmonisasi JPEG menyelesaikan seluruh confound akuisisi.

## Artefak hasil

Semua output berada di
[`semifinal/results/acquisition_multiview/20260926T180402Z/`](../results/acquisition_multiview/20260926T180402Z/):
`comparison.csv`, `stability.csv`, `source_mixing.csv`, `hierarchy.csv`,
`pairwise.csv`, `pairwise_seed_stability.csv`, `pareto.csv`,
`category_audit.csv`, `semantic_profiles.csv`, `failures.csv`,
`hypothesis_decisions.csv`, dan `evaluation_metadata.json`.

Log evaluator detached tersimpan pada
`semifinal/logs/evaluation_20260926T180402Z.stdout.log` dan
`semifinal/logs/evaluation_20260926T180402Z.stderr.log`. Launch/call IDs tercatat
di `semifinal/logs/framework_parallel_20260926T180402Z.jsonl`; monitor log Modal
dua-menit tersedia di `semifinal/scripts/watch_acquisition_multiview.ps1`.

## Addendum: MGE dengan operationalization CHC single-link

Tanggal: 27 September 2026. Status MGE diperbarui dari `SKIPPED_UNDER_SPECIFIED`
menjadi **dijalankan sebagai reimplementasi**, bukan official MGE run. Paper
GAPS AAAI 2026 dari lineage penulis CHC yang sama secara eksplisit menetapkan
jarak antarklaster dengan single-linkage dan jarak fused sebagai rata-rata
matriks jarak per-view ([paper GAPS](https://ojs.aaai.org/index.php/AAAI/article/download/39973/43934)).
Ini menyediakan dasar operasional yang defensible, tetapi tidak membuktikan
private implementation MGE 2025 memakai pilihan identik ([paper MGE
IJCAI 2025](https://www.ijcai.org/proceedings/2025/0756.pdf)).

Tiga view dinormalisasi L2 per baris; jarak sampel Euclidean; fused distance
adalah mean jarak `dinov3`, `siglip2`, dan `aimv2`. CHC memakai jarak cluster
single-link, adjacency 1-NN dengan kendala ukuran, dan biaya
`d² × |Ci| × |Cj|`; final secondary CHC menerima `D_B = 1 − B` dan menghasilkan
tepat 12 cluster. Fit hanya pada 3.579 discovery rows; 600 holdout dikecualikan.
Parameter `λ=0,5` dan `t=20` dikunci.

MGE menetapkan rumus jumlah granularity `ki = round(λ·mi)` tetapi tidak
menjabarkan jadwal cut yang dipilih. Untuk run ini, sebelum melihat hasil,
ditetapkan `mi=K=12` kandidat cut dengan jumlah cluster 12…1; `λ=0,5` memilih
enam cut yang merentang merata dari halus ke kasar: 12, 10, 8, 5, 3, 1. Ini
adalah operationalization tambahan yang dicatat sebagai batas bukti, bukan
diklaim sebagai jadwal resmi penulis.

| Kandidat | Cluster (min–maks) | Silhouette | Stability ARI | Source NMI | Function AMI | Semantic margin | Runtime | DMS AMI / partial R² / cosine gap | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MCSF (3-view) | 12 (219–457) | 0,2993 | 0,9946 | 0,1678 | 0,7884 | 1,0263 | 19,6 dtk/run | 0,1053 / 0,1314 / 0,0475 | **FAIL** |
| MGE single-link CHC reimplementation | 12 (122–669) | 0,2551 | — | 0,1653 | 0,7500 | 0,9461 | 21,0 dtk | 0,1006 / 0,1171 / 0,0350 | **FAIL** |
| MGE average-link CHC sensitivity | 12 (174–502) | 0,2654 | — | 0,1685 | 0,7409 | 0,7486 | 20,7 dtk | 0,1047 / 0,1295 / 0,0409 | Sensitivity only |
| K-Means fused 3-view (kontrol adil) | 12 (213–463) | 0,3004 | 0,9980 | 0,1593 | 0,7743 | 1,0263 | 0,51 dtk/seed | 0,1110 / 0,1335 / 0,0482 | Pembanding |
| K-Means fused 8-view (cache) | 12 (237–457) | 0,3052 | 0,9745 | 0,1648 | 0,7835 | 0,9853 | Cached; tidak diulang | 0,1140 / 0,1369 / 0,0501 | Pembanding |

MGE single-link lolos pemeriksaan validitas struktural: 12 cluster, ukuran
minimum 122, fraksi cluster terbesar 18,7%, dan Source NMI 0,1653 di bawah
batas 0,184801. Namun promotion gate lama tidak diubah dan MGE **tidak lolos**:
silhouette tidak naik `+0,01` atas kontrol 3-view, serta Function AMI dan
semantic margin juga lebih rendah. MCSF juga tidak lolos karena silhouette-nya
tidak mencapai ambang promosi, meskipun stability ARI antarseed 0,9946. DMS46
bersifat post-hoc dan tidak memengaruhi gate. Tidak ada stability ARI antarseed
untuk MGE karena hanya satu run deterministik; ARI antara partisi single-link
dan average-link adalah **0,6953**, sehingga linkage memberi sensitivitas yang
berarti. Partisi average-link hanya diagnostik dan tidak eligible
menggantikan primary. Fused K-Means tetap estimator utama.

Run MGE pertama bersifat provisional dan tidak dipakai sebagai hasil setelah
review implementasi menemukan konstruksi tree CHC perlu menyelesaikan adjacency
graph per connected component. Kedua linkage dijalankan ulang dari runner yang
diperbaiki; tabel di atas memakai output final itu. Biaya gross tetap
memperhitungkan dua call provisional yang telah berjalan. Nilai DMS MGE juga
dihitung ulang untuk partisi final, bukan diambil dari cache run provisional.

Modal App final [`ap-AsxzDJ5BkE74mL1fRNKo9g`](https://modal.com/apps/faizakbar2301/main/ap-AsxzDJ5BkE74mL1fRNKo9g)
memakai profile `faizakbar2301`, Volume `bdc-data`, CPU 16 / RAM 32 GiB; kedua
FunctionCall berjalan overlap dan selesai sukses:
single-link `fc-01M3GGD0D02SSN7Q86WRACFW8Y`, average sensitivity
`fc-01M3GGD15BJD54NWA8RW6XAAW5`. Runtime worker masing-masing 21,0 dan 20,7
detik. Artefak worker tersimpan di
`semifinal/results/mge_mcsf_fast/remote/`; evaluator bersama memperbarui
`comparison.csv`, `assignments.csv`, dan `metrics.json` di folder yang sama.
Estimasi biaya gross total eksperimen MCSF+MGE: US$0,152; upper bound dengan
buffer dan waktu startup: US$0,316 (bukan invoice billing). Estimasi wall time
end-to-end: sekitar 51 menit saat artefak final ditulis, di bawah batas 180 menit
dan US$6,50.
