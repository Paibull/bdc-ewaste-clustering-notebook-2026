# Eksperimen Partisi Multi-Lensa dan Validasi

Status: hasil komputasi final per 7 September 2026. Validasi manusia sudah
disiapkan, tetapi belum diisi; bagian itu tidak boleh diklaim selesai.

## Ringkasan hasil

Empat partisi final ialah A jenis/fungsi `k=11`, B material/nilai pemulihan
`k=6`, C kondisi barang `k=5`, dan D data-driven `k=12`. Rata-rata AMI enam
pasang lensa adalah 0,443, sedangkan empat K-Means dengan jumlah klaster sama
di ruang penuh mempunyai AMI 0,814. Proyeksi konsep karena itu menghasilkan
partisi yang lebih komplementer daripada sekadar mengganti `k` atau seed.

Semua partisi final mengalahkan 200 null marginal-shuffle dan 200 null Gaussian
cocok-kovarians pada silhouette, Calinski-Harabasz, dan Davies-Bouldin
(`p=(0+1)/(200+1)=0,004975` untuk tiap perbandingan). Hasil juga stabil tanpa
200 citra test: ARI 0,994/0,977/0,973/0,990 untuk A/B/C/D.

## Input dan ruang representasi

- Populasi utama berisi 4.179 citra: 3.979 train berlabel e-waste dan 200 test
  yang diprediksi e-waste. Kolom `split` mempertahankan asal setiap citra.
- Input aktif ialah `ewaste_emb_harm_resjpeg.npz`: semua citra diturunkan ke
  150 px lalu di-encode ulang JPEG quality 75, subsampling 4:2:0 sebelum
  ekstraksi delapan backbone. Dasar pemilihan ada di `02-HARMONISASI.md`.
- A–C memakai fitur non-GAP SigLIP2 dan koordinat pada rentang linear jangkar
  teks yang sudah dipusatkan. Rank numeriknya 11, 7, dan 10.
- D menyeimbangkan delapan backbone: L2 per backbone, PCA-32 per blok, L2,
  konkatenasi, L2, PCA-100, lalu L2. PCA akhir ditambahkan setelah audit
  menemukan versi awal masih 256 dimensi dan melanggar batas 50–100 dimensi
  yang telah disepakati. Semua hasil D kemudian dihitung ulang.
- Bank konsep memuat 31 konsep dan 368 prompt unik. Seluruh prompt di-encode
  memakai `hf-hub:timm/ViT-SO400M-16-SigLIP2-384`, konteks 64 token, bobot yang
  sama dengan encoder citra.

## Sweep metode dan panel indeks

Sebanyak 272 kandidat dijalankan: K-Means, Ward, dan GMM diagonal untuk
`k=4..24` pada empat lensa, ditambah HDBSCAN dengan `min_cluster_size` 20, 40,
80, 120, dan 200. K-Means/Ward/GMM dinilai dengan silhouette,
Calinski-Harabasz (CH), dan Davies-Bouldin (DB). HDBSCAN dinilai dengan DBCV,
bukan silhouette.

| Lensa | K-Means final | Silhouette | CH | DB | NMI korpus | Klaster minimum |
|---|---:|---:|---:|---:|---:|---:|
| A — jenis/fungsi | 11 | 0,499 | 1.654,36 | 0,864 | 0,109 | 273 |
| B — material | 6 | 0,243 | 895,16 | 1,470 | 0,032 | 459 |
| C — kondisi | 5 | 0,229 | 923,68 | 1,542 | 0,170 | 497 |
| D — data-driven | 12 | 0,303 | 494,32 | 1,463 | 0,166 | 277 |

Panel tidak selalu sepakat. Pada B, silhouette dan DB mendukung `k=6`, sedangkan
CH condong ke solusi lebih kasar. Pada D, silhouette maksimum berada di `k=16`
dan DB terbaik di `k=13`; pilihan final tidak ditentukan oleh salah satu argmax.
Ward dan GMM umumnya lebih lemah. HDBSCAN membuang 835 citra pada A, 2.366 pada
B, 2.112 pada C, dan 938 pada D pada konfigurasi DBCV terbaik. Karena tujuan
mencakup seluruh korpus dan struktur density C bahkan memberi DBCV negatif,
K-Means dipakai sebagai estimator utama.

## Pemilihan jumlah klaster

Kandidat dekat daerah masuk akal diuji dengan protokol chooseR: 100 subsample
80%, matriks ko-clustering dengan denominator pasangan yang benar-benar
tersampel, silhouette konsensus per klaster, 25.000 bootstrap untuk CI 95%, dan
Jaccard klaster yang dipasangkan memakai Hungarian matching.

Stabilitas diperlakukan sebagai syarat perlu. Kandidat harus mempunyai rata-rata
Jaccard minimum per klaster sedikitnya 0,80; setelah itu kandidat diurutkan dari
jumlah klaster di atas ambang CI, jumlah di bawah ambang, median konsensus, lalu
`k` yang lebih kecil. Aturan ini mencegah D `k=13` terpilih walau mempunyai 10
klaster di atas ambang, karena satu klasternya hanya Jaccard 0,610. D `k=12`
memiliki minimum 0,829.

| Lensa | k | Median konsensus | CI 95% | Mean Jaccard | Minimum Jaccard |
|---|---:|---:|---:|---:|---:|
| A | 11 | 0,9985 | [0,9979; 0,9993] | 0,9947 | 0,9763 |
| B | 6 | 0,9828 | [0,9678; 0,9929] | 0,9614 | 0,9293 |
| C | 5 | 0,9903 | [0,9876; 0,9978] | 0,9825 | 0,9737 |
| D | 12 | 0,9917 | [0,9777; 0,9981] | 0,9540 | 0,8287 |

## Komplementaritas

| Pasangan | AMI | VI |
|---|---:|---:|
| A–B | 0,401 | 2,450 |
| A–C | 0,361 | 2,510 |
| A–D | 0,781 | 1,050 |
| B–C | 0,268 | 2,438 |
| B–D | 0,400 | 2,526 |
| C–D | 0,446 | 2,239 |

A dan D masih dekat karena struktur alami ruang penuh didominasi bentuk/jenis
perangkat, sama seperti kontrol Clevr-4. B–C paling berbeda. Empat partisi naif
di ruang penuh dengan `k` 11/6/5/12 memberi AMI rata-rata 0,814 dan VI 0,754;
lensa terarah memberi AMI 0,443 dan VI 2,202.

## Kalibrasi null

Untuk setiap partisi final dibuat 200 null marginal-shuffle dan 200 null
Gaussian dengan skala principal-component cocok. Ketiga indeks teramati lebih
baik dari semua draw null. Contoh paling konservatif ialah B: silhouette 0,243
versus mean Gaussian 0,112; CH 895,16 versus 501,97; DB 1,470 versus 1,848.
Pada D: silhouette 0,303 versus 0,039; CH 494,32 versus 119,24; DB 1,463 versus
3,048. Angka lengkap ada di `null_calibration_summary.csv` dan seluruh 1.600
draw disimpan di `null_calibration_draws.csv`.

## Sensitivitas

- Tanpa 200 test, ARI A/B/C/D = 0,994/0,977/0,973/0,990.
- Sembilan seed K-Means memberi median ARI 0,999/0,991/0,982/0,997.
- Leave-one-template memberi rentang A 0,977–0,984, B 0,865–0,900, C
  0,832–0,896. Leave-one-concept lebih keras: minimum A/B/C
  0,870/0,359/0,526. Ini menunjukkan konsep tertentu memang menentukan sumbu,
  bukan sekadar variasi template.
- Varian koordinat kemiripan `sim` hanya memberi ARI 0,320/0,107/0,171 terhadap
  `span` A/B/C. Hasil Clevr-4 sudah menunjukkan `span` lebih benar secara
  ground-truth, sehingga perbedaan ini bukan alasan mengganti metode utama.
- D satu-backbone memberi ARI 0,723–0,898. Leave-one-backbone memberi
  0,923–0,991. Kontribusi multi-backbone karena itu tidak bergantung pada satu
  encoder, walau DINOv2 tunggal paling berbeda.
- Mengganti input final dengan embedding mentah atau resolusi-saja memberi ARI
  0,937–0,944 (A), 0,822–0,849 (B), 0,879–0,905 (C), dan 0,949 untuk D.
  Struktur utama bertahan, tetapi harmonisasi tetap material terutama untuk B.

## Baseline multiple clustering

Semua baseline memakai ruang D PCA-100 dan jumlah klaster `[11,6,5,12]`.
ClustPy 0.0.2 dijalankan di Modal; ENRC melatih autoencoder fitur, tetapi tidak
melatih ulang backbone citra.

| Metode | Detik | AMI antarpartisi | AMI optimal ke A–D | Silhouette subruang | NMI korpus |
|---|---:|---:|---:|---:|---:|
| NrKmeans | 6,19 | 0,267 | 0,421 | 0,302 | 0,109 |
| OrthogonalClustering | 0,38 | 0,105 | 0,335 | 0,106 | 0,094 |
| ENRC | 53,42 | 0,328 | 0,436 | 0,547 | 0,081 |
| Concept span | — | 0,443 | — | 0,319 | 0,119 |

ENRC mempunyai subruang yang paling kompak, dan metode klasik menghasilkan AMI
antarpartisi lebih rendah. Namun setelah empat partisi tiap metode dipasangkan
optimal ke A–D, kesesuaiannya hanya 0,335–0,436. Temuan ini mendukung perbedaan
klaim: multiple clustering dapat menghasilkan banyak partisi, sedangkan concept
span mengendalikan kriteria partisi dan memberi nama yang dapat diaudit.

TURTLE commit `4674efab38e8f3cd0a4cd1f25d3ffc88be3dcabc` dijalankan pada RADIO dan
SigLIP2 PCA-100, `k=12`, `gamma=10`, 10 inner steps, 6.000 outer iterations,
cold-start, tiga seed. Semua run menemukan 12 klaster dalam 65,6–68,5 detik.
AMI terhadap D ialah 0,860/0,708/0,783; mean AMI antarseed 0,731. Silhouette di
ruang D ialah 0,268/0,168/0,207, di bawah K-Means D 0,303. Generalization score
10-fold berada pada 0,921–0,968. TURTLE valid sebagai pembanding
multi-representasi, tetapi tidak menggantikan partisi D yang lebih stabil.

Multi-MaP commit `7fb610582c64eaf9f00af56af14398f4feaea1c1` dan Multi-Sub commit
`7f69a2e20d95871906cf87df1d41ccd6b6c28f92` diaudit dari repo yang dicatat di
katalog. Implementasi rilis terikat pada OpenAI CLIP ViT-B/32, `ImageFolder`,
perubahan proyeksi visual, dan proxy per citra; tidak ada antarmuka fitur
precomputed. Reimplementasi agar menerima SigLIP2 akan mengubah metode dan tidak
menjadi pembanding stok yang adil. Keduanya tidak dijalankan dan tidak boleh
ditulis sebagai hasil eksperimen.

## Interpretasi klaster

`cluster_concept_profiles.csv` menyimpan tiga konsep teratas dari ketiga bank
untuk setiap klaster. Skor ialah rerata z-score arah kontras SigLIP2, bukan label
ground-truth. `cluster_cards.csv` memberi nama utama, nama sekunder, kandidat
UNU-KEY/WEEE, tier bahaya A/B/C, dan rute penanganan. Pemetaan mengacu pada
UNU-KEY, WEEE Annex III, B107d PP 22/2021, dan tier A1181 yang dirangkum di
`01-KATALOG-METODE.md` §8.

Lensa A memisahkan printer, ponsel/tablet, layar, perangkat input, mesin cuci,
PCB/komponen, audio, laptop, baterai, peralatan memasak, dan desktop. B
memisahkan campuran, kaca layar, PCB/konektor, plastik/kabel, baterai, dan fraksi
logam. C memisahkan layar pecah, tumpukan campuran, perangkat terpasang,
komponen terurai, dan barang utuh/layak guna ulang. Montase setiap klaster berisi
empat medoid, empat inti, dan empat titik batas.

Tier dan rute adalah kandidat operasional berdasarkan benda/material yang
terlihat. Citra RGB tidak membuktikan kandungan Pb, Hg, Cd, BFR, jenis baterai,
atau konsentrasi logam. Klaim kimia per citra memerlukan metadata atau pengujian
material; paper harus memakai kata “potensi” dan “prioritas triase”.

## Validasi manusia yang masih terbuka

`intruder_study/bundles/` berisi tiga bundle offline terpisah dan kunci yang
tidak boleh diberikan kepada penilai. ZIP `penilai_N.zip` hanya memuat halaman
HTML, image lokal yang direferensikan, dan instruksi. Setiap penilai menerima
110 tugas: lima tugas per klaster pada A–C. Tiap tugas menampilkan tiga anggota
inti dan satu intruder sulit dari klaster lain, diacak per penilai, lalu
menanyakan kecocokan nama. Peluang tebak intruder adalah 25%. Semua item
berasal dari subset 600 citra yang ditarik sebelum partisi final.

Tiga berkas jawaban harus dinamai `jawaban_penilai_1.csv` sampai
`jawaban_penilai_3.csv`. `score_intruder_study.py` menghitung akurasi, Wilson CI
95%, persetujuan nama, dan Krippendorff alpha. Sampai berkas itu benar-benar
diisi manusia, tidak ada angka validasi manusia yang boleh masuk abstrak,
kesimpulan, atau tabel hasil.

## Jejak komputasi Modal

- Jangkar teks: `ap-vZFt5WnM8j9qdr7rvFdqvC`.
- ClustPy pada ruang awal dan rerun final PCA-100: `ap-VdTKSbXeKhhgxizKj8oBBU`
  dan `ap-j3smAUHa7tlmiJqP5RCRYH`.
- TURTLE pilot 2.000 iterasi: `ap-3eMGH4LXNXvi7vXB3HXEHq`,
  `ap-aG4WdaWiPfN137B4Ri2xiE`, `ap-DTQVOdry1iucGzgRHOMiBp`. Pilot diganti karena
  repo sumber memakai 6.000 iterasi dan PCA awal 128 melampaui batas proyek.
- TURTLE final: `ap-aPZGqEbShRGe1sDoJhZiBm`,
  `ap-PdCcDIY3x24GT30s71YmPQ`, `ap-L36jW7adfKekKBQLBSVTEw`.
- ENRC gagal impor karena `torchvision` tidak dideklarasikan ClustPy:
  `ap-iraCHcbki3aNkHSqP23QMR`. Setelah dependensi eksplisit ditambah, run awal
  `ap-qXSISk5ROjPPB2xVRK6Rdw` dan rerun PCA-100
  `ap-FBocZkqvIVY3k5Za8LIIJo` selesai.

## Reproduksi

Jalankan dari akar repo:

```powershell
python semifinal/scripts/lensa_ewaste.py
python semifinal/scripts/validasi_lensa.py
python semifinal/scripts/finalize_lenses.py
python semifinal/scripts/null_lensa.py
python semifinal/scripts/sensitivity_lenses.py
python semifinal/scripts/interpret_lenses.py
python semifinal/scripts/prepare_intruder_study.py
python semifinal/scripts/evaluate_multiple_baselines.py
python -m pytest -q semifinal/scripts
```

Artefak besar tidak masuk Git dan tersimpan lokal serta di volume Modal. Hash
final utama:

| Artefak | SHA-256 |
|---|---|
| `concept_bank.yaml` | `b68426c74c2e5efa981b483302b4515be7021a5cb9eeaebbabd002732eea03d3` |
| `concept_axes.npz` | `ffe7cfa2bc7f12400bfe3c70e01e70ba9fc47dd352fc133b2fe04f91895d9407` |
| `lens_spaces.npz` | `6242a4266f10d22e878568514785dddadb25fa85646c705c809a0cd258eaaa2a` |
| `lens_partitions.npz` | `9b2bae51e6ca583d819440fc7e7ff3b209d71fe0b04fb8358203f85dc22a1416` |
| `clusters_lenses_final.csv` | `f09e72daace2d3a8f4635d54e388cb0adf4553e57e5f31e7a985cc741023b60a` |
| `multiple_clustering_baselines.npz` | `1217784edd704d225b64f9c7580e66dedc70861632bb16de2530e9ea01f85729` |
| `enrc_baseline.npz` | `3eec5b05bccd73fa7f960d58931892426b3ca9f1cc8500d70b591893a307fada` |

CSV merupakan sumber angka paper. NPZ menyimpan koordinat/partisi yang diperlukan
untuk regenerasi, sedangkan script adalah sumber definisi parameter.
