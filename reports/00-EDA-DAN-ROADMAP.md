# Semifinal BDC 2026 — EDA subset e-waste & roadmap

Tanggal: 3 Sep 2026. Deadline: 28 Sep 2026 16.00 WIB.
Halaman versi web (dengan montase eksemplar): artifact "Dua Korpus E-Waste".

## 0. Subset & artefak

- N = **4.179** citra = 3.979 train berlabel E (`diagnostics/y_aggressive.npy == 1`)
  + 200 test berprediksi E (`diagnostics/final_test_pred.npy == 1`).
- Embedding diambil ulang dari `diagnostics/raw_emb.npz` (8 backbone, sudah ada dari
  penyisihan; tidak ada training baru).
- Berkas di folder ini:
  | berkas | isi |
  |---|---|
  | `ewaste_emb.npz` | 8 backbone × 4.179 citra (float16) + `key`, `split`, indeks asal |
  | `clusters_k16.csv` | partisi k-means k=16 per backbone (PCA-64) |
  | `imgstats.csv` | statistik per citra: W, H, border_white, white_frac, saturasi, edge, colorfulness |
  | `montage/` | montase 8 eksemplar per klaster (SigLIP2, k=16) |

Script yang dipakai ada di container sesi (`extract_e.py`, `combine_e.py`, `montage.py`,
`imgstats.py`) — perlu di-commit ke repo kalau mau reproducible.

## 1. Temuan utama: subset E terdiri dari DUA korpus

- **2.815 citra (67,4%) berukuran persis 150×150 px** → korpus web hasil scraping.
- **1.364 citra (32,6%) resolusi tinggi**, ukuran khas sensor ponsel:
  1152×864 (109), 1280×720 (66), 4608×3456 (52), 4160×2340 (49), dst.

Ukuran seberapa parah kedua korpus terpisah di ruang embedding (SigLIP2):

| ukuran | nilai | pembanding |
|---|---|---|
| AUC probe linier menebak korpus | **0,993** | 0,5 = tak terpisahkan (dinov3 0,927; radio 0,979) |
| tetangga lintas-korpus @kNN=15 | **1,6%** | harapan acak **44,0%** |
| NMI(klaster k=16, korpus) | **0,329** | proksi "studio vs lapangan" cuma 0,097 |

Per klaster, proporsi anggota dari korpus resolusi tinggi nyaris biner — **tidak ada satu pun
klaster yang mencampur kedua korpus**:

- ~100% hi-res: klaster 02 (98,7%), 04 (100%), 09 (100%), 11 (100%), 15 (99,5%)
- ~0% hi-res: klaster 00, 01, 03, 05 (6,9%), 06, 07, 08, 10, 12, 13, 14

Bukti kualitatif: klaster 08 (meja kerja, 150px) vs 09 (meja kerja, HD) isinya sama —
monitor+keyboard+mouse in-situ — tapi jadi dua klaster. Sama untuk 13 (tablet/layar retak,
150px) vs 15 (ponsel, HD).

## 2. Pembacaan klaster (SigLIP2, k=16) — USANG

> **DIGANTIKAN 5 Sep 2026 oleh `02-HARMONISASI.md` §8.** Tabel di bawah dihitung
> di embedding sebelum harmonisasi, jadi beberapa klasternya terbelah oleh korpus
> dan bukan oleh isi (08/09 dan 13/15 masing-masing satu benda yang terpotong
> dua). Disimpan sebagai arsip, jangan dikutip sebagai pembacaan data.

| k | isi | n | %hi-res | %studio |
|---|---|---|---|---|
| 00 | audio & radio portabel | 273 | 1,8 | 53,8 |
| 01 | mouse (foto katalog) | 233 | 1,7 | 34,3 |
| 02 | layar & kamera campuran | 230 | 98,7 | 9,6 |
| 03 | mesin cuci | 299 | 1,7 | 46,2 |
| 04 | laptop (foto lapangan) | 170 | 100 | 0,6 |
| 05 | PCB & komponen | 291 | 6,9 | 10,3 |
| 06 | baterai | 254 | 1,6 | 76,8 |
| 07 | printer & fotokopi | 294 | 2,4 | 32,3 |
| 08 | set meja kerja (150px) | 110 | 0,9 | 0 |
| 09 | set meja kerja (HD) | 270 | 100 | 0 |
| 10 | microwave | 270 | 2,6 | 24,8 |
| 11 | tumpukan & scrap e-waste | 425 | 100 | 4,0 |
| 12 | keyboard (foto katalog) | 248 | 0,8 | 36,7 |
| 13 | tablet & layar retak | 288 | 0,7 | 0 |
| 14 | TV & monitor | 309 | 0,3 | 12,0 |
| 15 | ponsel (foto lapangan) | 215 | 99,5 | 0 |

Catatan penting: **sebagian besar citra bukan foto limbah**, melainkan foto katalog
perangkat utuh (mesin cuci, microwave, baterai dalam kemasan). Konsisten dengan temuan
penyisihan bahwa label sumber mengikuti topik artikel, bukan isi visual.

## 3. Karakter data lain

- Dimensi intrinsik (TwoNN): 8,8–11,0 untuk semua backbone → manifold berdimensi rendah.
- Duplikat dekat: cos>0,99 → 122 citra (2,9%) dalam grup ≥2; cos>0,95 → 659 (15,8%).
- Stabilitas partisi k=16: ARI antar-seed 0,85–0,94; ARI antar-subsampel 80% 0,83–0,93.
- Puncak silhouette di k ≈ 12–16 (radio 0,296 @k=14; eva02 0,337 @k=14; dinov2 0,333 @k=36).
- NMI antar-backbone pada k=16: 0,705–0,844 (tertinggi siglip↔siglip2 0,844,
  convnext↔eva02 0,833, dinov3↔radio 0,817) — backbone sekeluarga menghasilkan partisi mirip.

## 4. Uji koreksi confound (ruang PCA-128, SigLIP2)

| varian | AUC probe korpus | NMI(klaster,korpus) | tetangga lintas-korpus |
|---|---|---|---|
| tanpa koreksi | 0,993 | 0,329 | 1,6% |
| INLP r=3 | 0,881 | 0,200 | 5,4% |
| INLP r=10 | 0,854 | 0,198 | 5,8% |
| penyeragaman per korpus (CORAL) | 0,903 | 0,101 | 8,8% |
| penyeragaman + INLP r=3 | 0,899 | **0,066** | **9,5%** |
| (ideal) | 0,500 | 0,000 | 44,0% |

radio: 0,979 / 0,178 / 6,7% → 0,869 / 0,089 / 12,0%.

Validasi pasangan kembar setelah koreksi:
- klaster 08 & 09 → jatuh ke 3 klaster tujuan yang sama, overlap **109** dari n=110 → MENYATU
- klaster 13 & 15 → overlap **163** → MENYATU
- kontrol negatif: klaster 01 (mouse) & 12 (keyboard) → overlap **8** → TETAP TERPISAH

Kesimpulan: koreksi ruang fitur menghentikan k-means memakai korpus sebagai sumbu pemisah,
tapi tidak menggabungkan kedua manifold. Perbaikan yang benar ada **di input**:
turunkan semua citra ke resolusi efektif sama (150 px → upsample ke input backbone) sebelum
embedding dihitung.

> **RALAT 5 Sep 2026 — lihat `02-HARMONISASI.md`.** Arah "di input" benar, tapi
> resolusinya salah tebak. Menyamakan resolusi saja cuma memindahkan AUC 0,993 → 0,983;
> dengan isi gambar dikunci (pasangan klaster 08&09) AUC tetap 0,999. Pemisah sebenarnya
> adalah riwayat encoder JPEG: 2.814 dari 2.815 citra scraping punya tabel kuantisasi
> identik (standar IJG q75, 4:2:0, std 0,000). Setelah re-encode q75 ke semua citra,
> AUC 08&09 jatuh ke 0,523 dari tebak-acak 0,500.

## 5. Rancangan solusi semifinal — empat lensa

Satu himpunan citra, satu ruang representasi terharmonisasi, empat proyeksi:

- **Lensa A — jenis & fungsi perangkat.** Baseline; struktur alami data sudah ke sana.
- **Lensa B — nilai daur ulang & material.** PCB/konektor emas, kabel tembaga, casing ABS,
  rangka logam, kaca layar. Sinyal terbukti ada (klaster 05, 11).
- **Lensa C — bahaya & prioritas penanganan.** Baterai litium (klaster 06 sudah utuh),
  CRT bertimbal (dalam klaster 14), lampu merkuri, kapasitor.
- **Lensa D — murni data-driven.** Kontrol: kalau A–C cuma mengulang D, proyeksi konsep
  tidak menambah apa-apa.

Mekanisme: sub-ruang konsep dari ruang gambar–teks SigLIP2. Bank prompt per lensa di-encode
menara teks `hf-hub:timm/ViT-SO400M-16-SigLIP2-384` (open_clip) — bobot yang SAMA dengan
embedding gambar yang tersimpan. Clustering dijalankan di dalam sub-ruang, bukan ruang penuh.
Kriteria keberhasilan: **NMI antar-lensa rendah** (bukti komplementaritas).

**Hambatan terkonfirmasi:** Hugging Face diblokir dari sesi ini (proxy container 403; jaringan
VM lokal gagal total). Encoding prompt harus dijalankan di **Kaggle atau Modal**.

## 6. Roadmap

| tanggal | fase | isi | keluaran |
|---|---|---|---|
| 3–6 Sep | 0 | ~~harmonisasi resolusi~~, re-embed di Modal — **SELESAI 5 Sep** | `ewaste_emb_harm_resjpeg.npz` + `tabel1_harmonisasi.csv` (Tabel 1 paper) |
| 6–11 Sep | 1 | bank prompt 40–60/lensa (rujuk Permen LHK B3 + kategori WEEE UE), encode teks, bangun sub-ruang | `concept_axes.npz` + bukti retrieval per sumbu |
| 11–16 Sep | 2 | clustering per lensa, pilih k (silhouette + gap + stabilitas bootstrap), matriks NMI antar-lensa | partisi final 4 lensa |
| 16–20 Sep | 3 | anotasi manual 30 citra/klaster dengan rubrik e-waste, hitung purity | kartu klaster (nama, n, eksemplar, purity, implikasi penanganan) |
| 18–24 Sep | 4 | dokumen metodologi penyisihan + video verifikasi | dokumen + link video teruji |
| 20–26 Sep | 5 | penulisan di template panitia (langsung di template, jangan konversi akhir) | draf lengkap H-4 |
| 26–28 Sep | — | review, cek similarity, uji akses link dari akun lain, kirim H-1 | submit |

### Isi video verifikasi (komponen 2)
1. Jalankan `pipeline_final.py` → tunjukkan md5 keluaran `745bc3add88bf14b329c6bd467d36c1f`
   identik dengan `submission_SD2026040000061.csv` yang dikumpulkan.
2. Telusuri `relabel_manifest.csv` (350 koreksi) dan `train_dropped/` (135 drop) sebagai
   jejak audit pembersihan label — tunjukkan bahwa keputusan per gambar oleh manusia.
3. Tunjukkan `diagnostics/raw_emb.npz` berasal dari `modal-train/embed_raw.py` (backbone
   pretrained publik, tanpa data eksternal masuk training/inferensi).
4. Tunjukkan split validasi seed 42 dan bahwa tidak ada label test yang dipakai.

## 7. Keputusan yang masih terbuka

1. **200 citra test ikut atau tidak** — label E-nya prediksi model, bukan kunci panitia.
   Kalau ikut, harus disebut eksplisit di metodologi.
2. **Citra tambahan dari sumber daring** — diizinkan, bisa menutup kekosongan lensa bahaya
   (CRT pecah, baterai bengkak), tapi menambah beban dokumentasi lisensi.
3. **Arah harmonisasi** — turunkan semua ke 150 px (buang detail 1.364 foto bagus) vs
   super-resolusi korpus rendah (mahal + artefak generatif masuk data analisis).
4. **Banyak citra bukan limbah** — bisa jadi kelemahan yang disembunyikan, atau justru
   diangkat sebagai lensa kelima: kondisi barang (utuh / rusak / terbongkar / tercampur),
   yang paling relevan untuk keputusan reuse-vs-recycle.
