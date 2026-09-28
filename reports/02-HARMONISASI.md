# Fase 0 — Harmonisasi akuisisi: confound-nya bukan resolusi

Tanggal: 5 Sep 2026. Menggantikan kesimpulan sementara di
`00-EDA-DAN-ROADMAP.md` §4 ("perbaikan yang benar ada di input: turunkan
resolusi"). Resolusi memang di input, tapi bukan resolusi yang jadi masalah.

Artefak: `ewaste_emb_harm_res.npz`, `ewaste_emb_harm_resjpeg.npz`,
`tabel1_harmonisasi.csv`. Script: `modal-train/harmonize.py`,
`modal-train/embed_harmonized.py`, `semifinal/scripts/eval_harmonisasi.py`.

## 1. Pertanyaannya

Subset E adalah dua korpus yang tidak bersinggungan: 2.815 citra persis 150×150
hasil scraping dan 1.364 foto lapangan resolusi tinggi. Probe linier menebak
korpus dengan AUC 0,993 di SigLIP2; tetangga lintas-korpus cuma 1,6% padahal
harapan acak 44,0%. Hipotesis awal: penyebabnya beda resolusi efektif, jadi
turunkan semua citra ke 150 px sebelum embedding dihitung.

## 2. Hipotesis resolusi: salah

Semua citra di-center-crop persegi lalu diturunkan ke 150×150 (LANCZOS), lalu
di-embed ulang dengan 8 backbone dan transform yang identik dengan penyisihan.

Rata-rata 8 backbone: probe AUC 0,943 → 0,908. SigLIP2: 0,993 → 0,983,
tetangga lintas-korpus 1,6% → 2,6%. Koreksi ruang fitur yang sudah ada
(INLP+CORAL) mencapai 9,7% — masih 3,7× lebih efektif daripada perbaikan input
versi ini.

Kontrol kejujuran pipeline: satu backbone (SigLIP2) juga di-embed ulang **tanpa**
harmonisasi lewat kode baru yang sama. Cosine terhadap cache lama minimum 0,9999,
median 1,0000. Jadi angka di atas hasil nyata, bukan bug preprocessing.

## 3. Pemisahnya bukan isi gambar

Probe korpus dijalankan **di dalam** pasangan klaster yang isinya sama tapi
korpusnya beda. Pasangan terbersih 08&09: dua-duanya set meja kerja in-situ
(monitor + keyboard + mouse), cuma beda korpus.

| pasangan | n | asli | INLP+CORAL | harm-res | harm-res+jpeg |
|---|---:|---:|---:|---:|---:|
| 08&09 meja kerja | 380 | 0,999 | 0,982 | 0,999 | **0,523** |
| 13&15 layar genggam | 503 | 0,996 | 0,982 | 0,993 | 0,874 |
| 02&14 TV/monitor (kontrol negatif) | 539 | 1,000 | 0,990 | 0,999 | 0,995 |

Kolom `harm-res` menutup kasusnya: dengan isi gambar dikunci, resolusi
disamakan, AUC tetap 0,999. Yang memisahkan bukan konten dan bukan resolusi.

02&14 sengaja dipasang sebagai kontrol negatif — isinya memang berbeda (layar &
kamera campuran lawan TV & monitor). AUC-nya tetap 0,995 di semua penanganan,
dan memang **harus** begitu: itu sinyal konten yang sah dan tidak boleh hilang.
Kalau ia ikut jatuh, penanganannya menghapus informasi, bukan confound.

## 4. Sidik jarinya: riwayat encoder JPEG

Pemindaian tabel kuantisasi seluruh korpus scraping:

| korpus | n | tabel kuantisasi | jumlah luma | subsampling |
|---|---:|---|---|---|
| scraping 150 px | 2.815 | **2 tabel unik; 2.814 identik** | 1858, std 0,000 | 4:2:0 |
| foto lapangan | 1.364 | tersebar | 64 – 10.714, std 931 | campur |

Tabel yang dipakai 2.814 citra itu persis tabel standar IJG pada **quality 75**
dengan subsampling 4:2:0 — terverifikasi dengan mengencode ulang citra uji pada
q75 dan mencocokkan kedua tabel elemen per elemen. Seluruh korpus scraping lewat
satu encoder dengan satu setelan.

Ini menjelaskan kenapa langkah resolusi tidak mempan: downsample LANCZOS dari
4608 px menghasilkan citra mulus tanpa blok 8×8, sementara korpus scraping
membawa artefak kuantisasi pada skala native-nya. Menyamakan resolusi justru
menyisakan kontras artefak dalam keadaan utuh.

Penanganan: re-encode **semua** citra lewat encoder yang sama (q75, 4:2:0)
setelah diturunkan ke 150 px.

## 5. Hasil

Rata-rata 8 backbone (`tabel1_harmonisasi.csv` menyimpan rincian per backbone):

| penanganan | probeAUC | xKNN@15 | NMI(c,prov) | silhouette |
|---|---:|---:|---:|---:|
| asli | 0,943 | 0,067 | 0,205 | 0,220 |
| INLP+CORAL (ruang fitur) | 0,882 | 0,110 | 0,091 | 0,050 |
| harm-res | 0,908 | 0,089 | 0,184 | 0,223 |
| **harm-res+jpeg** | **0,781** | 0,132 | 0,166 | **0,222** |
| keduanya | 0,889 | 0,151 | 0,078 | 0,049 |

SigLIP2 sendiri: 0,993 → 0,875 (probe), 0,016 → 0,107 (xKNN), 0,329 → 0,212
(NMI), silhouette 0,155 → 0,157.

Dua hal yang menentukan pilihan:

1. **Struktur selamat.** Koreksi ruang fitur menjatuhkan silhouette 0,220 →
   0,050 di semua backbone; ia meratakan ruang, bukan cuma menghapus confound.
   Harmonisasi input meninggalkan silhouette di 0,222 sambil menurunkan probe
   AUC lebih jauh.
2. **Tidak butuh label korpus.** INLP dan CORAL keduanya dipasang memakai label
   korpus, dan CORAL bahkan tidak bisa diterapkan ke citra baru tanpa tahu dulu
   citra itu dari korpus mana. Untuk paper unsupervised discovery, memerlukan
   label confound untuk membuang confound adalah kelemahan telak. Harmonisasi
   input tidak menyentuh label sama sekali.

Menumpuk keduanya (`keduanya`) menaikkan xKNN ke 0,151 tapi membayar dengan
silhouette 0,049, dan pada pasangan kembar 08&09 justru lebih buruk (0,592 lawan
0,523). Tidak dipakai.

## 6. Kenapa xKNN tidak perlu mencapai 0,440

0,440 adalah harapan acak **kalau** kedua korpus punya sebaran isi yang sama.
Sebarannya tidak sama: korpus scraping didominasi foto katalog produk utuh,
korpus lapangan berisi tumpukan dan bongkaran in-situ. Sebagian dari 0,440 yang
belum tertutup adalah perbedaan isi yang nyata, dan itu justru yang mau kita
klaster.

Ukuran yang benar adalah pasangan berkonten terkunci: di 08&09, AUC turun
0,999 → 0,523, dengan tebak-acak di 0,500. Di sana confound-nya habis.

## 7. Keputusan untuk fase berikutnya

- Ruang representasi untuk Lensa A–D: **`ewaste_emb_harm_resjpeg.npz`**.
- `ewaste_emb.npz` dan `ewaste_emb_harm_res.npz` disimpan sebagai baris ablasi
  Tabel 1, bukan dibuang.
- `clusters_k16.csv`, `montage/`, dan pembacaan klaster di
  `00-EDA-DAN-ROADMAP.md` §2 dihitung di atas embedding **lama**. Perlu dihitung
  ulang di ruang terharmonisasi sebelum dipakai untuk klaim apa pun.

## 8. Partisi k=16 setelah harmonisasi

`clusters_k16_harm.csv`, protokol sama dengan yang lama (L2 → PCA-64 → k-means
k=16, seed 0). **Menggantikan pembacaan di `00-EDA-DAN-ROADMAP.md` §2.**

Prediksi dipasang sebelum melihat hasil, dan diukur sebagai purity: bisakah
partisi baru masih membedakan anggota klaster lama a dari b, kalau tiap klaster
baru diberi label mayoritas?

| pasangan lama | n | dasar | purity ruang asli | purity terharmonisasi | |
|---|---:|---:|---:|---:|---|
| 08&09 meja kerja (isi sama) | 380 | 0,711 | 0,974 | **0,711** | menyatu |
| 13&15 layar genggam (isi sama) | 503 | 0,573 | 0,998 | **0,573** | menyatu |
| 01&12 mouse vs keyboard (isi beda) | 481 | 0,516 | 0,981 | **0,985** | tetap pisah |

Purity yang jatuh **persis ke dasarnya** berarti partisi baru sama sekali tidak
bisa lagi memisahkan meja-kerja-150px dari meja-kerja-HD. Kolom "purity ruang
asli" adalah k-means yang dijalankan ulang di embedding lama — ia memulihkan
keterpisahan (0,974/0,998/0,981), jadi penyatuan itu bukan variasi seed.

Ukuran "overlap anggota bersama" gaya EDA §4 dibuang: satu anggota nyasar sudah
cukup membuat sebuah klaster dihitung bersama, dan dengan ukuran itu kontrol
negatif 01&12 ikut lolos sebagai "menyatu" (232 dari 233) — yang jelas salah.

Komposisi korpus per klaster, sebelum dan sesudah:

| | klaster nyaris murni satu korpus | median %hi-res |
|---|---:|---:|
| partisi lama | 15 dari 16 | 2,1 |
| partisi terharmonisasi | 10 dari 16 | 9,6 |

Dua klaster kini benar-benar bercampur, dan keduanya persis pasangan kembar yang
diprediksi menyatu: klaster 0 (69,2% hi-res) dan klaster 2 (44,3% — praktis sama
dengan proporsi populasi 44,0%).

Empat klaster masih nyaris murni, tapi dua yang ekstrem punya penjelasan isi,
bukan artefak: klaster 4 (tumpukan & scrap, 99,1%) dan klaster 9 (laptop foto
lapangan, 98,9%). Tumpukan scrap memang tidak ada di korpus katalog hasil
scraping. Ini penegasan lain dari §6: sebagian sisa keterpisahan adalah
perbedaan isi yang nyata.

### Pembacaan 16 klaster (SigLIP2, terharmonisasi)

| k | isi | n | %hi-res |
|---|---|---:|---:|
| 0 | set meja kerja in-situ: keyboard, mouse, monitor | 312 | 69,2 |
| 1 | pemutar piringan hitam & radio retro | 116 | 1,7 |
| 2 | ponsel & tablet, banyak layar retak | 467 | 44,3 |
| 3 | audio portabel: CD player, boombox, speaker | 159 | 4,4 |
| 4 | tumpukan & scrap e-waste, PCB massal | 425 | 99,1 |
| 5 | baterai: pak laptop, sel, aki | 253 | 2,4 |
| 6 | perangkat rusak: layar retak, TV bergaris, bongkaran | 186 | 93,5 |
| 7 | foto gelap / layar menyala dalam gelap | 169 | 14,8 |
| 8 | TV & monitor, termasuk CRT | 265 | 4,5 |
| 9 | laptop, foto lapangan | 177 | 98,9 |
| 10 | PCB & papan sirkuit telanjang | 297 | 15,5 |
| 11 | mesin cuci | 298 | 3,4 |
| 12 | mouse | 273 | 14,7 |
| 13 | microwave & oven | 268 | 3,0 |
| 14 | printer & plotter | 297 | 4,0 |
| 15 | keyboard | 217 | 1,4 |

**Temuan yang relevan untuk keputusan terbuka #4:** klaster 6 (perangkat rusak)
dan 7 (foto gelap) bukan kategori jenis perangkat — keduanya kategori *kondisi*
dan *kondisi pengambilan*. Struktur alami data sudah menyediakan sumbu kondisi
barang tanpa diminta, yang memperkuat opsi menjadikannya lensa kelima.

Montase eksemplar: `semifinal/montage/clusters_k16_harm_siglip2_sheet*.png`.

## 9. Yang belum dikerjakan

- **Asimetri generasi JPEG.** Korpus scraping kini lewat dua generasi kompresi,
  korpus lapangan satu. Sisa asimetri ini jauh lebih kecil daripada kontras
  "punya artefak 8×8" lawan "tidak punya", tapi belum diukur. Uji: encode ganda
  korpus lapangan juga, lihat apakah probe AUC turun lagi.
- **Sumber sisa 0,781.** Kandidat berikutnya: pipeline warna kamera (white
  balance, sharpening in-camera), dan gamut sRGB hasil konversi CMS.
- **Pasangan 13&15 berhenti di 0,874.** Belum jelas apakah sisanya konten
  (tablet lawan ponsel memang beda) atau akuisisi.
