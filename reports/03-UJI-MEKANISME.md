# Uji validitas mekanisme empat lensa di Clevr-4

Tanggal: 5 Sep 2026. Artefak: `semifinal/clevr4_siglip2.npz`. Script:
`modal-train/embed_clevr4.py`, `semifinal/scripts/uji_mekanisme_clevr4.py`.

## 1. Kenapa diuji di sini dulu

Rancangan empat lensa bertumpu pada satu klaim yang belum pernah diperiksa:
bahwa membangun sub-ruang konsep dari bank prompt, lalu clustering **di dalam**
sub-ruang itu, menghasilkan partisi menurut kriteria yang diminta. Di data
e-waste klaim itu tidak bisa diuji — tidak ada label kriteria untuk
membandingkan.

Clevr-4 (Vaze, Vedaldi & Zisserman, NeurIPS 2023; CC BY 4.0) menyediakan
tepat itu: 10.531 citra, empat kriteria ortogonal (shape, color, texture,
count), masing-masing 10 kelas, setiap citra berlabel keempat-empatnya.

Ambang lulus dipasang sebelum melihat hasil: sub-ruang harus dominan diagonal,
**dan** selisih diagonal−luar-diagonalnya harus lebih besar daripada milik ruang
penuh. Kalau ruang penuh sama bagusnya, proyeksi tidak menyumbang apa pun.

## 2. Yang dibandingkan

Ketiganya memakai k = 10 (jumlah kelas ground-truth) mengikuti protokol seluruh
lini Multi-MaP/Multi-Sub, dan bobot SigLIP2 yang sama untuk menara gambar dan
menara teks (`hf-hub:timm/ViT-SO400M-16-SigLIP2-384`).

| | ruang tempat k-means dijalankan |
|---|---|
| `full` | ruang citra penuh, PCA-100 — kontrol tanpa proyeksi |
| `sim` | similarity terhadap jangkar teks per kelas, di-z-score per sumbu |
| `span` | koordinat ortonormal pada rentang linear jangkar teks kriteria itu |

Bank prompt: 4 template × 10 kelas = 40 prompt per kriteria, kosakata kelasnya
dibaca dari label dataset, bukan ditulis tangan. Panjang token diverifikasi 64
(aturan SigLIP), bukan diasumsikan.

## 3. Hasil

AMI antara partisi dari sub-ruang (baris) dan label ground-truth (kolom):

**`full` — kontrol tanpa proyeksi**

| sub-ruang | color | count | shape | texture |
|---|---:|---:|---:|---:|
| (keempatnya identik) | −0,000 | 0,004 | **0,995** | −0,000 |

**`sim`**

| sub-ruang | color | count | shape | texture |
|---|---:|---:|---:|---:|
| color | **0,681** | 0,046 | 0,015 | 0,030 |
| count | 0,011 | **0,408** | 0,104 | 0,023 |
| shape | 0,000 | 0,004 | **0,971** | 0,004 |
| texture | 0,010 | 0,030 | 0,296 | **0,260** |

diagonal 0,580 · luar-diagonal 0,048 · selisih **+0,532**

**`span`**

| sub-ruang | color | count | shape | texture |
|---|---:|---:|---:|---:|
| color | **0,850** | 0,001 | 0,008 | 0,022 |
| count | 0,001 | **0,642** | 0,014 | 0,002 |
| shape | −0,000 | 0,004 | **0,976** | 0,001 |
| texture | 0,005 | 0,003 | 0,307 | **0,443** |

diagonal 0,728 · luar-diagonal 0,031 · selisih **+0,697**

Komplementaritas antar-lensa (rata-rata AMI luar-diagonal antar partisi):
`sim` 0,112 · `span` 0,060. Pasangan terburuk di `span` adalah shape↔texture
(0,313); lima pasangan lainnya di bawah 0,03.

Angka `span` dihitung ulang 7 September setelah audit implementasi menemukan
SVD sebelumnya ikut mengambil satu vektor singular nol yang muncul karena
jangkar dipusatkan. Implementasi final memangkas basis ke rank numerik
sebenarnya. Kesimpulan tidak berubah dan selisih diagonal justru naik 0,003.
Kontrol ruang penuh juga diselaraskan dari PCA-128 ke PCA-100 pada 7 September
agar mematuhi protokol proyek; seluruh angka yang ditampilkan di atas berasal
dari eksekusi PCA-100. Partisinya dan kesimpulannya tidak berubah.

## 4. Bacaan

**Temuan utama ada di baris kontrol, bukan di sub-ruangnya.** Ruang citra penuh
hanya menemukan **shape** (AMI 0,995) dan praktis buta terhadap tiga kriteria
lain (semuanya ≈ 0,00), padahal ketiganya ada di citra yang sama dan berlabel.
Clustering polos di fitur frozen yang bagus bukan cuma "kurang lengkap" — ia
memilih satu kriteria dan menyembunyikan sisanya tanpa memberi tanda. Ini
argumen terkuat yang kita punya untuk rancangan multi-lensa, dan berlaku
untuk data apa pun, bukan cuma e-waste.

Proyeksi sub-ruang memulihkan keempat kriteria. `span` mengalahkan `sim` di
semua kriteria, jadi **`span` yang dipakai untuk lensa e-waste**.

Batas yang jujur:

- **Texture paling lemah** (0,443) dan bocor paling banyak ke shape (0,310).
  Kalau ada lensa e-waste yang analog dengan tekstur — material permukaan,
  misalnya lensa B — hasilnya harus dibaca dengan kecurigaan yang sama.
- **Count 0,642 justru di atas dugaan**; menghitung objek terkenal sulit untuk
  model gaya CLIP. Ini menunjukkan sumbu konsep bisa menarik keluar informasi
  yang tidak dominan di ruang penuh.
- Baris `full` punya AMI antar-partisi 1,000 secara konstruksi: tanpa proyeksi,
  keempat "lensa" adalah k-means yang sama persis. Jadi angka itu bukan ukuran
  komplementaritas yang bermakna, melainkan penegasan bahwa tanpa proyeksi hanya
  ada satu partisi yang bisa didapat pada k tetap.

## 5. Tindak lanjut pada e-waste

Eksperimen e-waste, sensitivitas bank prompt, null model, dan pembanding
multiple clustering sudah selesai; hasil dan alasan pemilihan metode ada di
`06-EKSPERIMEN-LENSA.md`. NrKmeans, OrthogonalClustering, ENRC, dan TURTLE
dijalankan. Multi-MaP dan Multi-Sub diaudit pada commit sumbernya tetapi tidak
dijalankan karena implementasi resminya mengikat OpenAI CLIP dan `ImageFolder`;
port ke fitur SigLIP2 tersimpan akan menjadi implementasi baru yang tidak lagi
adil sebagai pembanding stok.

Validasi manusia melalui *intruder study* sudah dipaketkan, tetapi nilainya
belum boleh dilaporkan sampai tiga penilai independen mengisi lembar jawaban.
Varian sparse nonnegatif dan residual sengaja tidak ditambah: hasil utama sudah
lulus, sedangkan varian itu menambah derajat kebebasan tanpa hipotesis baru.
