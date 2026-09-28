# Pembaruan konteks naskah, 28 September 2026

## Alur penulisan yang berlaku

Naskah dimulai dari persoalan e-waste dan kebutuhan mengenali perangkat,
kemudian penelitian terkait, EDA, alasan harmonisasi dan fusion, pengujian
cluster, external transfer, serta dukungan material. MD eksperimen menjadi
sumber fakta, bukan pola kalimat. Rancang pertanyaan dan hubungan bukti,
fungsi paragraf, baru kalimatnya. Abstrak dan kesimpulan menjawab tujuan
yang sama. Gunakan kata familiar seperti *error*, bukan galat.

K-Means merupakan komponen pipeline usulan, bukan algoritma baru.
Kata kompetitif berlaku bagi evaluator internal dan adapter pembanding
yang diuji. Tidak ada equivalence test atau perbandingan biaya lengkap.
TURTLE ICML 2024 mendukung relevansi K-Means pada gabungan frozen features,
tetapi TURTLE unggul pada benchmark penulis. MCSF CVPR 2026 menjadi dasar
pembanding fusion spectral/contrastive. Register mencatat PDF dan halaman
yang mendukung klaim tersebut. Kini tersedia 26 PDF lokal.

Holdout adalah 600 citra internal untuk evaluasi material, bukan Bangladesh
atau Roboflow. Gambar menggunakan contoh cluster, tanpa istilah atlas dan
ikon gembok. Referensi gaya Manik 2024, Suitela 2023, dan Pambayun 2024
digunakan untuk pola argumen dan kosakata tanpa menyalin kalimat.

Dokumen ini memperbarui keputusan lama pada `01-MASTER-SOURCE-PACK.md`, README, dan AGENTS. Gunakan angka hasil terbaru sesuai protokolnya; jangan mencampur versi partisi.

- Medoid **Fusion K12** berasal dari studi 100 seed/resampling. K14 tetap K terpilih pada evaluator penelusuran awal. Keduanya menggunakan aturan pemilihan yang berbeda.
- Aturan resampling adalah **adaptasi chooseR**: percentile bootstrap median skor per citra dan jumlah cluster melewati threshold. Original chooseR memakai skor per cluster dan BCa. Jangan menyebut replikasi identik atau jumlah cluster benar.
- Strict external transfer dan material holdout memakai **Fusion-L2 K12** yang mempunyai L2 setelah PCA per view. Medoid Fusion tidak memiliki normalisasi tersebut. ARI antarkedua partisi 0,974794. Validasi Fusion-L2 belum menguji persis medoid Fusion.
- Strict Macro Top-1 fused K12 = 0,888 pada 710 citra empat family. Seluruh centroid dicari; unmapped dihitung error. Nilai 0,905 adalah protokol lama yang membatasi pencarian pada family mapped.
- Material holdout incremental = 0,1145 untuk enam fraction dan 0,1528 untuk lima known. Ini penurunan SSE relatif terhadap baseline source/statistik akuisisi, bukan pixel accuracy/komposisi kimia. Populasi lima known adalah 598, bukan 600.
- Main thesis naskah adalah pipeline discovery family dari diagnosis/harmonisasi, fusion frozen representation, resampling, dan validasi dengan cakupan eksplisit. Tidak ada klaim algoritma baru atau kemenangan seluruh advanced methods.

## Implementasi terbaru

`scripts/run_k_stability_100.py`, `scripts/evaluate_strict_transfer_material_holdout.py`, dan `scripts/build_final_multiresolution_cases.py` menelusuri hasil terbaru. Laporan 113–115 dan tabel di `tables/k_stability_100`, `tables/final_claim_checks`, serta `tables/final_multiresolution` menjadi bukti primernya.

`paper/build_cluster_appendix.py` memakai kembali fungsi fusion studi stabilitas dan assignment tersimpan. Script menghasilkan proyeksi t-SNE serta satu citra asli terdekat ke centroid fused untuk masing-masing cluster, tanpa inference backbone atau clustering ulang. Parameter proyeksi: perplexity 30, seed 42, initialization PCA, 1.000 iterasi. Proyeksi tidak digunakan sebagai metric clustering. Pemilihan prototype memakai squared Euclidean distance pada fused 100-D; berbeda dari proxy RADIO pada atlas utama.

File script merupakan salinan sumber untuk pemeriksaan desain. Path script mengikuti layout proyek asal `semifinal/`; menjalankan compute memerlukan layout tersebut, data, embedding/checkpoint, dan environment eksperimen yang tidak lengkap dalam repository source pack. Jangan menjalankan ulang eksperimen untuk menyunting paper.

## Naskah dan figure terbaru

Versi Word/PDF di `paper/KARYA-ILMIAH-SD2026040000061-REVISI` berjumlah **20 halaman termasuk cover, referensi, dan lampiran**. Mulai penyuntingan dari `paper/NASKAH-PAPER-BDC.md`; `paper/DRAFT-PAPER-BDC.md` adalah sumber dengan slot authoring. Delapan figure utama dan satu figure lampiran tersedia di `paper/figures`. Pipeline dipisah menjadi Gambar 1 pengelompokan perangkat, Gambar 2 penerapan pada citra Bangladesh, dan Gambar 3 analisis material. Blok fitur pada Gambar 1 adalah ilustrasi skematis. Lampiran C menampilkan contoh dua cluster layar pada K16. Jangan mengubahnya menjadi attribution map atau ground truth semantik.

Gambar 4, 5, dan 7 menggunakan Apache ECharts 6.1.0 dari hasil tersimpan. Gambar 8 memperlihatkan citra asli dan profil DMS46 dari tiga contoh yang dapat ditelusuri ke tabel prediksi. Chart numerik material tetap tersedia di `paper/figures/fig6_material.png`, tetapi angkanya dijelaskan dalam naskah agar batas 20 halaman terpenuhi. SVG adalah sumber vector, sedangkan PNG dimasukkan ke Word. Register paper serta PDF lokal tersedia di `paper/references`, dengan kontribusi dan batas setiap sumber. Rujukan laras Indonesia utama adalah Manik et al. (2024), JPHPI SINTA 1, dengan sumber tambahan Pambayun dan Azhar (2024). Status indeks berasal dari penerbit, bukan asumsi kuartil atau citation count tinggi. Sumber gaya tidak dijadikan sitasi metode.

Pembukaan menggunakan Jaipuria et al. (2022) untuk clustering bias kondisi citra dan Ehrlich et al. (2021) untuk pengaruh kompresi JPEG pada vision. Kedua klaim dipisahkan. Temuan JPEG spesifik data internal tetap berasal dari eksperimen proyek. Naskah menggunakan istilah asing italic, menghindari em dash, dan menggunakan titik atau kata penghubung untuk transisi narasi.
