# Pengelompokan perangkat elektronik dan profil material visual

Repo ini memuat [paper semifinal](paper/KARYA-ILMIAH-SD2026040000061-4-CONTOH.pdf), [notebook pipeline](notebooks/01_pipeline_dan_evaluasi.ipynb), kode tahap utama, dan hasil terpilih yang mendukung klaim paper.

## Mulai dari sini

1. Buka `notebooks/01_pipeline_dan_evaluasi.ipynb` di Jupyter atau VS Code dengan Python 3.
2. Jalankan sel berurutan. Sel hanya membaca tabel lokal dan memakai Python standard library serta IPython bawaan notebook.
3. Alternatif terminal: `python reproduce_summary.py` dari root repo.

Notebook menunjukkan alur penyeragaman citra, ekstraksi *embedding*, fusion, K-Means, pemilihan K, transfer Bangladesh, dan analisis material. Perhitungan yang berjalan langsung ialah pemilihan K dari chooseR, rata-rata ARI 100 seed, dan macro Top-1 dari 710 prediksi external. Metrik material dibaca dari hasil tersimpan karena prediksi per citra tidak disertakan.

## Susunan repo

| Folder | Isi |
|---|---|
| `paper/` | PDF dan DOCX 20 halaman, sumber naskah, gambar, serta [evidence manager](paper/EVIDENCE-MANAGER.md) |
| `notebooks/` | Alur yang bisa dibaca dan dijalankan tanpa model atau GPU |
| `reporting/` | Class pembaca tabel dan pemeriksaan angka utama |
| `scripts/` | Runner tiap tahap eksperimen untuk dibaca atau dijalankan dalam lingkungan proyek asal |
| `tables/` | Hasil terpilih untuk pemilihan K, stabilitas, transfer, dan material |
| `reports/` | Laporan eksperimen yang mendukung keputusan metodologi |

## Hubungan dengan rubrik

Paper menjelaskan masalah foto multisumber dan alasan konfigurasi yang dipilih. Tabel hasil dan notebook memperlihatkan kualitas cluster, pemilihan K, stabilitas, serta transfer ke data external. Gambar cluster dan profil material memberi contoh interpretasi. Runner dipisahkan menurut tahap agar keputusan dan parameter bisa ditelusuri. [Konteks versi](LATEST-PAPER-CONTEXT.md) menjelaskan perbedaan Fusion dan Fusion-L2.

Data citra mentah, seluruh *embedding*, serta bobot model tidak dibagikan di repo ini. Karena itu, notebook mereproduksi pemeriksaan angka dari artefak hasil, bukan eksperimen dari awal. Analisis DMS46 menunjukkan material yang tampak pada citra, bukan komposisi fisik perangkat.
