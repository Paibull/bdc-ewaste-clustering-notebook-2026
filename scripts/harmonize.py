"""Harmonisasi akuisisi untuk subset e-waste semifinal.

Subset E berisi dua korpus: 2.815 citra persis 150x150 (scraping web) dan 1.364
foto lapangan resolusi tinggi. Backbone memisahkan keduanya nyaris sempurna
(probe AUC 0,993 di SigLIP2), dan pemisahnya BUKAN isi gambar: di dalam pasangan
klaster berkonten sama (meja kerja 150px vs meja kerja HD) AUC tetap 0,999.

Dua sidik jari akuisisi yang terukur, keduanya dibereskan di sini:

1. RESOLUSI. Korpus scraping seragam 150x150; foto lapangan sampai 4608x3456.
   -> center-crop persegi lalu turunkan ke 150 px.
   Sendirian, langkah ini nyaris tidak mempan: AUC 0,993 -> 0,983,
   tetangga lintas-korpus 1,6% -> 2,6% (harapan acak 44%).

2. RIWAYAT KOMPRESI JPEG. 2.814 dari 2.815 citra scraping punya tabel
   kuantisasi yang IDENTIK (jumlah luma 1858, std 0,000 antar-citra) =
   tabel standar IJG pada quality 75 dengan subsampling 4:2:0. Korpus hi-res
   tersebar liar (jumlah luma 64..10714, std 931) karena berasal dari puluhan
   encoder kamera. Downsample LANCZOS menghasilkan citra mulus tanpa blocking
   8x8, jadi langkah 1 justru menyisakan kontras artefak yang utuh.
   -> re-encode SEMUA citra lewat encoder yang sama (q75, 4:2:0).

Dipakai `embed_harmonized.py` (di Modal) dan diuji lokal lewat `python harmonize.py`.
"""
import io

from PIL import Image

HARM_PX = 150        # ukuran korpus scraping
JPEG_Q = 75          # quality tabel kuantisasi korpus scraping (terverifikasi 2814/2815)
JPEG_SUBSAMPLING = 2  # 4:2:0, sama dengan korpus scraping


def harmonize(im, px=HARM_PX, jpeg=True):
    """Samakan resolusi efektif DAN riwayat encoder JPEG.

    Center-crop persegi -> resize px x px (LANCZOS) -> re-encode JPEG q75 4:2:0.

    Citra yang sudah px x px lolos tahap resize tanpa resampling. Citra scraping
    tetap kena satu generasi JPEG tambahan (jadi dua generasi total, lawan satu
    generasi untuk foto lapangan) — asimetri sisa yang jauh lebih kecil daripada
    kontras "punya artefak 8x8" lawan "tidak punya sama sekali".

    LANCZOS, bukan BICUBIC: downsample 4608 -> 150 tanpa antialias menyisakan
    aliasing yang justru jadi sidik jari korpus baru.
    """
    w, h = im.size
    s = min(w, h)
    if w != h:
        l, t = (w - s) // 2, (h - s) // 2
        im = im.crop((l, t, l + s, t + s))
    if s != px:
        im = im.resize((px, px), Image.LANCZOS)
    if jpeg:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_Q, subsampling=JPEG_SUBSAMPLING)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    return im


def _selfcheck():
    import numpy as np

    rs = np.random.RandomState(0)

    # geometri: korpus scraping tidak di-resample (bandingkan tanpa tahap JPEG)
    a = Image.fromarray(rs.randint(0, 256, (150, 150, 3), np.uint8))
    assert np.array_equal(np.asarray(harmonize(a, jpeg=False)), np.asarray(a)), \
        "citra 150px tidak boleh di-resample"

    # foto lapangan 4:3 -> crop tengah persegi lalu turun ke 150
    b = Image.fromarray(rs.randint(0, 256, (864, 1152, 3), np.uint8))
    assert harmonize(b).size == (150, 150)

    # persegi tapi bukan 150 -> tetap di-resize
    assert harmonize(Image.new("RGB", (300, 300))).size == (150, 150)

    # crop diambil dari TENGAH, bukan pojok
    wide = Image.new("RGB", (300, 100), (0, 0, 0))
    wide.paste(Image.new("RGB", (100, 100), (255, 0, 0)), (100, 0))
    got = np.asarray(harmonize(wide, px=10, jpeg=False))
    assert got[:, :, 0].min() > 200 and got[:, :, 1].max() < 50, "crop tidak terpusat"

    # sidik jari encoder: keluaran harus membawa tabel kuantisasi korpus scraping
    buf = io.BytesIO()
    harmonize(b).save(buf, "JPEG", quality=JPEG_Q, subsampling=JPEG_SUBSAMPLING)
    buf.seek(0)
    q = Image.open(buf).quantization
    assert sum(q[0]) == 1858, f"jumlah tabel luma {sum(q[0])}, harusnya 1858"

    # dan foto hi-res harus benar-benar berubah oleh tahap JPEG
    assert not np.array_equal(np.asarray(harmonize(b, jpeg=True)),
                              np.asarray(harmonize(b, jpeg=False))), "tahap JPEG tidak jalan"

    print("harmonize selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
