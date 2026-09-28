"""Tabel 1 paper: apa yang benar-benar menutup confound dua-korpus?

Subset E terdiri dua korpus yang nyaris tidak bersinggungan di ruang embedding.
Script ini mengadu empat penanganan pada tiga ukuran yang sama dengan EDA:

  asli            embedding apa adanya
  INLP+CORAL      koreksi di RUANG FITUR (penyeragaman per-korpus + hapus 3 arah)
  harm-res        koreksi di INPUT: semua citra diturunkan ke 150 px
  harm-res+jpeg   koreksi di INPUT: 150 px DAN re-encode JPEG q75 4:2:0

Ukuran:
  probeAUC     seberapa mudah probe linier menebak korpus asal (0,5 = tak terpisahkan)
  xKNN@15      proporsi tetangga terdekat dari korpus SEBERANG (acak = 0,440)
  NMI(c,prov)  informasi korpus yang dipakai k-means k=16 (0 = tidak dipakai)
  sil          silhouette partisi k=16, penjaga supaya struktur tidak ikut hancur

Tabel kedua menjawab "apakah pemisahnya isi gambar atau cara pengambilannya":
probe korpus dijalankan DI DALAM pasangan klaster yang isinya sama tapi korpusnya
beda (mis. meja kerja 150px lawan meja kerja HD). Kalau AUC di situ tetap tinggi,
yang memisahkan bukan konten.

Jalankan dari akar repo:  python semifinal/scripts/eval_harmonisasi.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
SEMI = ROOT / "semifinal"
BB = ["dinov3", "radio", "aimv2", "siglip2", "dinov2", "siglip", "convnext", "eva02"]
PCA_D = 128
K = 16

# Pasangan klaster k=16 (kolom siglip2 clusters_k16.csv) untuk memisahkan "konten" dari
# "akuisisi". 08&09 adalah pasangan terbersih: keduanya set meja kerja in-situ, isinya sama
# persis, cuma korpusnya beda -> di sinilah AUC harus jatuh ke 0,5 kalau confound benar hilang.
# 02&14 sengaja disertakan sebagai KONTROL NEGATIF: isinya memang beda (layar & kamera
# campuran lawan TV & monitor), jadi AUC tinggi di situ adalah sinyal konten yang sah dan
# memang TIDAK boleh hilang.
KEMBAR = {"08&09 meja kerja": (8, 9), "13&15 layar genggam": (13, 15), "02&14 TV/monitor": (2, 14)}


def L2(X):
    X = X.astype(np.float32)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)


def probe(X, y):
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 10:
        return float("nan")
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=3,
                          method="predict_proba")[:, 1]
    a = roc_auc_score(y, p)
    return max(a, 1 - a)


def xknn(X, prov, k=15):
    S = X @ X.T
    np.fill_diagonal(S, -1)
    nb = np.argpartition(-S, k, axis=1)[:, :k]
    return float((prov[nb] != prov[:, None]).mean())


def inlp(X, y, r):
    """Iterative nullspace projection: hapus r arah linier yang mengkodekan y."""
    Xc, P = X.copy(), np.eye(X.shape[1], dtype=np.float32)
    for _ in range(r):
        w = LogisticRegression(max_iter=1000).fit(Xc, y).coef_[0].astype(np.float32)
        w /= np.linalg.norm(w)
        Q = np.eye(len(w), dtype=np.float32) - np.outer(w, w)
        P, Xc = P @ Q, Xc @ Q
    return P


def coral(X, y):
    """Penyeragaman per-korpus: pusatkan dan skalakan tiap korpus sendiri-sendiri."""
    Z = X.copy()
    for g in np.unique(y):
        m, s = X[y == g].mean(0), X[y == g].std(0) + 1e-6
        Z[y == g] = (X[y == g] - m) / s
    return Z


def prep(X):
    return L2(PCA(PCA_D, random_state=0).fit_transform(L2(X)))


def metrics(X, prov):
    lab = KMeans(K, n_init=10, random_state=0).fit_predict(X)
    return probe(X, prov), xknn(X, prov), nmi(lab, prov), silhouette_score(X, lab)


def main():
    base = np.load(SEMI / "ewaste_emb.npz", allow_pickle=True)
    kb = [str(x) for x in base["key"]]

    def load(fname):
        p = SEMI / fname
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=True)
        kh = [str(x) for x in d["key"]]
        order = np.arange(len(kb)) if kb == kh else np.array([kh.index(k) for k in kb])
        return d, order

    harm = {"harm-res": load("ewaste_emb_harm_res.npz"),
            "harm-res+jpeg": load("ewaste_emb_harm_resjpeg.npz")}
    for nm, v in harm.items():
        if v is None:
            print(f"[!] {nm}: npz belum ada, dilewati")
    harm = {k: v for k, v in harm.items() if v is not None}

    st = pd.read_csv(SEMI / "imgstats.csv").set_index("key").loc[kb]
    prov = (st[["W", "H"]].max(axis=1) > 160).astype(int).values   # 1 = korpus resolusi tinggi
    c16 = pd.read_csv(SEMI / "clusters_k16.csv").set_index("key").loc[kb, "siglip2"].values
    chance = 2 * prov.mean() * (1 - prov.mean())
    print(f"korpus hi-res {prov.sum()} · 150px {(1-prov).sum()} · xKNN harapan acak {chance:.3f}\n")

    # --- kontrol kejujuran pipeline: re-embed TANPA harmonisasi harus mereproduksi cache ---
    for nm, (d, order) in harm.items():
        if "siglip2_ctrl" in d.files:
            cos = (L2(base["siglip2"]) * L2(d["siglip2_ctrl"][order])).sum(1)
            verdict = "OK" if cos.min() > 0.99 else "GAGAL — selisih di bawah bukan efek akuisisi"
            print(f"[kontrol {nm}] cos(siglip2_ctrl, cache): min {cos.min():.4f} "
                  f"median {np.median(cos):.4f} -> {verdict}")
    print()

    def debias(X):
        Xc = coral(X, prov)
        return L2(Xc @ inlp(Xc, prov, 3))

    def variants(b):
        yield "asli", prep(base[b])
        yield "INLP+CORAL", debias(prep(base[b]))
        for nm, (d, order) in harm.items():
            if b in d.files:
                yield nm, prep(d[b][order])
        rj = harm.get("harm-res+jpeg")
        if rj and b in rj[0].files:
            yield "keduanya", debias(prep(rj[0][b][rj[1]]))

    hdr = (f"{'backbone':10s} {'penanganan':14s} {'probeAUC':>9s} {'xKNN@15':>8s} "
           f"{'NMI(c,prov)':>12s} {'sil':>7s}")
    print(hdr)
    print("-" * len(hdr))
    rows, keep = [], {}
    for b in BB:
        for nm, X in variants(b):
            m = metrics(X, prov)
            print(f"{b:10s} {nm:14s} {m[0]:9.3f} {m[1]:8.3f} {m[2]:12.3f} {m[3]:7.3f}", flush=True)
            rows.append(dict(backbone=b, penanganan=nm, probe_auc=m[0], xknn15=m[1],
                             nmi_prov=m[2], silhouette=m[3]))
            if b == "siglip2":
                keep[nm] = X
        print()
    print(f"target: probeAUC -> 0,500 · xKNN@15 -> {chance:.3f} · NMI -> 0,000\n")

    # --- konten atau akuisisi? probe di dalam pasangan klaster berkonten sama (siglip2) ---
    print("probe korpus DI DALAM pasangan klaster berkonten sama (siglip2):")
    hdr2 = f"{'pasangan':20s} {'n':>5s} " + " ".join(f"{nm:>14s}" for nm in keep)
    print(hdr2)
    print("-" * len(hdr2))
    for nm_pair, (a, c) in KEMBAR.items():
        m = np.isin(c16, [a, c])
        cells = " ".join(f"{probe(X[m], prov[m]):14.3f}" for X in keep.values())
        print(f"{nm_pair:20s} {m.sum():5d} {cells}")
        rows.append(dict(backbone="siglip2", penanganan="kembar:" + nm_pair, probe_auc=np.nan,
                         xknn15=np.nan, nmi_prov=np.nan, silhouette=np.nan,
                         **{f"auc_{k}": probe(X[m], prov[m]) for k, X in keep.items()}))

    out = SEMI / "tabel1_harmonisasi.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
