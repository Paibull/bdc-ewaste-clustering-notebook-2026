# embed_harmonized.py — FASE 0 semifinal: re-embed subset e-waste (4.179 citra)
# setelah SEMUA citra diturunkan ke resolusi efektif 150 px.
#
# Kenapa: subset E terdiri dua korpus yang tidak bersinggungan (2.815 citra persis
# 150x150 hasil scraping vs 1.364 foto lapangan resolusi tinggi). Probe linier
# menebak korpus dengan AUC 0,993; tetangga lintas-korpus cuma 1,6% dari harapan
# acak 44%. Koreksi di ruang fitur (INLP/CORAL) cuma menaikkannya ke 9,5% —
# tambalan. Perbaikan yang benar ada di input: samakan resolusi efektif SEBELUM
# embedding dihitung.
#
# Operasi harmonisasi ada di `harmonize.py`. Dua varian, dua berkas keluaran:
#
#   --jpeg=False -> ewaste_emb_harm_res.npz      resolusi saja
#   --jpeg=True  -> ewaste_emb_harm_resjpeg.npz  resolusi + riwayat encoder JPEG
#
# Varian resolusi-saja SUDAH DIJALANKAN dan hasilnya nyaris nihil (AUC 0,993 ->
# 0,983; tetangga lintas-korpus 1,6% -> 2,6% dari target 44%). Itu yang menuntun
# ke temuan tabel kuantisasi — lihat docstring harmonize.py.
#
#   MODAL_PROFILE=faizakbar2301 modal run --detach modal-train/embed_harmonized.py
#   MODAL_PROFILE=faizakbar2301 modal volume get bdc-data ckpt/ewaste_emb_harm_resjpeg.npz ./semifinal/ewaste_emb_harm_resjpeg.npz
#   (path TANPA leading slash)
#
# Kontrol kejujuran pipeline: siglip2 juga diembed TANPA harmonisasi (kunci
# `siglip2_ctrl`). Cosine terhadap `siglip2` di ewaste_emb.npz harus ~1,0 — kalau
# tidak, selisih yang terukur berasal dari bug preprocessing, bukan dari resolusi.
#
# Tahan-mati: npz ditulis ulang + vol.commit() tiap backbone kelar.
from pathlib import Path
import modal
from config import CFG
from harmonize import HARM_PX

HERE = Path(__file__).parent

# Sama persis dengan embed_raw.py — resolusi input WAJIB identik supaya kontrol
# `siglip2_ctrl` sebanding dengan embedding yang sudah di-cache.
BACKBONES = [
    ("dinov3",   "vit_large_patch16_dinov3.lvd1689m", 224),
    ("radio",    "HF:nvidia/C-RADIOv4-SO400M", 224),
    ("aimv2",    "aimv2_large_patch14_224.apple_pt", 224),
    ("siglip2",  "OC:timm/ViT-SO400M-16-SigLIP2-384", 384),
    ("dinov2",   "vit_large_patch14_reg4_dinov2.lvd142m", 224),
    ("siglip",   "vit_so400m_patch14_siglip_384", 378),
    ("convnext", "convnextv2_large.fcmae_ft_in22k_in1k", 224),
    ("eva02",    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", 448),
]

CTRL_BB = "siglip2"      # backbone yang juga diembed tanpa harmonisasi (kontrol pipeline)

app = modal.App("embed-harmonized")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.0", "torchvision==0.20.0", "timm>=1.0.20",
                      "transformers>=4.45", "open-clip-torch", "numpy", "pillow", "pyyaml", "einops")
         .add_local_python_source("config", "harmonize")
         .add_local_dir(str(HERE / "configs"), "/root/configs")
         .add_local_file(str(HERE.parent / "semifinal" / "ewaste_keys.npz"),
                         "/root/ewaste_keys.npz"))
vol = modal.Volume.from_name(CFG.volume)


def out_path(jpeg):
    return f"/data/ckpt/ewaste_emb_harm_{'resjpeg' if jpeg else 'res'}.npz"


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/data": vol})
def run(jpeg: bool = True):
    OUT = out_path(jpeg)
    import re, numpy as np, torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.datasets import ImageFolder
    from PIL import Image
    import timm
    from timm.data import resolve_model_data_config, create_transform
    from harmonize import harmonize
    dev = "cuda"

    meta = np.load("/root/ewaste_keys.npz", allow_pickle=True)
    keys = [str(k) for k in meta["key"]]
    split = meta["split"]

    # ---- petakan key -> path di volume ----
    # train: key = nama berkas saja (konvensi tr_names di embed_raw.py), kelas ada di subfolder
    by_name = {}
    for p, _ in ImageFolder("/data/train_clean").samples:
        by_name[Path(p).name] = p
    # test: key = "test/<id>.jpg", id = angka pertama di stem (konvensi predict.py)
    IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
    by_id = {}
    for p in Path("/data/test").rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            m = re.search(r"(\d+)", p.stem)
            if m:
                by_id[int(m.group(1))] = str(p)

    files, missing = [], []
    for k in keys:
        if k.startswith("test/"):
            p = by_id.get(int(re.search(r"(\d+)", k).group(1)))
        else:
            p = by_name.get(k)
        files.append(p) if p else missing.append(k)
    assert not missing, f"{len(missing)} key tidak ketemu di volume, contoh: {missing[:5]}"
    n_tr = int((split == "train").sum())
    print(f"subset {len(files)} citra ({n_tr} train + {len(files)-n_tr} test)", flush=True)

    class DS(Dataset):
        def __init__(s, tf, harm):
            s.tf, s.harm = tf, harm

        def __len__(s):
            return len(files)

        def __getitem__(s, i):
            im = Image.open(files[i]).convert("RGB")
            if s.harm:
                im = harmonize(im, jpeg=jpeg)
            return s.tf(im), i

    def embed(m, tf, feat_fn, bs, harm, tag):
        dl = DataLoader(DS(tf, harm), batch_size=bs, num_workers=8, pin_memory=True)
        nb = len(dl)
        step = max(1, nb // 6)
        E = None
        with torch.no_grad():
            for bi, (xb, idx) in enumerate(dl):
                xb = xb.to(dev)
                with torch.autocast("cuda"):
                    emb = feat_fn(m, xb).float()
                emb = torch.nn.functional.normalize(emb, dim=1).cpu().numpy()
                if E is None:
                    E = np.zeros((len(files), emb.shape[1]), np.float32)
                E[idx.numpy()] = emb
                if bi % step == 0 or bi == nb - 1:
                    print(f"    {tag}: batch {bi+1}/{nb}", flush=True)
        return E.astype(np.float16)

    out = {"key": np.array(keys), "split": split, "harm_px": np.array(HARM_PX),
           "harm_jpeg": np.array(jpeg)}
    try:
        prev = np.load(OUT, allow_pickle=True)
        if len(prev["key"]) == len(keys):
            done = {k: prev[k] for k in prev.files if k not in out}
            out.update(done)
            print(f"resume: sudah ada {sorted(done)}", flush=True)
    except FileNotFoundError:
        pass

    def flush():
        np.savez(OUT, **out)
        vol.commit()

    def load_radio(hf_repo, img):
        from transformers import AutoModel
        import torchvision.transforms as T
        m = AutoModel.from_pretrained(hf_repo, trust_remote_code=True).to(dev).eval()
        tf = T.Compose([T.Resize((img, img), interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor()])
        return m, tf, (lambda m, xb: m(xb)[0])

    def load_openclip(hf_id, img):
        import open_clip
        m, preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{hf_id}")
        return m.to(dev).eval(), preprocess, (lambda m, xb: m.encode_image(xb))

    for name, bb, img in BACKBONES:
        want = [name] + ([f"{name}_ctrl"] if name == CTRL_BB else [])
        if all(w in out for w in want):
            print(f"  {name}: sudah ada (resume), skip", flush=True)
            continue
        try:
            if bb.startswith("HF:"):
                m, tf, ff = load_radio(bb[3:], img)
                bs = 64
            elif bb.startswith("OC:"):
                m, tf, ff = load_openclip(bb[3:], img)
                bs = 64
            else:
                vk = {"dynamic_img_size": True} if bb.startswith("vit") else {}
                m = timm.create_model(bb, pretrained=True, num_classes=0, **vk).to(dev).eval()
                dc = resolve_model_data_config(m)
                dc["input_size"] = (3, img, img)
                dc["crop_pct"] = 1.0
                tf = create_transform(**dc, is_training=False)
                ff = lambda m, xb: m(xb)
                bs = 128
            for w in want:
                if w in out:
                    continue
                out[w] = embed(m, tf, ff, bs, harm=not w.endswith("_ctrl"), tag=w)
                print(f"  {w}: {out[w].shape}", flush=True)
            del m
            torch.cuda.empty_cache()
            flush()
        except Exception as e:
            print(f"  {name}: SKIP ({bb}) -> {type(e).__name__}: {e}", flush=True)
            continue

    flush()
    print(f"wrote {OUT} (keys: {sorted(out)})", flush=True)


@app.local_entrypoint()
def main(jpeg: bool = True):
    run.remote(jpeg)
