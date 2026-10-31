#!/usr/bin/env python3
"""
DHARA Analytics Dashboard — export_results.py
Author: Dr. Himanshu Sharma

Turns the DHARA "Saved Resources" folders into the small JSON files the
dashboard reads, so the repository never has to carry model states or the
multi-megabyte prediction dumps.

Expected layout (the same as the Google Drive folders):

    <src>/
      PASCAL 1K/Word Threshold-5/Batch-32/
          TrainingLogs.xlsx            per-epoch and per-batch training loss
          PredictionSplit.xlsx         per-epoch validation metrics
          GeneratedPredictions.txt     validation captions of every epoch   (optional)
          TestResults/TestResults_Batch_0_Epoch_<E>.xlsx
          TestResults/TestCaptions_Batch_0_Epoch_<E>.txt                    (optional)
          CustomizedPretrainedEmbeddings/FastText_Hindi_300_vocab<V>_*.npy  (only the name is read)
          PASCAL_Hindi/Preprocessed_Labeled_Dataset.txt                    (optional, reference captions)
      Flickr 8K/Word Threshold-1 … Word Threshold-5/Batch-32 … Batch-256/   (same files)

Usage:
    python export_results.py --src "path/to/DHARA Saved Resources" --out Data
    python export_results.py --src ... --human HumanEvaluation/DHARA_HumanVerifiedResults.xlsx

Needs Python 3.8+, pandas and openpyxl.
"""
import argparse, hashlib, json, math, pickle, re, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas openpyxl")

DATASETS = {  # folder name -> (key, label)
    "pascal 1k": ("pascal1k", "PASCAL 1K-Hindi"),
    "flickr 8k": ("flickr8k", "Flickr 8K-Hindi"),
}
PAPER_THRESHOLD = 5          # word-frequency threshold used for every result in the paper
EVOLUTION_IMAGES = 12        # validation images whose captions are followed across epochs
MAX_BATCH_POINTS = 240       # per-epoch batch-loss curve is bucket-averaged to at most this many points
METRICS = [("BLEU-1", "B1"), ("BLEU-2", "B2"), ("BLEU-3", "B3"), ("BLEU-4", "B4"), ("METEOR", "M"), ("CIDEr", "C")]


# ───────────────────────── helpers ─────────────────────────
def num(v, d=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, d)


def read_sheet(path):
    try:
        return pd.read_excel(path)
    except Exception as e:  # noqa: BLE001
        print(f"      ! cannot read {path.name}: {e}")
        return None


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None


def metric_row(r):
    """Logged CIDEr is a fraction (0.159); the paper reports it ×100 (15.9)."""
    out = {}
    for col, k in METRICS:
        v = num(r.get(col))
        if v is not None and k == "C":
            v = round(v * 100, 4)
        out[k] = v
    rouge = num(r.get("ROUGE"))
    out["R"] = rouge if rouge else None  # the pipeline logged ROUGE as 0 → treated as missing
    return out


TOKEN_STRIP = re.compile(r"[।\.\,\!\?\"'“”‘’\(\)\[\]]")


def tokens(text):
    return [t for t in TOKEN_STRIP.sub(" ", str(text or "")).split() if t]


def caption_stats(caps):
    """caps: list of caption strings."""
    toks = [tokens(c) for c in caps]
    n = len(toks)
    if not n:
        return None
    lens = [len(t) for t in toks]
    uni = Counter(w for t in toks for w in t)
    bi = Counter((a, b) for t in toks for a, b in zip(t, t[1:]))
    tot_uni = sum(uni.values()) or 1
    tot_bi = sum(bi.values()) or 1
    rep = sum(1 for t in toks if any(a == b for a, b in zip(t, t[1:])))
    # phrase repetition: some word pair occurs twice in one caption ("एक इमारत के सामने एक इमारत के सामने")
    loop = sum(1 for t in toks if len(t) > 3 and max(Counter(zip(t, t[1:])).values()) >= 2)
    hist = Counter(min(l, 40) for l in lens)
    return {
        "n": n,
        "mean_len": round(sum(lens) / n, 3),
        "distinct1": round(len(uni) / tot_uni * 100, 3),
        "distinct2": round(len(bi) / tot_bi * 100, 3),
        "unique_pct": round(len(set(caps)) / n * 100, 3),
        "repeat_pct": round(rep / n * 100, 3),
        "loop_pct": round(loop / n * 100, 3),
        "vocab_used": len(uni),
        "len_hist": {str(k): v for k, v in sorted(hist.items())},
        "top_words": uni.most_common(30),
    }


def sentence_bleu(hyp, refs, n=4):
    """Smoothed (add-one on higher orders) sentence BLEU, for ranking captions only."""
    h = tokens(hyp)
    R = [tokens(r) for r in refs]
    if not h or not R:
        return None
    logp = 0.0
    for k in range(1, n + 1):
        hc = Counter(tuple(h[i:i + k]) for i in range(len(h) - k + 1))
        if not hc:
            logp += math.log(1 / (len(h) + 1))
            continue
        mx = Counter()
        for r in R:
            for g, c in Counter(tuple(r[i:i + k]) for i in range(len(r) - k + 1)).items():
                mx[g] = max(mx[g], c)
        clip = sum(min(c, mx[g]) for g, c in hc.items())
        tot = sum(hc.values())
        p = clip / tot if k == 1 else (clip + 1) / (tot + 1)
        if p == 0:
            return 0.0
        logp += math.log(p)
    rl = min((abs(len(r) - len(h)), len(r)) for r in R)[1]
    bp = 1.0 if len(h) > rl else math.exp(1 - rl / max(len(h), 1))
    return round(bp * math.exp(logp / n) * 100, 2)


class _Stub:
    def __setstate__(self, s):
        if isinstance(s, dict):
            self.__dict__.update(s)


class _Inert(_Stub):
    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __init__(self, *a, **k):
        pass


class _SafeUnpickler(pickle.Unpickler):
    """Reads the Dataset pickle without importing TensorFlow or running any code:
    every class is replaced by an inert stub."""
    SAFE = {("collections", "OrderedDict"), ("builtins", "set"), ("builtins", "frozenset")}

    def find_class(self, module, name):
        if (module, name) in self.SAFE:
            return getattr(__import__(module), name)
        return _Inert


def load_references(batch_dir):
    for f in batch_dir.glob("*_Hindi/Preprocessed_Labeled_Dataset.txt"):
        try:
            with open(f, "rb") as fh:
                obj = _SafeUnpickler(fh).load()
            caps = getattr(obj, "all_image_captions", None)
            if isinstance(caps, dict):
                return caps, {
                    "avg_len": getattr(obj, "average_caption_length", None),
                    "max_len": getattr(obj, "max_caption_length", None),
                    "n_train": len(getattr(obj, "training_image_names", []) or []),
                    "n_val": len(getattr(obj, "validation_image_names", []) or []),
                    "n_test": len(getattr(obj, "testing_image_names", []) or []),
                }
        except Exception as e:  # noqa: BLE001
            print(f"      ! references not readable ({f.name}): {e}")
    return None, None


def parse_predictions(path):
    """GeneratedPredictions.txt → {epoch: {image: caption}}"""
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"-{20,}\s*Epoch Number\s+(\d+)\s*-{20,}", text)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        try:
            d, _ = json.JSONDecoder().raw_decode(parts[i + 1].strip().lstrip("-").strip())
            out[int(parts[i])] = {k: (v[0] if isinstance(v, list) and v else v) for k, v in d.items()}
        except Exception:  # noqa: BLE001
            continue
    return out


def bucket(vals, k):
    if len(vals) <= k:
        return [num(v, 4) for v in vals]
    step = len(vals) / k
    return [num(sum(vals[int(i * step):int((i + 1) * step)]) / max(1, int((i + 1) * step) - int(i * step)), 4) for i in range(k)]


# ───────────────────────── one run ─────────────────────────
def export_run(bdir, ds_key, ds_label, wt, bs, dup_of):
    run_id = f"{ds_key}-wt{wt}-b{bs}"
    out = {"id": run_id, "dataset": ds_key, "dataset_label": ds_label, "word_threshold": wt, "batch_size": bs,
           "paper_config": wt == PAPER_THRESHOLD, "exported": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "duplicate_logs_of": dup_of}
    av = {}

    # vocabulary size (from the cached embedding file name)
    vf = next(iter(bdir.glob("CustomizedPretrainedEmbeddings/*vocab*")), None)
    m = re.search(r"vocab(\d+)", vf.name) if vf else None
    out["vocab_size"] = int(m.group(1)) if m else None

    # training
    tl = bdir / "TrainingLogs.xlsx"
    if tl.exists() and not dup_of:
        df = read_sheet(tl)
        if df is not None and "Epoch Loss" in df.columns:
            bcols = [c for c in df.columns if str(c).startswith("Batch ")]
            rows, curves = [], []
            for _, r in df.iterrows():
                v = [float(r[c]) for c in bcols if num(r[c]) is not None]
                if not v:
                    continue
                mu = sum(v) / len(v)
                sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1))
                rows.append({"epoch": int(r["Epoch"]), "loss_sum": num(r["Epoch Loss"]), "mean": num(mu), "std": num(sd),
                             "min": num(min(v)), "max": num(max(v)), "first": num(v[0]), "last": num(v[-1])})
                curves.append(bucket(v, MAX_BATCH_POINTS))
            out["train"] = {"n_batches": len(bcols), "epochs": rows, "batch_curves": curves,
                            "batch_curve_bucketed": len(bcols) > MAX_BATCH_POINTS}
    av["train"] = "train" in out

    # validation
    ps = bdir / "PredictionSplit.xlsx"
    if ps.exists() and not dup_of:
        df = read_sheet(ps)
        if df is not None:
            ecol = "Epoch Number" if "Epoch Number" in df.columns else "Epoch"
            out["val"] = [{"epoch": int(r[ecol]), **metric_row(r)} for _, r in df.iterrows()]
    av["val"] = bool(out.get("val"))

    # test (the pipeline writes one file per tested checkpoint; the most recent is used)
    tests = sorted((bdir / "TestResults").glob("TestResults_*Epoch_*.xlsx"),
                   key=lambda p: p.stat().st_mtime)
    if not tests and (bdir / "TestResults.xlsx").exists():
        tests = [bdir / "TestResults.xlsx"]
    tests_out = []
    for t in tests:
        df = read_sheet(t)
        if df is None or df.empty:
            continue
        r = df.iloc[0]
        tests_out.append({"epoch": int(r["Epoch"]), "tf_loss": num(r.get("Teacher Forced Loss")), **metric_row(r)})
    if tests_out:
        out["test"] = tests_out[-1]
        out["test_all"] = tests_out
    av["test"] = bool(tests_out)

    # references (optional)
    refs, dmeta = load_references(bdir)
    if dmeta:
        out["dataset_meta"] = dmeta
    av["references"] = bool(refs)

    # test captions
    tc = None
    if out.get("test"):
        ep = out["test"]["epoch"]
        cand = list((bdir / "TestResults").glob(f"TestCaptions_*Epoch_{ep}.txt")) or list((bdir / "TestResults").glob("TestCaptions_*.txt"))
        if not cand and (bdir / "TestCaptions.txt").exists():
            cand = [bdir / "TestCaptions.txt"]
        if cand:
            try:
                raw = json.loads(cand[0].read_text(encoding="utf-8"))
                tc = {k: (v[0] if isinstance(v, list) and v else v) for k, v in raw.items()}
            except Exception as e:  # noqa: BLE001
                print(f"      ! test captions not readable: {e}")
    if tc:
        items = []
        for img, cap in tc.items():
            it = {"img": img, "cap": cap}
            if refs and img in refs:
                it["refs"] = refs[img]
                it["bleu4"] = sentence_bleu(cap, refs[img], 4)
                it["bleu1"] = sentence_bleu(cap, refs[img], 1)
            items.append(it)
        out["test_captions"] = {"stats": caption_stats(list(tc.values())), "items": items}
        if refs:
            ref_caps = [c for img in tc for c in refs.get(img, [])]
            out["reference_stats"] = caption_stats(ref_caps)
    av["captions"] = bool(tc)

    # validation captions across epochs
    gp = bdir / "GeneratedPredictions.txt"
    if gp.exists() and not dup_of:
        preds = parse_predictions(gp)
        if preds:
            eps = sorted(preds)
            out["val_caption_stats"] = [{"epoch": e, **{k: v for k, v in (caption_stats(list(preds[e].values())) or {}).items()
                                                         if k not in ("len_hist", "top_words")}} for e in eps]
            imgs = sorted(set.intersection(*[set(preds[e]) for e in eps]))
            step = max(1, len(imgs) // EVOLUTION_IMAGES)
            pick = imgs[::step][:EVOLUTION_IMAGES]
            out["evolution"] = {"epochs": eps, "images": pick,
                                "captions": {i: [preds[e].get(i) for e in eps] for i in pick},
                                "refs": {i: refs[i] for i in pick if refs and i in refs}}
    av["evolution"] = "evolution" in out

    out["availability"] = av
    return out


# ───────────────────────── human evaluation ─────────────────────────
def export_human(xlsx):
    """Aggregates only: rater names and roll numbers never leave the spreadsheet."""
    out = {}
    sheets = pd.read_excel(xlsx, sheet_name=None)
    for name, df in sheets.items():
        key = "pascal1k" if "pascal" in name.lower() else "flickr8k" if "flickr" in name.lower() else name.lower()
        df = df.dropna(subset=["Fluency", "Adequacy"])
        f = df["Fluency"].astype(float)
        a = df["Adequacy"].astype(float)
        joint = Counter(zip(f.astype(int), a.astype(int)))
        per_img = df.groupby("ImageID")[["Fluency", "Adequacy"]].mean()
        rater_col = next((c for c in ("RaterRoll", "RaterName") if c in df.columns), None)
        out[key] = {
            "n_responses": int(len(df)),
            "n_images": int(df["ImageID"].nunique()),
            "n_raters": int(df[rater_col].nunique()) if rater_col else None,
            "fluency": {"mean": num(f.mean(), 3), "std": num(f.std(), 3), "hist": {str(i): int((f == i).sum()) for i in range(1, 6)}},
            "adequacy": {"mean": num(a.mean(), 3), "std": num(a.std(), 3), "hist": {str(i): int((a == i).sum()) for i in range(1, 6)}},
            "joint": [[int(x), int(y), int(c)] for (x, y), c in sorted(joint.items())],
            "per_image": [[num(r.Fluency, 3), num(r.Adequacy, 3)] for _, r in per_img.iterrows()],
            "corr": num(f.corr(a), 3) if len(df) > 2 else None,
            "examples": [{"img": r.ImageID, "cap": r.Caption, "F": num(r.Fluency, 2), "A": num(r.Adequacy, 2)}
                         for r in df.groupby(["ImageID", "Caption"])[["Fluency", "Adequacy"]].mean().reset_index().itertuples()],
        }
    return out


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--src", default=str(here / "SavedResources"), help="folder holding 'PASCAL 1K' and 'Flickr 8K'")
    ap.add_argument("--out", default=str(here / "Data"))
    ap.add_argument("--human", default=str(here / "HumanEvaluation" / "DHARA_HumanVerifiedResults.xlsx"))
    ap.add_argument("--only", nargs="*", help="only these dataset folders, e.g. --only 'PASCAL 1K'")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.out)
    (dst / "runs").mkdir(parents=True, exist_ok=True)
    index = []
    for ds_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        key = DATASETS.get(ds_dir.name.lower().strip())
        if not key or (a.only and ds_dir.name not in a.only):
            continue
        ds_key, ds_label = key
        for wt_dir in sorted(ds_dir.glob("Word Threshold-*")):
            wt = int(re.sub(r"\D", "", wt_dir.name) or 0)
            bdirs = sorted([p for p in wt_dir.iterdir() if p.is_dir() and re.match(r"Batch[\s\-_]*\d+", p.name)],
                           key=lambda p: int(re.sub(r"\D", "", p.name)))
            hashes = {}
            for bdir in bdirs:
                bs = int(re.sub(r"\D", "", bdir.name))
                h = md5(bdir / "TrainingLogs.xlsx")
                dup = hashes.get(h) if h else None
                if h and not dup:
                    hashes[h] = f"{ds_key}-wt{wt}-b{bs}"
                print(f"  {ds_label} · threshold {wt} · batch {bs}" + (f"  (training logs identical to {dup} — skipped)" if dup else ""))
                run = export_run(bdir, ds_key, ds_label, wt, bs, dup)
                p = dst / "runs" / f"{run['id']}.json"
                p.write_text(json.dumps(run, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
                print(f"      → {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
                tv = run.get("val") or []
                bv = max(tv, key=lambda r: r["B4"] or -1) if tv else None
                index.append({k: run.get(k) for k in ("id", "dataset", "dataset_label", "word_threshold", "batch_size",
                                                       "vocab_size", "paper_config", "exported", "availability", "duplicate_logs_of")}
                             | {"test": run.get("test"), "best_val": bv,
                                "n_epochs": len((run.get("train") or {}).get("epochs", [])) or len(tv),
                                "n_batches": (run.get("train") or {}).get("n_batches"),
                                "loss": [r["mean"] for r in (run.get("train") or {}).get("epochs", [])],
                                "val": tv})
    if not index:
        sys.exit(f"No runs found under {src}")
    (dst / "runs.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  → runs.json ({len(index)} runs)")

    hx = Path(a.human)
    if hx.exists():
        (dst / "human_eval.json").write_text(json.dumps(export_human(hx), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print("  → human_eval.json")
    else:
        print(f"  (no human-evaluation sheet at {hx})")


if __name__ == "__main__":
    main()
