"""Near-duplicate check between document image collections: are Tobacco-3482 (and RVL-CDIP-N) documents also in
RVL-CDIP? Tobacco-3482 and RVL-CDIP both come from the IIT-CDIP tobacco-litigation archive, so a DIL result
RVL-CDIP -> Tobacco-3482 could partly measure documents the base model has already seen.

Method:
  1. a 64-bit perceptual hash (DCT of the 32x32 grey image, low 8x8 frequencies above their median; robust to
     rescaling, JPEG/TIFF re-encoding and small contrast changes) of every image, cached per collection;
  2. candidates: pairs with Hamming distance <= --max_hamming (default 10 of 64 bits);
  3. confirmation: Pearson correlation of the two 128x128 grey thumbnails >= --min_corr (default 0.95), so that
     different documents filled in on the same form template are not counted (at most --top_k candidates per
     query image are checked, the nearest first).
The report also lists the distribution of each query image's nearest Hamming distance, to show whether duplicates
and distinct documents separate cleanly.

Outputs in --out: overlap_pairs.csv (confirmed pairs: query, match, Hamming distance, correlation, classes, splits),
overlap.md (counts per query split and class, against each RVL-CDIP split and against the base models' training
documents) and hashes_<name>.npz (the cached hashes).

Example (CPU, TinyFAT; scripts/data_prep/run_overlap_check.sh):
  python tools/data/overlap_check.py --rvl $RVL_DIR --tobacco $DATA_ROOT/$TOB_DOMAIN \
      --rvl_n $DATA_ROOT/RVL-CDIP-N --out $RUN_ROOT/overlap --workers 56
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_subset import IMAGE_EXTENSIONS, base_training_documents, split_documents  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def grey(path, size):
    im = Image.open(path)
    im.seek(0)                                   # first page of multi-page TIFFs
    return np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)


def phash(path):
    """64-bit perceptual hash as uint64, or None if the image cannot be read."""
    try:
        low = dctn(grey(path, 32), norm="ortho")[:8, :8]
    except Exception:
        return None
    bits = (low > np.median(low)).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def collect(root, splits):
    """[(relative path '<split>/<class>/<file>' or '<class>/<file>', split, class)] of a collection."""
    out = []
    for split in splits:
        base = os.path.join(root, split) if split else root
        for c in sorted(os.listdir(base)):
            d = os.path.join(base, c)
            if os.path.isdir(d):
                out += [(os.path.join(split, c, f) if split else os.path.join(c, f), split, c)
                        for f in sorted(os.listdir(d)) if f.lower().endswith(IMAGE_EXTENSIONS)]
    return out


def hashes(root, items, cache, workers):
    """uint64 hashes of the items (cached in `cache`); unreadable images are dropped."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["paths"]) == [p for p, _, _ in items]:
            print(f"cached: {cache}", flush=True)
            return z["hashes"], z["ok"]
    with Pool(workers) as pool:
        hs = pool.map(phash, [os.path.join(root, p) for p, _, _ in items], chunksize=256)
    ok = np.array([h is not None for h in hs])
    arr = np.array([h if h is not None else 0 for h in hs], dtype=np.uint64)
    np.savez(cache, paths=np.array([p for p, _, _ in items], dtype=object), hashes=arr, ok=ok)
    print(f"hashed {len(items)} images of {root} ({(~ok).sum()} unreadable) -> {cache}", flush=True)
    return arr, ok


def candidates(q, r, max_hamming, top_k=5, chunk=128):
    """(query index, reference index, distance) for the up to top_k nearest references with distance <=
    max_hamming, and each query's nearest distance."""
    pairs, nearest = [], np.empty(len(q), dtype=np.int64)
    for s in range(0, len(q), chunk):
        d = np.bitwise_count(np.bitwise_xor(q[s:s + chunk, None], r[None, :])).astype(np.int16)
        nearest[s:s + chunk] = d.min(axis=1)
        for a in np.flatnonzero(nearest[s:s + chunk] <= max_hamming):
            near = np.argsort(d[a], kind="stable")[:top_k]
            pairs += [(s + a, int(b), int(d[a, b])) for b in near if d[a, b] <= max_hamming]
    return pairs, nearest


def correlation(a, b):
    a, b = grey(a, 128).ravel(), grey(b, 128).ravel()
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def compare(name, q_root, q_items, q_hash, q_ok, r_root, r_items, r_hash, r_ok, args):
    qi_map, ri_map = np.flatnonzero(q_ok), np.flatnonzero(r_ok)
    pairs, nearest = candidates(q_hash[qi_map], r_hash[ri_map], args.max_hamming, args.top_k)
    rows = []
    for a, b, d in pairs:
        qa, rb = q_items[qi_map[a]], r_items[ri_map[b]]
        corr = correlation(os.path.join(q_root, qa[0]), os.path.join(r_root, rb[0]))
        rows.append({"comparison": name, "query": qa[0], "query_split": qa[1], "query_class": qa[2],
                     "match": rb[0], "match_split": rb[1], "match_class": rb[2], "hamming": d, "corr": corr,
                     "confirmed": corr >= args.min_corr})
    hist = np.bincount(nearest, minlength=65)
    print(f"{name}: {len(qi_map)} query images, {len(pairs)} candidates, "
          f"{sum(r['confirmed'] for r in rows)} confirmed pairs", flush=True)
    return pd.DataFrame(rows), hist


def md_table(frame):
    """A DataFrame (with a named index) as a Markdown table."""
    frame = frame.reset_index()
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)])


def report(df, hists, q_counts, base_train, args):
    lines = ["# Near-duplicate check", "",
             f"Perceptual hash, candidates with Hamming distance <= {args.max_hamming} of 64 bits, confirmed by a "
             f"128x128 thumbnail correlation >= {args.min_corr} (at most {args.top_k} candidates per image).", ""]
    for name, hist in hists.items():
        sub = df[(df.comparison == name) & df.confirmed] if len(df) else df
        n_q = q_counts[name]
        dup_q = sub["query"].nunique() if len(sub) else 0
        lines += [f"## {name}", "",
                  f"- query images: {n_q}; with at least one confirmed duplicate: **{dup_q}** "
                  f"({100 * dup_q / max(n_q, 1):.1f} %)",
                  "- nearest Hamming distance of the query images: " + ", ".join(
                      f"{d}: {c}" for d, c in enumerate(hist) if c and d <= 20) +
                  (f", >20: {int(hist[21:].sum())}" if hist[21:].sum() else "")]
        if len(sub):
            sub = sub.assign(base_train=sub["match"].isin(base_train))
            by = (sub.groupby(["query_split", "match_split"])["query"].nunique().unstack(fill_value=0))
            lines += ["", "Query documents with a duplicate, by query split (rows) and matched RVL-CDIP split "
                      "(columns):", "", md_table(by), "",
                      f"- of which matched to a base model's training document: "
                      f"**{sub[sub.base_train]['query'].nunique()}**",
                      f"- same class label in both collections: {(sub.query_class == sub.match_class).mean() * 100:.1f} % "
                      "of the pairs", "",
                      "By query class:", "",
                      md_table(sub.groupby("query_class")["query"].nunique().rename("documents").to_frame())]
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rvl", required=True, help="RVL-CDIP with train/ val/ test/ class folders")
    ap.add_argument("--tobacco", required=True, help="Tobacco-3482 class folders")
    ap.add_argument("--rvl_n", help="RVL-CDIP-N class folders (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--max_hamming", type=int, default=10)
    ap.add_argument("--min_corr", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rvl = collect(args.rvl, ["train", "val", "test"])
    rvl_h, rvl_ok = hashes(args.rvl, rvl, os.path.join(args.out, "hashes_rvl.npz"), args.workers)
    base_train = set()
    for c in sorted({c for _, s, c in rvl if s == "train"}):
        base_train.update(os.path.relpath(p, args.rvl) for p in base_training_documents(
            [os.path.join(args.rvl, p) for p, s, k in rvl if s == "train" and k == c], c))

    queries = {"Tobacco-3482 vs RVL-CDIP": ("tobacco", args.tobacco)}
    if args.rvl_n:
        queries["RVL-CDIP-N vs RVL-CDIP"] = ("rvl_n", args.rvl_n)
    frames, hists, counts = [], {}, {}
    for name, (key, root) in queries.items():
        split = split_documents(root)                          # the fixed 70/15/15 split used by the DIL runs
        items = [(p, next((s for s in ("train", "val", "test") if p in split[s]), ""), c)
                 for p, _, c in collect(root, [""])]
        h, ok = hashes(root, items, os.path.join(args.out, f"hashes_{key}.npz"), args.workers)
        df, hist = compare(name, root, items, h, ok, args.rvl, rvl, rvl_h, rvl_ok, args)
        frames.append(df)
        hists[name] = hist
        counts[name] = int(ok.sum())
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(os.path.join(args.out, "overlap_pairs.csv"), index=False)
    with open(os.path.join(args.out, "overlap.md"), "w") as f:
        f.write(report(df, hists, counts, base_train, args))
    print(f"-> {args.out}/overlap.md")


if __name__ == "__main__":
    main()
