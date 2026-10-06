"""Reproducible per-class image subsets and dataset splits, shared by the EAML and LayoutLMv3 data loaders.

With the same image folder, images_per_class and seed, both backbones train on exactly the same documents: the
candidates are sorted by path (file-system listing order differs between machines) and sampled with a private
random generator (independent of the global random state).
"""
import os
import random

import numpy as np

DEFAULT_SEED = 42
IMAGE_EXTENSIONS = (".tif", ".png", ".jpg", ".jpeg")


def select_per_class(paths, images_per_class=None, seed=DEFAULT_SEED, class_name=""):
    """Up to `images_per_class` of the image paths of one class, as a sorted list; all of them if it is None.
    Only image files count (IMAGE_EXTENSIONS), so other files in a class folder cannot shift the sample.
    The class name is part of the random stream, so every class gets its own reproducible sample."""
    paths = sorted(str(p) for p in paths if str(p).lower().endswith(IMAGE_EXTENSIONS))
    if images_per_class is None or images_per_class >= len(paths):
        return paths
    return sorted(random.Random(f"{seed}:{class_name}").sample(paths, images_per_class))


SPLIT_FRACTIONS = (0.7, 0.15, 0.15)   # train / val / test of datasets without split folders (Tobacco-3482)


def _rel(path):
    """'<class>/<file>' of an image path: the document's identity in split_documents()."""
    path = str(path)
    return os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)


def split_documents(root, fractions=SPLIT_FRACTIONS, seed=DEFAULT_SEED):
    """Stratified train / val / test split of a dataset stored as <root>/<class>/<image> without split folders.
    Returns {"train" | "val" | "test": set of '<class>/<file>'}. The split depends only on the files and the seed
    (default 42, not the run seed): every backbone and every run gets the same documents in each split."""
    from sklearn.model_selection import StratifiedShuffleSplit
    docs = sorted(f"{c}/{f}" for c in os.listdir(root) if os.path.isdir(os.path.join(root, c))
                  for f in os.listdir(os.path.join(root, c)) if f.lower().endswith(IMAGE_EXTENSIONS))
    labels = [d.split("/")[0] for d in docs]
    n = len(docs)
    train_idx, rest = next(StratifiedShuffleSplit(n_splits=1, test_size=1 - fractions[0], random_state=seed)
                           .split(np.zeros(n), labels))
    val_rel, test_rel = next(StratifiedShuffleSplit(n_splits=1, test_size=fractions[2] / (1 - fractions[0]),
                                                    random_state=seed + 1)
                             .split(np.zeros(len(rest)), [labels[i] for i in rest]))
    return {"train": {docs[i] for i in train_idx}, "val": {docs[rest[i]] for i in val_rel},
            "test": {docs[rest[i]] for i in test_rel}}


def split_indices(sample_paths, root, split, fractions=SPLIT_FRACTIONS, seed=DEFAULT_SEED):
    """Indices of the samples (given by their image paths) that belong to `split` of split_documents(root)."""
    keep = split_documents(root, fractions, seed)[split]
    return [i for i, p in enumerate(sample_paths) if _rel(p) in keep]
