"""Reproducible per-class image subsets, shared by the EAML and LayoutLMv3 data loaders.

With the same image folder, images_per_class and seed, both backbones train on exactly the same documents: the
candidates are sorted by path (file-system listing order differs between machines) and sampled with a private
random generator (independent of the global random state).
"""
import random

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
