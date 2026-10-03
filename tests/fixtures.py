"""Synthetic data for the end-to-end pipeline tests.

Builds (once per directory) a tiny dataset with the real class names and the directory layouts the scripts expect:

    <root>/all_prepdataset/{train,val,test}/<16 RVL-CDIP classes>/*.jpg     (CIL + DIL base domain)
    <root>/Tobacco3482-jpg/<10 Tobacco classes>/*.jpg                       (DIL incremental domain)
    <root>/eaml_ocr.pt            EAML OCR: {image path: {input_ids, attention_mask}}  (bert-base-uncased ids)
    <root>/bert_ocr_rvl/, bert_ocr_tobacco/   LayoutLMv3 OCR: one <stem>.pt per image (ids + boxes)
    <root>/class_mapping.json
    <root>/ckpt/{eaml,llmv3}_base_{3,16}.pt   randomly initialised base checkpoints ({"model_state_dict": ...})

Images carry a class-dependent colour so the tiny runs have something to learn; nothing here measures accuracy.
"""
import json
import os

import numpy as np
import torch
from PIL import Image

RVL_CLASSES = ['letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific_report',
               'scientific_publication', 'specification', 'file_folder', 'news_article', 'budget', 'invoice',
               'presentation', 'questionnaire', 'resume', 'memo']
TOBACCO_CLASSES = ['letter', 'form', 'email', 'advertisement', 'scientific_report', 'news_article', 'resume', 'memo',
                   'Note', 'Report']
GLOBAL_CLASSES = RVL_CLASSES + ['Note', 'Report']

# Class-incremental setting used by the tests: 3 base classes, 1 new class, 1 later class (unknown in open-set eval)
CIL_BASE = ['letter', 'form', 'email']
CIL_NEW = 'memo'
CIL_ALL = CIL_BASE + [CIL_NEW, 'budget']

RVL_PER_SPLIT = {'train': 3, 'val': 2, 'test': 2}
TOBACCO_PER_CLASS = 8          # stratified 70/15/15 split needs >= 1 sample per class in val and test
EAML_TOKENS, BERT_LEN = 128, 512


def _image(path, cls_idx, i):
    rng = np.random.default_rng(cls_idx * 1000 + i)
    base = np.array([(37 * cls_idx) % 256, (91 * cls_idx) % 256, (53 * cls_idx) % 256], dtype=np.float32)
    img = np.clip(base + rng.normal(0, 25, (96, 72, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=90)


def _bert_ids(rng, length, n_real, cls_idx):
    ids = np.zeros(length, dtype=np.int64)
    ids[0], ids[n_real - 1] = 101, 102                        # [CLS] ... [SEP]
    ids[1:n_real - 1] = rng.integers(2000, 2400, n_real - 2) + 300 * cls_idx
    mask = (np.arange(length) < n_real).astype(np.int64)
    return torch.tensor(ids), torch.tensor(mask)


def build_fixture(root):
    """Create the synthetic dataset and checkpoints under `root` (skipped if already complete)."""
    marker = os.path.join(root, ".complete")
    if os.path.exists(marker):
        return root
    os.makedirs(root, exist_ok=True)
    eaml_ocr = {}
    rng = np.random.default_rng(0)

    def add(img_path, cls_idx, bert_dir):
        _image(img_path, cls_idx, len(eaml_ocr))
        ids, mask = _bert_ids(rng, EAML_TOKENS, 40, cls_idx)
        eaml_ocr[img_path] = {"input_ids": ids, "attention_mask": mask}
        ids, mask = _bert_ids(rng, BERT_LEN, 60, cls_idx)
        n = int(mask.sum())
        bbox = torch.zeros(BERT_LEN, 4, dtype=torch.long)
        bbox[1:n - 1] = torch.tensor([[50, 20 * j % 900, 400, 20 * j % 900 + 15] for j in range(1, n - 1)])
        bbox[n - 1] = torch.tensor([1000, 1000, 1000, 1000])
        stem = os.path.splitext(os.path.basename(img_path))[0]
        torch.save({"input_ids": ids, "attention_mask": mask, "bbox": bbox, "image_path": img_path},
                   os.path.join(bert_dir, stem + ".pt"))

    rvl_bert, tob_bert = os.path.join(root, "bert_ocr_rvl"), os.path.join(root, "bert_ocr_tobacco")
    os.makedirs(rvl_bert, exist_ok=True)
    os.makedirs(tob_bert, exist_ok=True)
    for split, n in RVL_PER_SPLIT.items():
        for c, cls in enumerate(RVL_CLASSES):
            d = os.path.join(root, "all_prepdataset", split, cls)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                add(os.path.join(d, f"rvl_{split}_{cls}_{i}.jpg"), c, rvl_bert)
    for cls in TOBACCO_CLASSES:
        d = os.path.join(root, "Tobacco3482-jpg", cls)
        os.makedirs(d, exist_ok=True)
        for i in range(TOBACCO_PER_CLASS):
            add(os.path.join(d, f"tob_{cls}_{i}.jpg"), GLOBAL_CLASSES.index(cls), tob_bert)
    torch.save(eaml_ocr, os.path.join(root, "eaml_ocr.pt"))
    with open(os.path.join(root, "class_mapping.json"), "w") as f:
        json.dump({str(i): c for i, c in enumerate(RVL_CLASSES)}, f, indent=1)

    ckpt = os.path.join(root, "ckpt")
    os.makedirs(ckpt, exist_ok=True)
    from utils.eaml.eaml_model import EAMLModel
    from utils.llmv3.llmv3_model_loader import LayoutLMv3
    torch.manual_seed(0)
    for n in (len(CIL_BASE), len(RVL_CLASSES)):
        torch.save({"model_state_dict": EAMLModel(num_classes=n).state_dict()}, os.path.join(ckpt, f"eaml_base_{n}.pt"))
        torch.save({"model_state_dict": LayoutLMv3(num_labels=n).state_dict()}, os.path.join(ckpt, f"llmv3_base_{n}.pt"))
    open(marker, "w").close()
    return root
