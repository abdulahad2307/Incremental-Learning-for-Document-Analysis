"""Step-0 predictions of the base models (tools/eval/base_predictions.py), for both backbones and settings."""
import glob
import os
import unittest

from tests.fixtures import CIL_ALL, CIL_BASE, GLOBAL_CLASSES, RVL_CLASSES
from tests.runner import PipelineTestCase, out_dir, run_script

TOOL = "tools/eval/base_predictions.py"


class TestBasePredictions(PipelineTestCase):
    results = {}

    def run_tool(self, backbone, setting, ckpt, base_classes, all_classes, ocr):
        out = out_dir("base_predictions", f"{backbone}_{setting}")
        run_script(self, TOOL, ["--backbone", backbone, "--setting", setting,
                                "--checkpoint", os.path.join(self.data, "ckpt", ckpt),
                                "--base_classes", ",".join(base_classes), "--all_classes", ",".join(all_classes),
                                "--data_root", self.data, "--ocr", *[os.path.join(self.data, o) for o in ocr],
                                "--out_dir", out, "--batch_size", 2], cwd=out)
        from utils.eval.predictions import load_predictions
        files = glob.glob(os.path.join(out, "predictions", f"base_{setting.lower()}_seed42.pt"))
        self.assertEqual(len(files), 1, f"no base_{setting.lower()}_seed42.pt in {out}/predictions")
        r = load_predictions(files[0])
        TestBasePredictions.results[(backbone, setting)] = r
        self.assertEqual(r["meta"]["step"], 0)
        self.assertEqual(r["meta"]["method"], "base")
        self.assertEqual(r["output_classes"], list(base_classes))
        expected = ["seen", "unseen"] if setting == "CIL" else ["rvl", "tobacco"]
        self.assertEqual(sorted(r["sets"]), sorted(expected))
        for name, s in r["sets"].items():
            self.assertGreater(len(s["y_true"]), 0, f"set {name} is empty")
            self.assertTrue(all("/" in d for d in s["doc_id"]), f"{name}: doc ids are not '<class>/<file>' paths")
            self.assertEqual(s["logits"].shape, (len(s["y_true"]), len(base_classes)))
        if setting == "CIL":  # the unseen set holds exactly the classes the base model has not learned
            unseen = r["sets"]["unseen"]
            self.assertEqual(sorted({unseen["class_names"][y] for y in unseen["y_true"]}),
                             sorted(c for c in all_classes if c not in base_classes))

    def test_eaml_cil(self):
        self.run_tool("eaml", "CIL", "eaml_base_3.pt", CIL_BASE, CIL_ALL, ["eaml_ocr.pt"])

    def test_eaml_dil(self):
        self.run_tool("eaml", "DIL", "eaml_base_16.pt", RVL_CLASSES, GLOBAL_CLASSES, ["eaml_ocr.pt", "eaml_ocr.pt"])

    def test_layoutlmv3_cil(self):
        self.run_tool("llmv3", "CIL", "llmv3_base_3.pt", CIL_BASE, CIL_ALL, ["bert_ocr_rvl"])

    def test_layoutlmv3_dil(self):
        self.run_tool("llmv3", "DIL", "llmv3_base_16.pt", RVL_CLASSES, GLOBAL_CLASSES, ["bert_ocr_rvl", "bert_ocr_tobacco"])

    def test_tobacco_split_shared(self):
        """Both backbones evaluate DIL on the same Tobacco-3482 test documents (utils/data_subset.split_documents)."""
        if ("eaml", "DIL") not in self.results:
            self.test_eaml_dil()
        if ("llmv3", "DIL") not in self.results:
            self.test_layoutlmv3_dil()
        eaml = sorted(self.results[("eaml", "DIL")]["sets"]["tobacco"]["doc_id"])
        llmv3 = sorted(self.results[("llmv3", "DIL")]["sets"]["tobacco"]["doc_id"])
        self.assertEqual(eaml, llmv3)


if __name__ == "__main__":
    unittest.main()
