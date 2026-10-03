"""Base-model training pipelines (the starting points of CIL / DIL): CNN baseline, EAML, LayoutLMv3."""
import os
import unittest

import torch

from tests.fixtures import CIL_BASE
from tests.runner import PipelineTestCase, out_dir, run_script

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestBaseline(PipelineTestCase):
    def test_cnn_baseline(self):
        out = out_dir("baseline", "cnn")
        run_script(self, "src/base_models/baseline_model.py", [
            "--data_dir", os.path.join(self.data, "all_prepdataset"), "--classes", *CIL_BASE,
            "--batch_size", 4, "--epochs", 1, "--device", DEVICE, "--output_dir", out,
        ], cwd=out, expect_logs=["Test Accuracy"])

    def test_eaml_base(self):
        out = out_dir("baseline", "eaml")
        run_script(self, "src/base_models/sota_eaml_model.py", [
            "--data_dir", os.path.join(self.data, "all_prepdataset"),
            "--ocr_data_path", os.path.join(self.data, "eaml_ocr.pt"),
            "--class_mapping_path", os.path.join(self.data, "class_mapping.json"),
            "--classes", ",".join(CIL_BASE), "--output_dir", out,
            "--batch_size", 2, "--num_epochs", 1, "--device", DEVICE,
        ], cwd=out, expect_files=[os.path.join(out, "eaml_best_model.pt")])

    def test_layoutlmv3_base(self):
        out = out_dir("baseline", "layoutlmv3")
        run_script(self, "src/base_models/sota_llmv3_model.py", [
            "--dataset", "rvl_cdip", "--ocr_tensor_file", os.path.join(self.data, "bert_ocr_rvl"),
            "--image_dir", os.path.join(self.data, "all_prepdataset"),  # the script appends /train
            "--base_classes", ",".join(CIL_BASE), "--bbox_style", "rect",
            "--batch_size", 2, "--epochs", 1, "--save_dir", out, "--device", DEVICE,
        ], cwd=out, expect_files=[os.path.join(out, "layoutlmv3_rvl_cdip_best.pt")])


if __name__ == "__main__":
    unittest.main()
