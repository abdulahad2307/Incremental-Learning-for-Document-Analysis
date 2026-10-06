"""LayoutLMv3 domain-incremental pipeline (RVL-CDIP -> Tobacco-3482): every method x both strategies
(ER, EWC, BC on; last-layer training)."""
import glob
import os
import unittest

import torch

from tests.fixtures import GLOBAL_CLASSES
from tests.runner import PipelineTestCase, add_matrix_tests, check_predictions, out_dir, run_script

SRC = "src/domain_incremental/llmv3/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OOD = ["--ood_method", "vim", "--ood_tpr", 0.95, "--ood_max_per_class", 8, "--lambda_ood", 0.1]
METHODS = {
    "no_evm":  ("llmv3_domain_incremental.py", [], []),
    "evm":     ("llmv3_domain_incremental_evm_training.py", ["--lambda_evm", 0.1], ["EVM open set test"]),
    "evm_ood": ("llmv3_domain_incremental_evm_ood.py", ["--lambda_evm", 0.1] + OOD, ["EVM open set test", "Domain shift"]),
    "ievm":    ("llmv3_domain_incremental_ievm_training.py", ["--lambda_evm", 0.1], ["EVM open set test"]),
    "regevm":  ("llmv3_domain_incremental_reg_evm_training.py", ["--lambda_evm", 0.1], ["EVM open set test"]),
}

# method -> (EVM probabilities, OOD scores) expected in its saved predictions
SAVES = {"no_evm": (False, False), "evm": (True, False), "evm_ood": (True, True), "ievm": (True, False),
         "regevm": (True, False), "ood_only": (False, True), "evm_posthoc": (True, False)}


def run_method(test, method, strategy):
    script, extra, logs = METHODS[method]
    out = out_dir("layout_dil", f"{method}_{strategy}")
    run_script(test, SRC + script, [
        "--data_dir", test.data, "--ocr_tensor_path_base", os.path.join(test.data, "bert_ocr_rvl"),
        "--ocr_tensor_path_inc", os.path.join(test.data, "bert_ocr_tobacco"), "--all_classes", ",".join(GLOBAL_CLASSES),
        "--dataset_base", "rvl_cdip", "--dataset_inc", "tobacco3482",
        "--base_model_path", os.path.join(test.data, "ckpt", "llmv3_base_16.pt"), "--checkpoint_dir", out,
        "--batch_size", 2, "--lr", 1e-4, "--num_epochs", 1, "--patience", 1, "--images_per_class", 3,
        "--strategy", strategy, "--use_ewc", "--max_exemplars", 2, "--exemplar_selection", "random",
        "--use_bias_correction", "--training_mode", "last_layer", "--full_model_acc", 0.9, "--device", DEVICE,
    ] + extra, cwd=out, expect_logs=logs)
    test.assertTrue(glob.glob(os.path.join(out, "layoutlmv3_domain_incremental*_best.pt")), f"no best checkpoint in {out}")
    check_predictions(test, out, "dil", ["rvl", "tobacco"], *SAVES[method])


class TestLayoutDIL(PipelineTestCase):
    pass


add_matrix_tests(TestLayoutDIL, METHODS, run_method)

if __name__ == "__main__":
    unittest.main()
