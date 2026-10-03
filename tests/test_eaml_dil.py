"""EAML domain-incremental pipeline (RVL-CDIP -> Tobacco-3482): every method x both strategies
(ER replay of RVL-CDIP exemplars, EWC, BC on; last-layer training)."""
import os
import unittest

from tests.fixtures import GLOBAL_CLASSES
from tests.runner import PipelineTestCase, add_matrix_tests, out_dir, run_script

SRC = "src/domain_incremental/eaml/"
OOD = ["--ood_method", "vim", "--ood_tpr", 0.95, "--ood_max_per_class", 8]
METHODS = {
    "no_evm":      ("domain_incremental.py", [], []),
    "evm":         ("domain_incremental_evm_training.py", ["--lambda_evm", 0.1], ["EVM known acc"]),
    "evm_ood":     ("domain_incremental_evm_ood.py", OOD + ["--lambda_evm", 0.1, "--lambda_ood", 0.1], ["Domain shift"]),
    "ievm":        ("domain_incremental_ievm_training.py", ["--lambda_evm", 0.1], ["EVM known acc"]),
    "regevm":      ("domain_incremental_reg_evm_training.py", ["--lambda_evm", 0.1], ["EVM known acc"]),
    # not in Fig. 1, kept in the codebase
    "ood_only":    ("domain_incremental_ood.py", OOD, ["Domain shift"]),
    "evm_posthoc": ("domain_incremental_evm.py", [], ["EVM known acc"]),
}


def run_method(test, method, strategy):
    script, extra, logs = METHODS[method]
    out = out_dir("eaml_dil", f"{method}_{strategy}")
    ocr = os.path.join(test.data, "eaml_ocr.pt")
    run_script(test, SRC + script, [
        "--data_dir", test.data, "--domains", "all_prepdataset,Tobacco3482-jpg", "--ocr_tensor_dirs", ocr, ocr,
        "--global_classes", ",".join(GLOBAL_CLASSES), "--eaml_ckpt_path", os.path.join(test.data, "ckpt", "eaml_base_16.pt"),
        "--checkpoint_dir", out, "--batch_size", 2, "--lr", 1e-4, "--num_epochs", 1, "--patience", 1,
        "--strategy", strategy, "--use_ewc", "--use_exemplars", "--max_exemplars", 64, "--exemplar_selection", "random",
        "--use_bias_correction", "--finetune_mode", "last_layer",
    ] + extra, cwd=out, expect_files=[os.path.join(out, "best_model.pth")], expect_logs=["Replay memory"] + logs)


class TestEAMLDIL(PipelineTestCase):
    pass


add_matrix_tests(TestEAMLDIL, METHODS, run_method)

if __name__ == "__main__":
    unittest.main()
