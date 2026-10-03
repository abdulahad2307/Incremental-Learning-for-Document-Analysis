"""LayoutLMv3 class-incremental pipeline: every method x both strategies (ER, EWC, BC, balanced sampling on;
last-layer training), plus EVM/ViM persistence across two jobs and the joint-training baseline."""
import glob
import os
import unittest

import torch

from tests.fixtures import CIL_ALL, CIL_BASE, CIL_NEW
from tests.runner import PipelineTestCase, add_matrix_tests, out_dir, run_script

SRC = "src/class_incremental/llmv3/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OOD = ["--ood_method", "vim", "--ood_tpr", 0.95, "--ood_max_per_class", 8, "--lambda_ood", 0.1]
METHODS = {
    "no_evm":  ("llmv3_class_incremental.py", [], []),
    "evm":     ("llmv3_class_incremental_evm_training.py", ["--lambda_evm", 0.1], ["EVM open set test"]),
    "evm_ood": ("llmv3_class_incremental_evm_ood.py", ["--lambda_evm", 0.1] + OOD, ["EVM open set test", "Open-set (+"]),
    "ievm":    ("llmv3_class_incremental_ievm_training.py", ["--lambda_evm", 0.1], ["EVM open set test"]),
    "regevm":  ("llmv3_class_incremental_reg_evm_training.py", ["--lambda_evm", 0.1], ["EVM open set test"]),
}


def cil_args(data, base_model, base_classes, new_class, ckpt_dir, strategy):
    return [
        "--data_dir", os.path.join(data, "all_prepdataset"), "--ocr_tensor_path", os.path.join(data, "bert_ocr_rvl"),
        "--all_classes", ",".join(CIL_ALL), "--base_classes", ",".join(base_classes), "--unseen_classes", new_class,
        "--dataset_name", "rvl_cdip", "--base_model_path", base_model, "--checkpoint_dir", ckpt_dir,
        "--batch_size", 2, "--lr", 1e-4, "--num_epochs", 1, "--patience", 1, "--images_per_class", 3,
        "--strategy", strategy, "--use_ewc", "--max_exemplars", 2, "--exemplar_selection", "herding",
        "--use_bias_correction", "--use_balanced_sampler", "--training_mode", "last_layer",
        "--base_model_acc", 0.9, "--full_model_acc", 0.9, "--device", DEVICE,
    ]


def best_checkpoint(folder):
    found = glob.glob(os.path.join(folder, "layoutlmv3_cil_incremental_*_best.pt"))
    return found[0] if found else os.path.join(folder, "<no layoutlmv3_cil_incremental_*_best.pt>")


def run_method(test, method, strategy):
    script, extra, logs = METHODS[method]
    out = out_dir("layout_cil", f"{method}_{strategy}")
    base = os.path.join(test.data, "ckpt", "llmv3_base_3.pt")
    run_script(test, SRC + script, cil_args(test.data, base, CIL_BASE, CIL_NEW, out, strategy) + extra,
               cwd=out, expect_logs=logs)
    test.assertTrue(os.path.exists(best_checkpoint(out)), f"no best checkpoint in {out}")


class TestLayoutCIL(PipelineTestCase):
    def test_evm_persistence_two_jobs(self):
        """Job 1 adds memo and saves EVM + ViM; job 2 adds budget and loads them (EVM+OOD, --evm_persist)."""
        script, extra, _ = METHODS["evm_ood"]
        out1, out2 = out_dir("layout_cil", "persist_1"), out_dir("layout_cil", "persist_2")
        base = os.path.join(self.data, "ckpt", "llmv3_base_3.pt")
        run_script(self, SRC + script, cil_args(self.data, base, CIL_BASE, CIL_NEW, out1, "standard") + extra + ["--evm_persist"],
                   cwd=out1)
        step1 = best_checkpoint(out1)
        self.assertTrue(os.path.exists(step1[:-3] + "_evm_state.pt"), "EVM state not saved next to the step-1 checkpoint")
        run_script(self, SRC + script,
                   cil_args(self.data, step1, CIL_BASE + [CIL_NEW], "budget", out2, "standard") + extra + ["--evm_persist"],
                   cwd=out2, expect_logs=["Loaded open-set state"])

    def test_joint_training_baseline(self):
        out = out_dir("layout_cil", "joint")
        base = os.path.join(self.data, "ckpt", "llmv3_base_3.pt")
        run_script(self, SRC + METHODS["no_evm"][0],
                   cil_args(self.data, base, CIL_BASE, CIL_NEW, out, "standard") + ["--joint_training"],
                   cwd=out, expect_logs=["Joint training on all data"])


add_matrix_tests(TestLayoutCIL, METHODS, run_method)

if __name__ == "__main__":
    unittest.main()
