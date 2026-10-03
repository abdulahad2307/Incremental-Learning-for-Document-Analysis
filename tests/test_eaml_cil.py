"""EAML class-incremental pipeline: every method x both strategies (ER, EWC, BC and balanced sampling on;
last-layer training), plus EVM/ViM persistence across two jobs and the joint-training baseline."""
import os
import unittest

from tests.fixtures import CIL_ALL, CIL_BASE, CIL_NEW
from tests.runner import PipelineTestCase, add_matrix_tests, out_dir, run_script

SRC = "src/class_incremental/eaml/"
OOD = ["--ood_method", "vim", "--ood_tpr", 0.95, "--ood_max_per_class", 8]
# method -> (script, extra args, log lines showing the method's own part ran)
METHODS = {
    "no_evm":      ("class_incremental.py", [], []),
    "evm":         ("class_incremental_evm_training.py", [], []),
    "evm_ood":     ("class_incremental_evm_ood.py", OOD + ["--lambda_evm", 0.1, "--lambda_ood", 0.1], ["Open-set step"]),
    "ievm":        ("class_incremental_ievm_training.py", [], ["iEVM"]),
    "regevm":      ("class_incremental_reg_evm_training.py", [], []),
    # not in Fig. 1, kept in the codebase
    "ood_only":    ("class_incremental_ood.py", OOD, ["Open-set step"]),
    "evm_posthoc": ("class_incremental_evm.py", [], []),
}


def cil_args(data, base_model, base_classes, new_class, ckpt_dir, strategy):
    return [
        "--data_dir", os.path.join(data, "all_prepdataset"), "--ocr_tensor_path", os.path.join(data, "eaml_ocr.pt"),
        "--all_classes", ",".join(CIL_ALL), "--base_classes", ",".join(base_classes), "--unseen_classes", new_class,
        "--base_model_path", base_model, "--model_name", "eaml", "--checkpoint_dir", ckpt_dir,
        "--batch_size", 2, "--lr", 1e-4, "--num_epochs", 1, "--patience", 1, "--strategy", strategy,
        "--use_ewc", "--use_exemplars", "--max_exemplars", 32, "--exemplar_selection", "herding",
        "--use_bias_correction", "--use_balanced_sampler", "--training_mode", "last_layer",
    ]


def run_method(test, method, strategy):
    script, extra, logs = METHODS[method]
    out = out_dir("eaml_cil", f"{method}_{strategy}")
    base = os.path.join(test.data, "ckpt", "eaml_base_3.pt")
    run_script(test, SRC + script, cil_args(test.data, base, CIL_BASE, CIL_NEW, out, strategy) + extra,
               cwd=out, expect_files=[os.path.join(out, f"best_model_{CIL_NEW}.pth")], expect_logs=logs)


class TestEAMLCIL(PipelineTestCase):
    def test_evm_persistence_two_jobs(self):
        """Job 1 adds memo and saves EVM + ViM; job 2 adds budget and loads them (EVM+OOD, --evm_persist)."""
        script, extra, _ = METHODS["evm_ood"]
        out = out_dir("eaml_cil", "persist")
        base = os.path.join(self.data, "ckpt", "eaml_base_3.pt")
        run_script(self, SRC + script, cil_args(self.data, base, CIL_BASE, CIL_NEW, out, "standard") + extra + ["--evm_persist"],
                   cwd=out, expect_files=[os.path.join(out, f"evm_state_{CIL_NEW}.pt")])
        step1 = os.path.join(out, f"best_model_{CIL_NEW}.pth")
        run_script(self, SRC + script,
                   cil_args(self.data, step1, CIL_BASE + [CIL_NEW], "budget", out, "standard") + extra + ["--evm_persist"],
                   cwd=out, expect_files=[os.path.join(out, "evm_state_budget.pt")], expect_logs=["Loaded open-set state"])

    def test_joint_training_baseline(self):
        out = out_dir("eaml_cil", "joint")
        base = os.path.join(self.data, "ckpt", "eaml_base_3.pt")
        run_script(self, SRC + METHODS["no_evm"][0],
                   cil_args(self.data, base, CIL_BASE, CIL_NEW, out, "standard") + ["--joint_training"],
                   cwd=out, expect_logs=["Joint training on all data"])


add_matrix_tests(TestEAMLCIL, METHODS, run_method)

if __name__ == "__main__":
    unittest.main()
