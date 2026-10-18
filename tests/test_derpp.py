"""DER++ (utils/class_IL/derpp.py): the memory, the replay loss, and the CIL pipeline over two jobs for both
backbones (step 1 builds and saves the memory, step 2 loads it)."""
import glob
import os
import tempfile
import unittest

import torch

from tests.fixtures import CIL_ALL, CIL_BASE, CIL_NEW
from tests.runner import PipelineTestCase, check_predictions, out_dir, run_script
from tests.test_eaml_cil import cil_args as eaml_args, without
from tests.test_layout_cil import cil_args as layout_args, best_checkpoint
from utils.class_IL.derpp import DerppMemory, derpp_replay_loss, logit_matching, memory_path


class TestMemory(unittest.TestCase):
    def test_add_save_load_batches(self):
        m = DerppMemory()
        m.add(["a", "b"], [0, 1], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        m.add(["c"], [2], torch.tensor([[5.0, 6.0, 7.0]]))                  # stored when one more class was known
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "mem.pt")
            m.save(p)
            m = DerppMemory.load(p)
        self.assertEqual(len(m), 3)
        batch, labels, stored, mask = next(m.batches(3, collate=list, seed=0))
        self.assertEqual(sorted(batch), ["a", "b", "c"])
        self.assertEqual(stored.shape, (3, 3))
        for item, lab, row, mrow in zip(batch, labels.tolist(), stored, mask):
            n = 3 if item == "c" else 2
            self.assertEqual(int(mrow.sum()), n)                              # only the classes known when stored
            self.assertEqual(lab, {"a": 0, "b": 1, "c": 2}[item])
        it = m.batches(2, collate=list)
        self.assertEqual([len(next(it)[0]) for _ in range(4)], [2, 1, 2, 1])   # endless: passes of 2 + 1

    def test_memory_path(self):
        self.assertEqual(memory_path("/x/best_model_memo.pth"), "/x/best_model_memo_derpp_memory.pt")

    def test_logit_matching(self):
        stored = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        current = torch.tensor([[1.0, 2.0, 9.0, 9.0], [3.0, 4.0, 5.0, 9.0]], requires_grad=True)
        self.assertEqual(float(logit_matching(current, stored, mask)), 0.0)  # masked / extra classes do not count
        shifted = (current + 1.0)
        self.assertAlmostEqual(float(logit_matching(shifted, stored, mask)), 1.0)
        logit_matching(shifted, stored, mask).backward()
        self.assertEqual(current.grad[0, 2].item(), 0.0)                      # no gradient on unmatched classes
        self.assertEqual(current.grad[1, 3].item(), 0.0)

    def test_replay_loss(self):
        m = DerppMemory()
        m.add([torch.ones(4), torch.zeros(4)], [0, 1], torch.tensor([[2.0, -2.0], [-2.0, 2.0]]))
        lin = torch.nn.Linear(4, 3)
        loss = derpp_replay_loss(lambda b: lin(torch.stack(b)), m.batches(2, collate=list), "cpu", 0.1, 0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(lin.weight.grad)


class TestDerppPipeline(PipelineTestCase):
    def test_eaml_two_jobs(self):
        out = out_dir("derpp", "eaml")
        base = os.path.join(self.data, "ckpt", "eaml_base_3.pt")
        args = lambda base_model, base_classes, new: without(
            eaml_args(self.data, base_model, base_classes, new, out, "standard"),
            "--use_ewc", "--use_bias_correction", "--use_balanced_sampler") + ["--derpp"]
        run_script(self, "src/class_incremental/eaml/class_incremental.py", args(base, CIL_BASE, CIL_NEW), cwd=out,
                   expect_files=[memory_path(os.path.join(out, f"best_model_{CIL_NEW}.pth"))],
                   expect_logs=["Built the DER++ memory", "Saved the DER++ memory"])
        check_predictions(self, out, f"cil_{CIL_NEW}", ["seen", "unseen"], method="DER++")
        run_script(self, "src/class_incremental/eaml/class_incremental.py",
                   args(os.path.join(out, f"best_model_{CIL_NEW}.pth"), CIL_BASE + [CIL_NEW], "budget"), cwd=out,
                   expect_logs=["Loaded the DER++ memory"])
        mem = DerppMemory.load(memory_path(os.path.join(out, "best_model_budget.pth")))
        self.assertEqual(sorted({mem.labels[i] for i in range(len(mem))}), list(range(len(CIL_ALL))))
        self.assertEqual({t.numel() for t in mem.logits}, {len(CIL_BASE), len(CIL_BASE) + 1, len(CIL_ALL)})

    def test_layout_two_jobs(self):
        out1, out2 = out_dir("derpp", "layout_1"), out_dir("derpp", "layout_2")
        base = os.path.join(self.data, "ckpt", "llmv3_base_3.pt")
        args = lambda base_model, base_classes, new, out: without(
            layout_args(self.data, base_model, base_classes, new, out, "standard"),
            "--use_ewc", "--use_bias_correction", "--use_balanced_sampler") + ["--derpp"]
        run_script(self, "src/class_incremental/llmv3/llmv3_class_incremental.py", args(base, CIL_BASE, CIL_NEW, out1),
                   cwd=out1, expect_logs=["Built the DER++ memory", "Saved the DER++ memory"])
        step1 = best_checkpoint(out1)
        self.assertTrue(os.path.exists(memory_path(step1)), "DER++ memory not saved next to the step-1 checkpoint")
        check_predictions(self, out1, f"cil_{CIL_NEW}", ["seen", "unseen"], method="DER++")
        run_script(self, "src/class_incremental/llmv3/llmv3_class_incremental.py",
                   args(step1, CIL_BASE + [CIL_NEW], "budget", out2), cwd=out2, expect_logs=["Loaded the DER++ memory"])
        self.assertTrue(glob.glob(os.path.join(out2, "*_derpp_memory.pt")))


if __name__ == "__main__":
    unittest.main()
