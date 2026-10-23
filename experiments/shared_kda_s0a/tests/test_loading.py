import tempfile,unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from beatfcn_s0.pupu import load_teacher
from beatfcn_s0.model import SharedModel
from test_s0 import fixture

class LoadingTests(unittest.TestCase):
    def test_missing_or_extra_keys_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.safetensors'
            save_file({'patch_embed.bad':torch.zeros(1)},str(p))
            with self.assertRaisesRegex(ValueError,'keys mismatch'):load_teacher(p)
    def test_toy_state_roundtrip_exact(self):
        m,c,y=fixture();m.eval();expected=m(c,y)['action']
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.pt';torch.save(m.state_dict(),p)
            restored=SharedModel(m.cfg);restored.load_state_dict(torch.load(p,weights_only=True),strict=True);restored.eval()
        torch.testing.assert_close(expected,restored(c,y)['action'],atol=0,rtol=0)
