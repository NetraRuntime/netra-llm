import torch
import transformers
import dllm
from .common import ERROR_THRESHOLD


def _layer(bidirectional):
    base = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B-Base").text_config
    cfg = dllm.pipelines.a2d.A2DQwen3_5TextConfig(
        **{**base.to_dict(), "hidden_size": 128, "head_dim": 32},
        bidirectional_linear=bidirectional,
    )
    from dllm.pipelines.a2d.models.qwen3_5.modeling_qwen3_5 import A2DQwen3_5GatedDeltaNet
    torch.manual_seed(0)
    return A2DQwen3_5GatedDeltaNet(cfg, layer_idx=0).eval()


def test_gated_deltanet_bidirectional_future_affects_past():
    torch.manual_seed(0)
    layer = _layer(bidirectional=True)
    x = torch.randn(1, 6, 128)
    # Perturb a FUTURE position (3) relative to the checked past position (1).
    # The gated delta-rule forgets quickly (random A_log ~ U(0,16) => ~1e-5/step
    # retention), so the backward-scan signal is only reliably above threshold a
    # few steps out; distance 2 cleanly exercises the future->past path.
    x2 = x.clone(); x2[:, 3, :] += 1.0
    with torch.no_grad():
        o, o2 = layer(x), layer(x2)
    diff_past = (o[:, 1, :] - o2[:, 1, :]).abs().max().item()
    assert diff_past > ERROR_THRESHOLD, f"deltanet not bidirectional, diff={diff_past}"


def test_gated_deltanet_causal_when_disabled():
    torch.manual_seed(0)
    layer = _layer(bidirectional=False)
    x = torch.randn(1, 6, 128)
    x2 = x.clone(); x2[:, 3, :] += 1.0   # perturb a FUTURE position
    with torch.no_grad():
        o, o2 = layer(x), layer(x2)
    diff_past = (o[:, 1, :] - o2[:, 1, :]).abs().max().item()
    assert diff_past < ERROR_THRESHOLD, f"expected causal, diff={diff_past}"
