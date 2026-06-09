import torch
import transformers
import dllm
from .common import ERROR_THRESHOLD


def _tiny_a2d_qwen3_5():
    base = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B-Base").text_config
    cfg = dllm.pipelines.a2d.A2DQwen3_5TextConfig(
        **{**base.to_dict(),
           "num_hidden_layers": 4,
           "layer_types": ["linear_attention", "linear_attention",
                           "linear_attention", "full_attention"],
           "hidden_size": 128, "head_dim": 32, "intermediate_size": 256}
    )
    torch.manual_seed(0)
    return dllm.pipelines.a2d.A2DQwen3_5LMHeadModel(cfg).eval()


def test_a2d_qwen3_5_future_affects_past():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _tiny_a2d_qwen3_5().to(device)
    a = torch.tensor([[101, 102, 103, 104]], device=device)
    b = torch.tensor([[101, 102, 999, 104]], device=device)  # perturb position 2 (future of 1)
    with torch.no_grad():
        la = model(a).logits
        lb = model(b).logits
    diff = (la[:, 1, :] - lb[:, 1, :]).abs().max().item()
    assert diff > ERROR_THRESHOLD, f"not bidirectional, diff={diff}"


def test_a2d_qwen3_5_config_bidirectional_flag_roundtrips(tmp_path):
    import transformers, dllm
    base = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B-Base").text_config
    cfg = dllm.pipelines.a2d.A2DQwen3_5TextConfig(**base.to_dict(), bidirectional_linear=True)
    cfg.save_pretrained(tmp_path)
    cfg2 = dllm.pipelines.a2d.A2DQwen3_5TextConfig.from_pretrained(tmp_path)
    assert cfg2.bidirectional_linear is True


def test_a2d_qwen3_5_bidirectional_linear_future_affects_past():
    import transformers, torch, dllm
    base = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B-Base").text_config
    cfg = dllm.pipelines.a2d.A2DQwen3_5TextConfig(
        **{**base.to_dict(),
           "num_hidden_layers": 3,
           "layer_types": ["linear_attention", "linear_attention", "linear_attention"],
           "hidden_size": 128, "head_dim": 32, "intermediate_size": 256},
        bidirectional_linear=True,
    )
    torch.manual_seed(0)
    model = dllm.pipelines.a2d.A2DQwen3_5LMHeadModel(cfg).eval()
    a = torch.tensor([[101, 102, 103, 104]])
    b = torch.tensor([[101, 102, 103, 999]])  # perturb LAST token
    with torch.no_grad():
        diff = (model(a).logits[:, 0] - model(b).logits[:, 0]).abs().max().item()
    from .common import ERROR_THRESHOLD
    assert diff > ERROR_THRESHOLD, "linear layers not bidirectional"
