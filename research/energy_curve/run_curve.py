"""Train modded_nanogpt once, snapshot weights at 10 evenly-spaced steps,
record per-checkpoint NVML joules, then eval each snapshot for char-acc.

Outputs a JSON with [{step, joules_J, duration_s, val_char_accuracy}, ...]
so you can plot acc vs joules.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wikitext import evaluate, load_wikitext103  # noqa: E402

SUBMISSION = REPO / "submissions" / "modded_nanogpt" / "submission.py"
_spec = importlib.util.spec_from_file_location("modded", SUBMISSION)
modded = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(modded)


def _open_nvml():
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    pynvml.nvmlDeviceGetTotalEnergyConsumption(h)  # raises if unsupported
    return pynvml, h


def _read_joules(pynvml, handle) -> float:
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0


def train_with_checkpoints(
    text: str,
    cfg: modded.TrainConfig,
    device: torch.device,
    checkpoint_steps: list[int],
    idle_watts: float = 50.0,
) -> list[dict]:
    """Forked _train_modded that snapshots state_dict + records net joules
    at each step in `checkpoint_steps` (1-indexed step count, taken AFTER
    `optimizer.step()` for that step)."""
    raw = text.encode("utf-8")
    train_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    n = train_bytes.numel()

    model = modded.GPT(
        vocab_size=256,
        num_layers=cfg.num_layers,
        model_dim=cfg.model_dim,
        head_dim=cfg.head_dim,
        max_len=cfg.max_len,
    ).to(device)
    modded._init_modded(model)

    block_2d = [p for p in model.blocks.parameters() if p.ndim >= 2]
    scalars = [p for p in model.parameters() if p.ndim < 2]
    optimizer1 = AdamW(
        [
            dict(params=[model.embed.weight], lr=cfg.embed_lr),
            dict(params=[model.proj.weight], lr=cfg.head_lr),
            dict(params=scalars, lr=cfg.scalar_lr),
        ],
        betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    optimizer2 = modded.Muon(block_2d, lr=cfg.muon_lr, weight_decay=cfg.muon_wd)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    def set_lr(step: int) -> None:
        progress = step / cfg.n_steps
        if progress < 1 - cfg.cooldown_frac:
            eta = 1.0
        else:
            eta = max(0.0, (1 - progress) / cfg.cooldown_frac)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * eta

    pynvml, handle = _open_nvml()

    print(f"[curve] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  cfg={cfg}")
    print(f"[curve] checkpoint_steps={checkpoint_steps}")

    checkpoints: list[dict] = []
    pending = list(checkpoint_steps)

    model.train()
    use_amp = device.type == "cuda"
    torch.cuda.synchronize() if device.type == "cuda" else None
    e0 = _read_joules(pynvml, handle)
    t0 = time.monotonic()

    for step in range(1, cfg.n_steps + 1):
        set_lr(step - 1)
        idx = torch.randint(0, n - cfg.max_len - 1, (cfg.batch_size,), device=device)
        offsets = idx[:, None] + torch.arange(cfg.max_len + 1, device=device)[None, :]
        flat = train_bytes[offsets].long()
        x, y = flat[:, :-1], flat[:, 1:]

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        for opt in optimizers:
            opt.step()

        if pending and step == pending[0]:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - t0
            raw_j = _read_joules(pynvml, handle) - e0
            net_j = raw_j - idle_watts * elapsed
            sd = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            checkpoints.append({
                "step": step,
                "duration_s": elapsed,
                "raw_gpu_joules_J": raw_j,
                "joules_J": net_j,
                "loss": float(loss.item()),
                "state_dict": sd,
            })
            print(f"[curve] step {step:5d}/{cfg.n_steps}  "
                  f"loss {loss.item():.4f}  elapsed {elapsed:.1f}s  "
                  f"raw_gpu={raw_j:,.0f}J  net={net_j:,.0f}J", flush=True)
            pending.pop(0)

    # Stash the architectural config so we can reload checkpoints into a
    # fresh GPT instance during eval.
    return checkpoints


def eval_checkpoint(state_dict: dict, cfg: modded.TrainConfig,
                    val_text: str, device: torch.device,
                    progress_every: int) -> tuple[float, int, float]:
    model = modded.GPT(
        vocab_size=256, num_layers=cfg.num_layers,
        model_dim=cfg.model_dim, head_dim=cfg.head_dim,
        max_len=cfg.max_len,
    ).to(device)
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    char_model = modded.ModdedNanoGPTCharModel(model, device=device)
    r = evaluate(char_model, val_text, progress_every=progress_every)
    return r.accuracy, r.n_chars, r.duration_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("/data"),
                   help="Dir holding wiki.{train,valid,test}.raw")
    p.add_argument("--n-checkpoints", type=int, default=10)
    p.add_argument("--eval-chars", type=int, default=60_000,
                   help="Val window scored per checkpoint (matches leaderboard at 60k)")
    p.add_argument("--idle-watts", type=float, default=50.0,
                   help="Idle baseline subtracted from raw NVML joules; matches EnergyMeter default")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).with_name("result.json"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        sys.exit("CUDA not available — this script needs an NVML-capable GPU.")

    print(f"[curve] loading WikiText-103 from {args.data_dir}")
    train_text = load_wikitext103(args.data_dir, "train")
    valid_text = load_wikitext103(args.data_dir, "valid")
    val_score = valid_text[: args.eval_chars] if args.eval_chars else valid_text

    cfg = modded.TrainConfig()
    step_grid = [round(cfg.n_steps * i / args.n_checkpoints)
                 for i in range(1, args.n_checkpoints + 1)]

    checkpoints = train_with_checkpoints(
        train_text, cfg, device, step_grid, idle_watts=args.idle_watts,
    )

    progress_every = max(1, len(val_score) // 10)
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        args.out.write_text(json.dumps({
            "n_steps": cfg.n_steps,
            "idle_watts": args.idle_watts,
            "eval_chars": args.eval_chars,
            "checkpoints": rows,
        }, indent=2) + "\n")

    for ck in checkpoints:
        print(f"[curve] eval step={ck['step']}  net_J={ck['joules_J']:,.0f}")
        acc, n_chars, dur = eval_checkpoint(
            ck["state_dict"], cfg, val_score, device, progress_every,
        )
        rows.append({
            "step": ck["step"],
            "training_duration_s": ck["duration_s"],
            "raw_gpu_joules_J": ck["raw_gpu_joules_J"],
            "joules_J": ck["joules_J"],
            "train_loss": ck["loss"],
            "val_char_accuracy": acc,
            "val_chars": n_chars,
            "eval_duration_s": dur,
        })
        print(f"[curve]   acc={acc:.4f}  eval={dur:.1f}s")
        flush()  # checkpoint-level durability against Modal timeouts

    print(f"[curve] wrote {args.out}")


if __name__ == "__main__":
    main()
