
# train_ofdm_cplus.py
# -*- coding: utf-8 -*-
"""
Minimal training loop for ModelCPlus:
- Reads YAML config (if PyYAML available), otherwise uses defaults
- Dummy dataset (each sample = 1 RB): enc_feats [24,d_in], dec_feats [144,d_in], target_bits [144,bits]
- Metrics: BER / BLER / ECE
- Checkpoints: best.pt (by BER) and last.pt
Run:
    python train_ofdm_cplus.py --config config_ofdm_cplus.yaml
"""
import os, sys, time, math, argparse, json, random
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure local import works when launched from anywhere
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from transformer_ofdm_schemeC_plus_demo import ModelCPlus, ModelCPlusConfig


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_load_yaml(path: str) -> Optional[Dict[str, Any]]:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        return None


def now_str():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


# -----------------------------
# Dummy Dataset
# -----------------------------
class DummyOFDMDataset(Dataset):
    """
    Each item corresponds to one RB sample.
    """
    def __init__(self, num_samples: int, d_in: int, bits_per_symbol: int):
        self.N = num_samples
        self.d_in = d_in
        self.bits = bits_per_symbol

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        enc = torch.randn(24, self.d_in)        # pilot RE features
        dec = torch.randn(144, self.d_in)       # data RE features
        bits = torch.randint(0, 2, (144, self.bits)).float()  # labels
        return enc, dec, bits


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> Dict[str, float]:
    """
    logits: [B, 144, bits]
    labels: [B, 144, bits]  (0/1)
    Returns: dict with BER, BLER, ECE
    """
    B = logits.size(0)
    bits = logits.size(-1)

    # BER
    preds = (logits > 0).float()
    bit_err = (preds != labels).float()
    ber = bit_err.mean().item()

    # BLER (1 if any bit error within a block)
    block_err = (bit_err.view(B, -1).sum(dim=1) > 0).float().mean().item()

    # ECE (binary class): confidence = max(p, 1-p), accuracy = 1 if predicted class equals label
    probs = torch.sigmoid(logits)                              # P(bit=1)
    pred_is_one = preds > 0.5
    conf = torch.where(pred_is_one, probs, 1.0 - probs)        # confidence of predicted class
    correct = (preds == labels).float()

    conf = conf.view(-1)
    correct = correct.view(-1)

    # binning
    bin_ids = torch.clamp((conf * n_bins).long(), max=n_bins - 1)
    bin_total = torch.bincount(bin_ids, minlength=n_bins).float()
    bin_conf_sum = torch.bincount(bin_ids, weights=conf, minlength=n_bins).float()
    bin_acc_sum = torch.bincount(bin_ids, weights=correct, minlength=n_bins).float()

    nonzero = bin_total > 0
    avg_conf = torch.zeros(n_bins, device=logits.device)
    avg_acc = torch.zeros(n_bins, device=logits.device)
    avg_conf[nonzero] = bin_conf_sum[nonzero] / bin_total[nonzero]
    avg_acc[nonzero] = bin_acc_sum[nonzero] / bin_total[nonzero]

    ece = torch.sum(bin_total[nonzero] * torch.abs(avg_acc[nonzero] - avg_conf[nonzero])) / conf.numel()
    return {"BER": ber, "BLER": block_err, "ECE": ece.item()}


# -----------------------------
# Train / Eval
# -----------------------------
def run_epoch(model, loader, optimizer=None, scaler=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_main, total_cons, n_batches = 0.0, 0.0, 0.0, 0

    for enc, dec, bits in loader:
        enc = enc.to(device)
        dec = dec.to(device)
        bits = bits.to(device)

        if is_train and scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(enc, dec)
                loss_main = F.binary_cross_entropy_with_logits(out["logits"], bits)
                loss_cons = F.mse_loss(out["logits_learned"], out["logits_analytic"].detach())
                loss = loss_main + 0.05 * loss_cons
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(enc, dec)
            loss_main = F.binary_cross_entropy_with_logits(out["logits"], bits)
            loss_cons = F.mse_loss(out["logits_learned"], out["logits_analytic"].detach())
            loss = loss_main + 0.05 * loss_cons
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        total_main += loss_main.item()
        total_cons += loss_cons.item()
        n_batches += 1

    avg = {
        "loss": total_loss / max(1, n_batches),
        "loss_main": total_main / max(1, n_batches),
        "loss_cons": total_cons / max(1, n_batches),
    }
    return avg


@torch.no_grad()
def evaluate(model, loader, device="cpu") -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_bits = []
    for enc, dec, bits in loader:
        enc = enc.to(device)
        dec = dec.to(device)
        bits = bits.to(device)
        out = model(enc, dec)
        all_logits.append(out["logits"].detach())
        all_bits.append(bits.detach())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_bits, dim=0)
    metrics = compute_metrics(logits, labels, n_bins=15)
    return metrics


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_ofdm_cplus.yaml")
    args = parser.parse_args()

    cfg_yaml = try_load_yaml(args.config)
    if cfg_yaml is None:
        # Defaults if YAML not available
        cfg_yaml = {
            "seed": 0,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "output_dir": "./ofdm_cplus_runs",
            "epochs": 3,
            "batch_size": 8,
            "num_train": 600,
            "num_val": 120,
            "amp": True,
            "opt": {"lr": 3e-4, "weight_decay": 1e-2},
            "model": {
                "d_in": 4, "d_model": 64, "nhead": 4,
                "num_enc_layers": 2, "num_dec_layers": 3,
                "dim_ff": 128, "dropout": 0.1,
                "bits_per_symbol": 4, "neighbor_K": 2,
                "stem_hidden": 64, "mix_init_logit": 0.0,
            },
        }

    if cfg_yaml.get("device", "auto") == "auto":
        cfg_yaml["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(int(cfg_yaml.get("seed", 0)))
    device = cfg_yaml.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Datasets / loaders
    mcfg = cfg_yaml["model"]
    d_in = mcfg["d_in"]
    bits = mcfg["bits_per_symbol"]
    train_set = DummyOFDMDataset(cfg_yaml["num_train"], d_in=d_in, bits_per_symbol=bits)
    val_set = DummyOFDMDataset(cfg_yaml["num_val"], d_in=d_in, bits_per_symbol=bits)

    train_loader = DataLoader(train_set, batch_size=cfg_yaml["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=cfg_yaml["batch_size"], shuffle=False, drop_last=False)

    # Model / opt
    model = ModelCPlus(ModelCPlusConfig(**mcfg)).to(device)
    opt_cfg = cfg_yaml["opt"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg.get("lr", 3e-4),
                                  weight_decay=opt_cfg.get("weight_decay", 1e-2))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg_yaml.get("amp", True)) and device.startswith("cuda"))

    # Output dir
    os.makedirs(cfg_yaml["output_dir"], exist_ok=True)
    run_dir = os.path.join(cfg_yaml["output_dir"], time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    os.makedirs(run_dir, exist_ok=True)

    # Save resolved config
    with open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(cfg_yaml, f, ensure_ascii=False, indent=2)

    best_ber = 1.0
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")

    for epoch in range(1, int(cfg_yaml["epochs"]) + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer=optimizer, scaler=scaler, device=device)
        val = evaluate(model, val_loader, device=device)
        dt = time.time() - t0

        # Save last
        torch.save({"model": model.state_dict(), "epoch": epoch, "val": val, "train": tr}, last_path)

        # Save best (by BER; tie break by ECE)
        is_best = (val["BER"] < best_ber) or (math.isclose(val["BER"], best_ber) and epoch == 1)
        if is_best:
            best_ber = val["BER"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val, "train": tr}, best_path)

        log = {
            "epoch": epoch,
            "time_sec": round(dt, 3),
            **{f"train_{k}": round(v, 6) for k, v in tr.items()},
            **{f"val_{k}": round(v, 6) for k, v in val.items()},
            "best_BER": round(best_ber, 6),
        }
        print(log)

    print(f"Training done. Best ckpt: {best_path}")
    print(f"Last ckpt: {last_path}")


if __name__ == "__main__":
    main()
