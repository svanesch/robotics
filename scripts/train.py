"""Train the small VLA on a collected dataset: pixi run train [--config configs/train/small_vla.yaml] [--resume]"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from robotics.policy.data import VLADataset
from robotics.policy.model import VLA, VLAConfig, count_params
from robotics.policy.tokenizer import Tokenizer
from robotics.policy.train_utils import augment, policy_loss, save_checkpoint, to_device
from robotics.policy.views import save_view, train_view, val_view
from robotics.status import StatusReporter

def masked_l1(pred, target, is_pad):
    valid = (~is_pad)[..., None].float()
    return ((pred - target).abs() * valid).sum() / (valid.sum() * pred.shape[-1] + 1e-8)


@torch.no_grad()
def evaluate(model, loader, device, stats, max_batches, grip_bce=False):
    action_std = stats["action_std"]
    model.eval()
    tot, n, per_joint = 0.0, 0, torch.zeros(len(action_std), device=device)
    for b, batch in enumerate(loader):
        if b >= max_batches:
            break
        x = to_device(batch, device)
        pred = model(x["wrist"], x["external"], x["state"], x["tokens"])
        tot += policy_loss(pred, x["actions"], x["is_pad"], stats, grip_bce).item()
        valid = (~x["is_pad"])[..., None].float()
        per_joint += ((pred - x["actions"]).abs() * valid).sum((0, 1)) / valid.sum()
        n += 1
    model.train()
    raw = (per_joint / n).cpu().numpy() * action_std  # mean abs error in raw units (rad; gripper in [0,1])
    return tot / n, raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train/small_vla.yaml")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--dataset", nargs="+", default=None, help="one or more dataset roots (concatenated)")
    p.add_argument("--init-from", default=None, help="fine-tune: start from this checkpoint (weights and normalisation stats)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.steps:
        cfg["steps"] = a.steps
    if a.run_name:
        cfg["run_name"] = a.run_name
    if a.dataset:
        cfg["dataset"] = a.dataset
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = torch.device(a.device)
    run = Path("runs") / cfg["run_name"]
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))

    roots = cfg["dataset"] if isinstance(cfg["dataset"], list) else [cfg["dataset"]]
    init_ck = torch.load(a.init_from, map_location="cpu", weights_only=False) if a.init_from else None
    stats = {k: np.asarray(v) for k, v in init_ck["stats"].items()} if init_ck else None  # keep the old normalisation when fine-tuning
    train_sets = []
    for r in roots:
        train_sets.append(VLADataset(r, "train", cfg["chunk"], stats=stats or (train_sets[0].stats if train_sets else None)))
    first = train_sets[0]
    val_sets = [VLADataset(r, "val", cfg["chunk"], stats=first.stats) for r in roots]
    train_ds, val_ds = ConcatDataset(train_sets), ConcatDataset(val_sets)
    print(f"train frames {len(train_ds)} ({sum(len(d.lengths) for d in train_sets)} episodes) | val frames {len(val_ds)} ({sum(len(d.lengths) for d in val_sets)} episodes) | roots {roots}")
    kw = dict(num_workers=cfg["num_workers"], pin_memory=device.type == "cuda", persistent_workers=cfg["num_workers"] > 0)
    boost, n_early = cfg.get("early_frame_weight", 1.0), cfg.get("early_frames", 50)
    rw = cfg.get("root_weights") or [1.0] * len(train_sets)  # per-dataset sampling weight (e.g. upweight DAgger data)
    rel_w, rel_win = cfg.get("release_weight", 1.0), cfg.get("release_window", 40)

    def release_boost(d) -> np.ndarray:  # frames around the moment the gripper opens: placing is short but decides the stack
        w = np.ones(len(d))
        closed = d.action[:, -1] > 0.5
        for st, ln in zip(d.starts, d.lengths):
            c = closed[st:st + ln]
            for t in np.flatnonzero(c[:-1] & ~c[1:]) + 1:
                w[st + max(0, t - rel_win):st + min(ln, t + rel_win // 2)] = rel_w
        return w

    if boost != 1.0 or rel_w != 1.0 or any(x != 1.0 for x in rw):  # the first frames of an episode are where the prompt decides which block to go for
        w = np.concatenate([np.where(np.arange(len(d)) - d.starts[d.frame_ep] < n_early, boost, 1.0) * release_boost(d) * k for d, k in zip(train_sets, rw)])
        sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, cfg["batch_size"], sampler=sampler, drop_last=True, **kw)
    else:
        train_loader = DataLoader(train_ds, cfg["batch_size"], shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, cfg["batch_size"], shuffle=True, drop_last=True, **kw)

    mc = VLAConfig(vocab_size=len(Tokenizer()), chunk=cfg["chunk"], state_dim=first.state.shape[1],
                   action_dim=first.action.shape[1], max_text=16, **cfg["model"])
    model = VLA(mc).to(device)
    if init_ck:
        model.load_state_dict(init_ck["model"])
        print("initialised from", a.init_from, "(step", init_ck["step"], ")")
    model.train()
    print(f"parameters: {count_params(model) / 1e6:.1f} M | device {device}")
    for mod in list(model.backbone.children())[: cfg.get("freeze_backbone_modules", 0)]:  # frozen generic low-level filters
        for q in mod.parameters():
            q.requires_grad = False
    lang_names = ("text_", "film", "query_proj", "key_proj", "coord_proj", "readout_cam")  # prompt-grounding parameters
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone") and p.requires_grad]
    lang = [p for n, p in model.named_parameters() if n.startswith(lang_names)]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone") and not n.startswith(lang_names)]
    print(f"trainable parameters: {sum(p.numel() for p in bb + rest + lang) / 1e6:.1f} M")
    lang_lr = cfg.get("lang_lr", cfg["lr"])
    groups = [(g, lr) for g, lr in ((bb, cfg["backbone_lr"]), (rest, cfg["lr"]), (lang, lang_lr)) if g]  # a text-free model has no `lang`
    optim = torch.optim.AdamW([{"params": g, "lr": lr} for g, lr in groups], weight_decay=cfg["weight_decay"])
    base_lrs = [lr for _, lr in groups]
    step, best = 0, float("inf")
    if a.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optim.load_state_dict(ck["optim"]); step = ck["step"]; best = ck.get("best_val", best)
        print("resumed at step", step)

    def lr_scale(s):
        if s < cfg["warmup_steps"]:
            return (s + 1) / cfg["warmup_steps"]
        return 0.5 * (1 + math.cos(math.pi * (s - cfg["warmup_steps"]) / max(1, cfg["steps"] - cfg["warmup_steps"])))

    aug = cfg["augment"]
    grip_bce = bool(cfg.get("grip_bce", False))
    rep = StatusReporter("train", cfg["run_name"], total=cfg["steps"],
                         meta={"dataset": str(cfg["dataset"]), "batch": cfg["batch_size"], "init_from": a.init_from or "-", "device": str(device)})
    rep.update(progress=step)
    rep.state["progress0"] = float(step)  # a resumed run only did the steps after this point (for the ETA)
    if step and (run / "log.jsonl").exists():  # resumed: bring the curves back
        for line in (run / "log.jsonl").read_text().splitlines():
            rec = json.loads(line)
            for k in ("train_l1", "val_l1"):
                if k in rec and rec["step"] <= step:
                    rep.series(k, rec[k], x=rec["step"])
    last_train = None
    val_idx = np.random.default_rng(0).choice(len(val_ds), 4, replace=False)
    log = open(run / "log.jsonl", "a")
    t0, run_loss, it = time.time(), 0.0, iter(train_loader)
    while step < cfg["steps"]:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)
        x = to_device(batch, device)
        for k in ("wrist", "external"):
            x[k] = augment(x[k], aug["shift"], aug["brightness"], aug["contrast"], aug["saturation"])
        for g, lr in zip(optim.param_groups, base_lrs):
            g["lr"] = lr * lr_scale(step)
        pred = model(x["wrist"], x["external"], x["state"], x["tokens"])
        loss = policy_loss(pred, x["actions"], x["is_pad"], first.stats, grip_bce, cfg.get("bce_weight", 1.0))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optim.step()
        step += 1
        run_loss += loss.item()
        if step % cfg["log_every"] == 0:
            rec = {"step": step, "train_l1": run_loss / cfg["log_every"], "lr": optim.param_groups[-1]["lr"],
                   "sec_per_step": (time.time() - t0) / cfg["log_every"]}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            rep.update(progress=step, train_l1=rec["train_l1"], lr=rec["lr"], sec_per_step=rec["sec_per_step"])
            rep.series("train_l1", rec["train_l1"], x=step)
            last_train = train_view(x)
            rep.preview(last_train, key="train", min_interval=3.0)
            run_loss, t0 = 0.0, time.time()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            val, raw = evaluate(model, val_loader, device, first.stats, cfg["val_batches"], grip_bce)
            rec = {"step": step, "val_l1": val, "val_l1_raw_per_joint": [round(float(v), 4) for v in raw]}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            rep.update(progress=step, val_l1=val)
            rep.series("val_l1", val, x=step)
            vv = val_view(model, val_ds, device, val_idx)
            rep.preview(vv, key="val", min_interval=0.0)
            save_view(run / "views" / f"val_{step:06d}.jpg", vv)  # kept per checkpoint: the dashboard's history pages browse these
            if last_train is not None:
                save_view(run / "views" / f"train_{step:06d}.jpg", last_train)
            if val < best:
                best = val
                save_checkpoint(run / "best.pt", model, cfg, first.stats, step, {"best_val": best})
            save_checkpoint(run / "last.pt", model, cfg, first.stats, step, {"best_val": best}, optim)
    rep.finish("done", best_val=best)
    print(f"done. best val l1 (normalised): {best:.4f}")


if __name__ == "__main__":
    main()
