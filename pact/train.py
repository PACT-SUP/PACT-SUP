"""Train PACT on LP-PDBbind (seeds 0-5) or on one PDBbind CleanSplit fold (0-4), as in the paper."""

import math
from collections import Counter
from functools import lru_cache

import pandas as pd
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

import paths
from pact.model import PACT, global_prior, graphs, normalisation, sparsemax

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
LR_PATIENCE = 5
PATIENCE = 20
MIN_EPOCHS = 25
GRAD_CLIP = 5.0
MAX_RECOVERIES = 5
RANK_WEIGHT = 0.2
RANK_TEMPERATURE = 0.25
LOCAL_WEIGHT = 800
TRANSPORT_WEIGHT = 10000
POSE_JITTER = 0.05  # A, for the transport penalty


@lru_cache(maxsize=1)
def cleansplit():
    """CleanSplit.csv (membership, the five folds, CASF-2016) and {pdb_id: graph}."""
    return (
        pd.read_csv(paths.need("cleansplit/CleanSplit.csv")),
        torch.load(paths.need("cleansplit/cleansplit_all.pt"), weights_only=False)[
            "graphs"
        ],
    )


def load_split(dataset, index):
    """Train and validation graphs, and the table with their protein sequences."""
    if dataset == "lp":
        return (
            list(graphs("train")),
            list(graphs("val")),
            pd.read_csv(paths.need("lp_pdbbind/LP_PDBBind.csv")),
        )
    table, have = cleansplit()
    fold = table[f"fold{index}"]
    train = [have[p] for p in table.pdb_id[fold == "train"] if p in have]
    val = [have[p] for p in table.pdb_id[fold == "val"] if p in have]
    return train, val, table


def add_cluster_weights(graph_list, table):
    """Give each complex a cluster (its protein sequence) and a weight of 1 / cluster size, rescaled to mean 1."""
    sequences = dict(zip(table.pdb_id, table.seq.astype(str)))
    numbers = {}
    clusters = [
        numbers.setdefault(sequences.get(g.pdb_id, g.pdb_id), len(numbers))
        for g in graph_list
    ]
    size = Counter(clusters)
    scale = len(graph_list) / sum(1.0 / size[c] for c in clusters)
    for graph, cluster in zip(graph_list, clusters):
        graph.cluster = torch.tensor([cluster])
        graph.weight = torch.tensor([scale / size[cluster]])


def local_penalty(readout, out, atom_features):
    """||d channel / d basis - coefficient||^2: the pooled basis times the coefficient shown beside it should be
    the channel value."""
    base = out["basis"]
    z = base.detach().requires_grad_(True)
    safe = torch.where(base.abs() > 1e-12, base, torch.ones_like(base))
    value = z * (out["pre_mlp"] / safe).detach() * (1.0 + readout.interaction_graph(z))
    gate = (
        z @ readout.group_gate_weight
        + readout.group_gate_bias
        + atom_features[out["group_atom"]] @ readout.atom_gate_weight
    )
    vdw_scale = torch.ones_like(base)
    vdw_scale[:, 0] = sparsemax(out["distance_prior"].detach() + gate, out["segment"])
    channels = value * (
        readout.physical_weights()[None] * vdw_scale * out["local_coef"].detach()
    )
    gradient = torch.autograd.grad(channels.sum(), z, create_graph=True)[0]
    mask = (base.abs() > 1e-9).to(base.dtype)
    return ((gradient - channels / safe) * mask).square().sum() / mask.sum().clamp_min(
        1.0
    )


def transport_penalty(model, batch, out):
    """z(x')^2 (theta(x') - theta(x))^2 for a slightly jittered pose x': zero when the coefficients do not move."""
    base = out["basis"]
    live = base.abs() > 1e-9
    theta = (out["channels"] / torch.where(live, base, torch.ones_like(base))).detach()
    pos = batch.pos
    batch.pos = pos + POSE_JITTER * torch.randn_like(pos)
    moved = model(batch)
    batch.pos = pos
    mask = (live & (moved["basis"].abs() > 1e-9)).to(base.dtype)
    return (
        (moved["channels"] - moved["basis"] * theta) * mask
    ).square().sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def validate(model, loader, mean, std, device):
    """Plain and cluster-balanced validation RMSE."""
    model.eval()
    prediction, target, cluster = [], [], []
    for batch in loader:
        batch = batch.to(device)
        prediction.append((model(batch)["prediction"] * std + mean).cpu())
        target.append(batch.y.cpu())
        cluster.append(batch.cluster.cpu())
    model.train()
    prediction, target, cluster = (
        torch.cat(prediction),
        torch.cat(target),
        torch.cat(cluster),
    )
    error = (prediction - target).square()
    per_cluster = torch.zeros(int(cluster.max()) + 1).index_add_(0, cluster, error)
    counts = torch.bincount(cluster, minlength=len(per_cluster))
    seen = counts > 0
    return float(error.mean().sqrt()), math.sqrt(
        float((per_cluster[seen] / counts[seen]).mean())
    )


def train(dataset, index, epochs=400, folder=None, device="cuda:0"):
    """Train LP-PDBbind seed `index` or CleanSplit fold `index` and save the best epoch as
    <folder>/<dataset><index>.pt. Non-finite weights roll back to the best state and halve the learning rate."""
    name = f"{dataset}{index}"
    torch.manual_seed(index)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_graphs, val_graphs, table = load_split(dataset, index)
    add_cluster_weights(train_graphs, table)
    add_cluster_weights(val_graphs, table)
    mean, std = normalisation(train_graphs)
    prior = global_prior(train_graphs, mean, std)
    model = PACT(prior).to(device)
    mean_t, std_t = torch.tensor(mean, device=device), torch.tensor(std, device=device)
    train_loader = DataLoader(
        train_graphs,
        BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(index),
    )
    val_loader = DataLoader(val_graphs, BATCH_SIZE)
    physical = model.pair_readout.raw_physical_weight
    others = [p for p in model.parameters() if p is not physical]
    optimizer = torch.optim.AdamW(
        [
            {"params": others, "weight_decay": WEIGHT_DECAY},
            {"params": [physical], "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=LR_PATIENCE, min_lr=1e-6
    )
    print(
        f"{name}: train {len(train_graphs)}  val {len(val_graphs)}  mean {mean:.4f}  std {std:.4f}",
        flush=True,
    )

    best, best_epoch, stale, recoveries = float("inf"), 0, 0, 0
    state = {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            target = (batch.y.view(-1) - mean_t) / std_t
            weight = batch.weight.view(-1)
            out = model(batch)
            prediction = out["prediction"]
            loss = (weight * (prediction - target).square()).sum() / weight.sum()
            ordered = (target[:, None] - target[None, :]) > 0
            if ordered.any():
                gap = (prediction[:, None] - prediction[None, :])[ordered]
                loss = loss + RANK_WEIGHT * F.softplus(-gap / RANK_TEMPERATURE).mean()
            local = local_penalty(model.pair_readout, out, batch.atom_features)
            transport = transport_penalty(model, batch, out)
            loss = loss + LOCAL_WEIGHT * local
            loss = loss + TRANSPORT_WEIGHT * transport
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if not torch.isfinite(
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            ):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            total += float(loss) * batch.num_graphs
            seen += batch.num_graphs

        if not all(p.isfinite().all() for p in model.parameters()):
            recoveries += 1
            model.load_state_dict(state)
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"] * 0.5, 1e-6)
            optimizer.state = type(optimizer.state)()
            print(
                f"{name} {epoch:03d} non-finite weights: rolled back, lr halved",
                flush=True,
            )
            if recoveries >= MAX_RECOVERIES:
                break
            continue

        val_rmse, macro = validate(model, val_loader, mean, std, device)
        scheduler.step(macro)
        if macro < best:
            best, best_epoch, stale = macro, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        train_loss = math.sqrt(total / seen) * std if seen else float("nan")
        print(
            f"{name} {epoch:03d} train={train_loss:.4f} val={val_rmse:.4f} macro={macro:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.1e}",
            flush=True,
        )
        if stale >= PATIENCE and epoch >= MIN_EPOCHS:
            break

    folder = folder or paths.ROOT / "checkpoints" / "retrained"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.pt"
    torch.save({"state_dict": state, "mean": mean, "std": std, "prior": prior}, path)
    print(f"{name} done: best epoch {best_epoch}, macro val {best:.4f}", flush=True)
    return path
