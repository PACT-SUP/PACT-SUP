"""The paper's PACT numbers, one function per table or paragraph (`PORTS` at the end).

Each function recomputes its numbers from the checkpoints in paths.CHECKPOINTS; run.py compares them with
the printed values in expected/paper_values.json.
"""

import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch_geometric.loader import DataLoader

import paths
from pact.model import CHANNELS, CONTEXT_RANGE, CUTOFF, graphs, load, predict
from pact.train import cleansplit

SEEDS = [f"lp{s}" for s in range(6)]  # the six LP-PDBbind checkpoints
FOLDS = [f"cleansplit{f}" for f in range(5)]  # the five CleanSplit checkpoints
# ProLIF interaction families; each has the model channel of the same name
FAMILIES = (
    "hbond",
    "hydrophobic",
    "vdw",
    "aromatic_proximity",
    "ionic",
    "cationic_aromatic_proximity",
    "halogen_acceptor_proximity",
    "metal",
)
K_VALUES = (1, 3, 5, 8)
AMPLITUDES = (0.10, 0.25)
# printed baseline values the text compares PACT with
RF_SCORE_CASF_R = 0.794
RF_SCORE_DISCREPANCY = 0.323
CHEAPNET_SLOPE = 0.780
SCHNET_PARAMETERS = 281_395


def mean_sd(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}


def cohort(graph_list, size):
    """One complex per protein sequence, taken in the order of a hash of the PDB id."""
    sequence = (
        pd.read_csv(paths.need("lp_pdbbind/LP_PDBBind.csv"))
        .set_index("pdb_id")["seq"]
        .astype(str)
    )
    picked = {}
    for graph in sorted(
        graph_list, key=lambda g: hashlib.sha256(g.pdb_id.encode()).hexdigest()
    ):
        picked.setdefault(sequence.get(graph.pdb_id, graph.pdb_id), graph)
    return list(picked.values())[:size]


def ligand_ordinal(batch):
    """Each node's position among its own complex's ligand atoms."""
    ordinal = torch.full((batch.num_nodes,), -1, dtype=torch.long)
    ligand = torch.where(batch.ligand.bool())[0]
    counts = torch.bincount(batch.batch[ligand], minlength=batch.num_graphs)
    ordinal[ligand] = (
        torch.arange(len(ligand))
        - (torch.cumsum(counts, 0) - counts)[batch.batch[ligand]]
    )
    return ordinal


@torch.inference_mode()
def infer(name, graph_list):
    """Per complex: prediction, baseline, physics (the sum over groups) and target. Per (ligand atom, residue)
    group: its claim and its channel contributions. All in pK."""
    model = load(name)
    complexes, groups = [], []
    for batch in DataLoader(graph_list, 64):
        out = model(batch)
        prediction = out["prediction"] * model.std + model.mean
        claim = out["channels"].sum(dim=1) * model.std
        physics = torch.zeros_like(prediction).index_add_(0, out["group_graph"], claim)
        pdb = list(batch.pdb_id)
        complexes.append(
            pd.DataFrame(
                {
                    "pdb_id": pdb,
                    "target": batch.y.flatten().numpy(),
                    "prediction": prediction.numpy(),
                    "baseline": (out["baseline"] * model.std + model.mean).numpy(),
                    "physics": physics.numpy(),
                }
            )
        )
        frame = pd.DataFrame(
            {
                "pdb_id": [pdb[i] for i in out["group_graph"].numpy()],
                "residue_id": out["group_residue"].numpy(),
                "ligand_atom": ligand_ordinal(batch)[out["group_atom"]].numpy(),
                "claim": claim.numpy(),
            }
        )
        frame[list(CHANNELS)] = (out["channels"] * model.std).numpy()
        groups.append(frame)
    return pd.concat(complexes, ignore_index=True), pd.concat(groups, ignore_index=True)


def residues(groups):
    """{pdb_id: (residue ids, claims)}, each residue's claim summed over its groups."""
    total = groups.groupby(["pdb_id", "residue_id"]).claim.sum()
    return {
        pdb: (rows.index.get_level_values(1).to_numpy(), rows.to_numpy())
        for pdb, rows in total.groupby(level=0)
    }


def drop(graph, residue_ids):
    """The complex with every contact to these residues deleted; the residue nodes stay."""
    changed = graph.clone()
    removed = torch.isin(
        graph.residue_id[graph.contact_edge_index[1]],
        torch.as_tensor(list(residue_ids), dtype=torch.long),
    )
    changed.contact_edge_index = graph.contact_edge_index[:, ~removed]
    return changed


def scores(prediction, target):
    """RMSE, Pearson, Spearman and Harrell's concordance index."""
    p, y = np.asarray(prediction, float), np.asarray(target, float)
    difference = (p[:, None] - p[None, :])[y[:, None] > y[None, :]]
    return {
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "pearson": float(np.corrcoef(p, y)[0, 1]),
        "spearman": float(stats.spearmanr(p, y).statistic),
        "ci_harrell": float(((difference > 0) + 0.5 * (difference == 0)).mean()),
    }


def fidelity(claimed, observed):
    """How well claimed contributions match the observed prediction changes."""
    claimed, observed = np.asarray(claimed, float), np.asarray(observed, float)
    nonzero = (claimed != 0) & (observed != 0)
    slope = np.linalg.lstsq(
        np.column_stack([np.ones(len(claimed)), claimed]), observed, rcond=None
    )[0][1]
    return {
        "normalised_infidelity": float(
            ((claimed - observed) ** 2).sum() / (observed**2).sum()
        ),
        "calibration_slope": float(slope),
        "over_claim_pk": float(np.abs(claimed - observed).mean()),
        "sign_agreement": float(
            (np.sign(claimed[nonzero]) == np.sign(observed[nonzero])).mean()
        ),
    }


def top(values, k):
    """Indices of the k largest |values|, ties going to the lower index."""
    return set(np.lexsort((np.arange(len(values)), -np.abs(values)))[:k])


def overlap(a, b, k):
    """Shared fraction of the top k of a and the top k of b."""
    return len(top(a, k) & top(b, k)) / k


def mean_seed_prediction(graph_list):
    return np.mean([predict(s, graph_list) for s in SEEDS], axis=0)


# accuracy


def seed_scores(sample):
    y = np.concatenate([g.y.numpy() for g in sample])
    per_seed = [scores(predict(s, sample), y) for s in SEEDS]
    return {
        "n": len(sample),
        **{k: mean_sd([r[k] for r in per_seed]) for k in per_seed[0]},
    }


def casf_independent():
    """The 144 CASF-2016 complexes with no similar complex anywhere in PDBbind."""
    table, have = cleansplit()
    keep = table.eval_set.fillna("").str.contains("casf2016") & table.casf2016_indep
    return [have[p] for p in table.pdb_id[keep] if p in have]


def heavy_atoms(graph_list):
    return np.array([float(g.ligand.bool().sum()) for g in graph_list])


def targets(graph_list):
    return np.concatenate([g.y.numpy() for g in graph_list]).astype(float)


def accuracy():
    """Table 1: RMSE, Pearson, Spearman and Harrell's C (six-seed mean and SD) on the 2,133-complex cohort; the ablation
    row: RMSE and Pearson on all 2,171 test complexes; Pearson r of the five CleanSplit folds on the independent
    CASF-2016 subset (None unless all five checkpoints are there: training CleanSplit is optional); and the size
    control, a straight line from ligand heavy atoms to affinity fitted on the training split."""
    test = graphs("test")
    keep = set(paths.need("cohorts/ids_2133.txt").read_text().split())
    lp2133 = [g for g in test if g.pdb_id.lower() in keep]
    lp2171 = seed_scores(test)
    out = {
        "lp2133": seed_scores(lp2133),
        "lp2171": {k: lp2171[k] for k in ("n", "rmse", "pearson")},
    }

    casf = casf_independent()
    slope, intercept = np.polyfit(
        heavy_atoms(graphs("train")), targets(graphs("train")), 1
    )
    out["size_control"] = {
        **scores(intercept + slope * heavy_atoms(lp2133), targets(lp2133)),
        "casf_r": float(np.corrcoef(heavy_atoms(casf), targets(casf))[0, 1]),
    }

    out["casf"] = {
        "n": len(casf),
        "mean": None,
        "sd": None,
        "minus_rfscore": None,
        "minus_size_control": None,
    }
    if all((paths.CHECKPOINTS / f"{fold}.pt").is_file() for fold in FOLDS):
        r = [
            float(stats.pearsonr(predict(fold, casf), targets(casf)).statistic)
            for fold in FOLDS
        ]
        mean = float(np.mean(r))
        out["casf"].update(
            mean=mean,
            sd=float(np.std(r, ddof=1)),
            minus_rfscore=mean - RF_SCORE_CASF_R,
            minus_size_control=mean - out["size_control"]["casf_r"],
        )
    return out


def external():
    """Pearson r and Harrell's C (six-seed mean and SD) on BDB2020+, Mpro and EGFR."""
    truth = pd.read_csv(paths.need("external/targets.csv"))
    out = {}
    for name, sample in torch.load(
        paths.need("external/graphs.pt"), weights_only=False
    ).items():
        if name == "CASF-2016":
            continue
        y = (
            truth[truth.benchmark == name]
            .drop_duplicates("pdb_id")
            .set_index("pdb_id")
            .target.reindex([g.pdb_id for g in sample])
            .to_numpy()
        )
        ok = np.isfinite(y)
        per_seed = [scores(predict(s, sample)[ok], y[ok]) for s in SEEDS]
        out[name] = {
            "n": int(ok.sum()),
            "r": mean_sd([r["pearson"] for r in per_seed]),
            "ci": mean_sd([r["ci_harrell"] for r in per_seed]),
        }
    return out


def parameters():
    model = load("lp0")
    total = sum(p.numel() for p in model.parameters())
    return {
        "total": total,
        "mlp": sum(
            p.numel() for p in model.pair_readout.interaction_graph.parameters()
        ),
        "schnet_over_total": SCHNET_PARAMETERS / total,
    }


def splits():
    """LP-PDBbind split sizes, before and after the CL1/CL2 and covalent filters; and how many of the filtered training
    labels are IC50 values, and how many are a bound or an estimate (<, >, ~) rather than a single number."""
    table = pd.read_csv(paths.need("lp_pdbbind/LP_PDBBind.csv"))

    def flag(column):
        return table[column].astype(str).str.strip().str.lower().isin({"true", "1"})

    split, covalent = (
        table.new_split.astype(str).str.strip().str.lower(),
        flag("covalent"),
    )
    level = {"train": flag("CL1"), "val": flag("CL2"), "test": flag("CL2")}
    train, label = (split == "train") & level["train"] & ~covalent, table["kd/ki"].astype(str)
    return {
        "raw": {k: int((split == k).sum()) for k in level},
        "filtered": {
            k: int(((split == k) & level[k] & ~covalent).sum()) for k in level
        },
        "train_labels": {
            "ic50": int((train & label.str.lower().str.startswith("ic50")).sum()),
            "bound_or_estimate": int((train & label.str.contains("[<>~]")).sum()),
        },
    }


# stability and robustness


def stability():
    """Across the six seeds, on every test complex: top-5 residue agreement over seed pairs, the cross-seed SD of a
    residue's contribution and how it scales with magnitude, and how often the k-th and (k+1)-th residues are
    separated by less than their noise."""
    claims = {s: residues(infer(s, graphs("test"))[1]) for s in SEEDS}
    matrix = [np.vstack([claims[s][pdb][1] for s in SEEDS]) for pdb in claims[SEEDS[0]]]
    agreement, zero = [], []
    for rows in matrix:
        zero.append(np.mean([float(np.mean(r == 0.0)) for r in rows]))
        k = min(5, rows.shape[1])
        agreement += [
            overlap(rows[i], rows[j], k) for i, j in combinations(range(len(rows)), 2)
        ]
    magnitude = np.concatenate([np.abs(r.mean(0)) for r in matrix])
    sd = np.concatenate([r.std(0, ddof=1) for r in matrix])
    out = {
        "exact5": float(np.mean([a == 1.0 for a in agreement])),
        "overlap5": float(np.nanmean(agreement)),
        "zero_fraction": float(np.nanmean(zero)),
        "mean_sd_pk": float(np.mean(sd)),
        "pearson_magnitude_sd": float(np.corrcoef(magnitude, sd)[0, 1]),
        "mean_abs_claim_pk": float(magnitude.mean()),
        "relative_dispersion": float(sd.mean() / magnitude.mean()),
        "boundaries": {},
    }
    for k in (1, 2, 3, 5, 8, 10):
        gaps, ratios = [], []
        for rows in matrix:
            if rows.shape[1] <= k:
                continue
            mean = np.abs(rows.mean(0))
            order = np.argsort(-mean)
            gap = mean[order[k - 1]] - mean[order[k]]
            noise = np.sqrt(
                rows[:, order[k - 1]].var(ddof=1) + rows[:, order[k]].var(ddof=1)
            )
            gaps.append(gap)
            if noise > 0:
                ratios.append(gap / noise)
        out["boundaries"][str(k)] = [
            len(gaps),
            float(np.median(gaps)),
            float(np.mean(np.array(ratios) < 1)),
        ]
    return out


def rotate_and_translate(pos, ligand, rms, generator):
    """Move the ligand by `rms` A RMS: a rotation about its centroid using half the budget, then a translation."""
    centre = pos[ligand].mean(0)
    centred = pos[ligand] - centre
    axis = torch.randn(3, generator=generator)
    axis = axis / axis.norm()
    lever = float(
        (centred - (centred @ axis)[:, None] * axis).square().sum(1).mean().sqrt()
    )
    angle = 2.0 * math.asin(min(rms * math.sqrt(0.5) / (2.0 * lever), 1.0))
    k = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotated = (
        centred
        @ (torch.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)).T
    )
    achieved = float((rotated - centred).square().sum(1).mean().sqrt())
    step = torch.randn(3, generator=generator)
    moved = pos.clone()
    moved[ligand] = (
        rotated
        + centre
        + step / step.norm() * math.sqrt(max(rms**2 - achieved**2, 0.0))
    )
    return moved


def with_contacts(graph, pos, tag):
    """The complex with new coordinates and its contacts rebuilt within the cutoff."""
    ligand, protein = (
        torch.where(graph.ligand.bool())[0],
        torch.where(~graph.ligand.bool())[0],
    )
    i, j = (torch.cdist(pos[ligand], pos[protein]) < CUTOFF).nonzero(as_tuple=True)
    moved = graph.clone()
    moved.pos, moved.pdb_id, moved.contact_edge_index = (
        pos,
        tag,
        torch.stack([ligand[i], protein[j]]),
    )
    return moved


def ligand_moves(sample):
    """Per complex and amplitude: three pure translations (the same directions at both amplitudes) and four
    translation + rotation draws, as (pdb_id, amplitude, moved complex); moves that lose every contact are dropped."""
    moves = {"translation_only": [], "translation_plus_rotation": []}
    translations = torch.Generator().manual_seed(1906)
    for graph in sample:
        ligand = graph.ligand.bool()
        directions = torch.randn(3, 3, generator=translations)
        directions /= directions.norm(dim=1, keepdim=True)
        for amplitude in AMPLITUDES:
            for direction in directions:
                pos = graph.pos.clone()
                pos[ligand] += amplitude * direction
                moves["translation_only"].append((graph, amplitude, pos))
        for amplitude in AMPLITUDES:
            seed = (
                20260916
                + int.from_bytes(graph.pdb_id.encode(), "little") % 10_000_000
                + (0 if amplitude == 0.10 else 7)
            )
            generator = torch.Generator().manual_seed(seed)
            for _ in range(4):
                pos = rotate_and_translate(graph.pos, ligand, amplitude, generator)
                moves["translation_plus_rotation"].append((graph, amplitude, pos))
    for arm, cloud in moves.items():
        rebuilt = [
            (graph.pdb_id, amplitude, with_contacts(graph, pos, str(i)))
            for i, (graph, amplitude, pos) in enumerate(cloud)
        ]
        moves[arm] = [move for move in rebuilt if move[2].contact_edge_index.shape[1]]
    return moves


def rigid():
    """Six-seed mean and SD of |prediction change|, Exact@5 and Overlap@5 of the top-5 residues when the ligands of 250
    complexes move rigidly by 0.10 or 0.25 A RMS, with the contacts rebuilt."""
    sample = cohort(graphs("test"), 250)
    moves = ligand_moves(sample)
    per_seed = defaultdict(list)
    for name in SEEDS:
        complexes, groups = infer(name, sample)
        base_pk, base_claims = (
            dict(zip(complexes.pdb_id, complexes.prediction)),
            residues(groups),
        )
        for arm, cloud in moves.items():
            complexes, groups = infer(name, [graph for *_, graph in cloud])
            moved_pk, moved_claims = (
                dict(zip(complexes.pdb_id, complexes.prediction)),
                residues(groups),
            )
            rows = defaultdict(list)
            for pdb, amplitude, graph in cloud:
                (base_ids, before), (moved_ids, after) = (
                    base_claims[pdb],
                    moved_claims[graph.pdb_id],
                )
                ids = np.union1d(base_ids, moved_ids)
                a, b = np.zeros(len(ids)), np.zeros(len(ids))
                a[np.searchsorted(ids, base_ids)] = before
                b[np.searchsorted(ids, moved_ids)] = after
                agreement = overlap(a, b, min(5, len(ids)))
                shift = abs(float(moved_pk[graph.pdb_id] - base_pk[pdb]))
                rows[amplitude].append((shift, agreement == 1.0, agreement))
            for amplitude, r in rows.items():
                for metric, value in zip(
                    ("dpk", "exact5", "overlap5"), np.mean(r, axis=0)
                ):
                    per_seed[f"{arm} {amplitude:.2f}", metric].append(float(value))
    out = defaultdict(dict, n=len(sample))
    for (key, metric), values in per_seed.items():
        out[key][metric] = mean_sd(values)
    out["largest_shift_010"] = max(
        out["translation_only 0.10"]["dpk"]["mean"],
        out["translation_plus_rotation 0.10"]["dpk"]["mean"],
    )
    return out


def invariance():
    """Largest prediction change of seed 0 under three random whole-complex rotations and translations, 60 complexes."""
    rng = np.random.default_rng(7)
    sample = cohort(graphs("test"), 250)[:60]
    base, worst = predict("lp0", sample), 0.0
    for _ in range(3):
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        rotation = torch.as_tensor(rotation, dtype=torch.float32)
        shift = torch.as_tensor(rng.normal(size=3) * 10, dtype=torch.float32)
        rotated, shifted = [g.clone() for g in sample], [g.clone() for g in sample]
        for r, s in zip(rotated, shifted):
            r.pos, s.pos = r.pos @ rotation.T, s.pos + shift
        for moved in (rotated, shifted):
            worst = max(worst, float(np.abs(predict("lp0", moved) - base).max()))
    return {"max_dpk": worst}


# faithfulness


@torch.inference_mode()
def contact_claims(name, sample):
    """{pdb_id: each contact's contribution in pK}, contacts in graph order."""
    model = load(name)
    claims = {}
    for batch in DataLoader(sample, 32):
        out = model(batch)
        contact, graph = (
            (out["contact"] * model.std).numpy(),
            out["contact_graph"].numpy(),
        )
        for i, pdb in enumerate(batch.pdb_id):
            claims[pdb] = contact[graph == i]
    return claims


def nearest_residues(graph):
    """The residues in contact with the ligand, closest first."""
    ligand_atom, protein_atom = graph.contact_edge_index
    distance = (graph.pos[ligand_atom] - graph.pos[protein_atom]).norm(dim=1)
    closest = {}
    for residue, d in zip(graph.residue_id[protein_atom].tolist(), distance.tolist()):
        closest[residue] = min(d, closest.get(residue, d))
    return [residue for residue, _ in sorted(closest.items(), key=lambda item: item[1])]


def contact_residues(graph):
    return np.unique(graph.residue_id[graph.contact_edge_index[1]].numpy()).tolist()


def single_contact_deletion(name, sample, base, rng):
    """Delete the 4 strongest and 4 random contacts of each complex, one at a time."""
    claims = contact_claims(name, sample)
    jobs, meta = [], []
    for graph in sample:
        n, claim = graph.contact_edge_index.shape[1], claims[graph.pdb_id]
        if n < 2:
            continue
        strong = np.argsort(-np.abs(claim))[:4]
        rest = np.setdiff1d(np.arange(n), strong)
        for edge in np.concatenate(
            [strong, rng.choice(rest, size=min(4, len(rest)), replace=False)]
        ):
            changed = graph.clone()
            changed.contact_edge_index = graph.contact_edge_index[
                :, torch.arange(n) != int(edge)
            ]
            jobs.append(changed)
            meta.append((graph.pdb_id, float(claim[edge])))
    observed = [base[pdb] - after for (pdb, _), after in zip(meta, predict(name, jobs))]
    return {**fidelity([c for _, c in meta], observed), "n": len(meta)}


def residue_deletion(name, sample, base, rng, nearest):
    """Delete the top-k residues by claim, k random ones and the k nearest ones. The draws for 10/25/50% subsets are
    unused but kept, so the random stream matches the paper's runs."""
    claims = residues(infer(name, sample)[1])
    jobs, meta = [], []
    for graph in sample:
        present = contact_residues(graph)
        if len(present) < 2:
            continue
        ids, claim = claims[graph.pdb_id]
        order = np.argsort(-np.abs(claim))
        ids, claim = ids[order], claim[order]
        claim_of = dict(zip(ids, claim))
        for k in K_VALUES:
            if k > len(ids):
                continue
            random = list(rng.choice(present, size=min(k, len(present)), replace=False))
            near = nearest[graph.pdb_id][:k]
            for rule, chosen, total in (
                ("top", list(ids[:k]), claim[:k].sum()),
                ("random", random, sum(claim_of.get(r, 0.0) for r in random)),
                ("nearest", near, sum(claim_of.get(r, 0.0) for r in near)),
            ):
                jobs.append(drop(graph, chosen))
                meta.append((graph.pdb_id, rule, k, float(total)))
        for fraction in (0.10, 0.25, 0.50):
            rng.choice(
                present, size=max(1, int(round(fraction * len(present)))), replace=False
            )
    observed = np.array(
        [base[m[0]] - after for m, after in zip(meta, predict(name, jobs))]
    )
    claimed, rule, size = (
        np.array([m[3] for m in meta]),
        np.array([m[1] for m in meta]),
        np.array([m[2] for m in meta]),
    )
    top_rule = rule == "top"
    out = {
        "pooled": {
            **fidelity(claimed[top_rule], observed[top_rule]),
            "effect": float(np.abs(observed[top_rule]).mean()),
            "n": int(top_rule.sum()),
        }
    }
    for k in K_VALUES:
        at_k = size == k
        out[k] = {
            **fidelity(claimed[top_rule & at_k], observed[top_rule & at_k]),
            **{
                r: float(np.abs(observed[(rule == r) & at_k]).mean())
                for r in ("top", "random", "nearest")
            },
        }
    return out


def occlusion_deletion(name, sample, base, nearest):
    """Rank residues by their own single-residue deletion effect, then delete the top k or the k nearest jointly."""
    solo = [(graph, residue) for graph in sample for residue in contact_residues(graph)]
    effect = defaultdict(dict)
    for (graph, residue), prediction in zip(
        solo, predict(name, [drop(graph, [residue]) for graph, residue in solo])
    ):
        effect[graph.pdb_id][residue] = float(base[graph.pdb_id] - prediction)
    jobs, meta = [], []
    for graph in sample:
        if len(effect[graph.pdb_id]) < 2:
            continue
        ids = np.array(sorted(effect[graph.pdb_id]), dtype=np.int64)
        values = np.array([effect[graph.pdb_id][int(i)] for i in ids])
        order = np.argsort(-np.abs(values))
        ids, values = ids[order], values[order]
        effect_of = dict(zip(ids.tolist(), values))
        for k in K_VALUES:
            if k > len(ids):
                continue
            near = nearest[graph.pdb_id][:k]
            for rule, chosen, total in (
                ("top", list(ids[:k]), float(values[:k].sum())),
                ("nearest", near, float(sum(effect_of.get(int(r), 0.0) for r in near))),
            ):
                jobs.append(drop(graph, chosen))
                meta.append((graph.pdb_id, rule, k, total))
    observed = np.array(
        [base[m[0]] - after for m, after in zip(meta, predict(name, jobs))]
    )
    claimed, rule, size = (
        np.array([m[3] for m in meta]),
        np.array([m[1] for m in meta]),
        np.array([m[2] for m in meta]),
    )
    out = {}
    for k in K_VALUES:
        top_k, near_k = (rule == "top") & (size == k), (rule == "nearest") & (size == k)
        out[k] = {
            "top": float(np.abs(observed[top_k]).mean()),
            "nearest": float(np.abs(observed[near_k]).mean()),
            "over_claim_pk": fidelity(claimed[top_k], observed[top_k])["over_claim_pk"],
        }
    return out


def interventions():
    """Delete what the model credits and compare the prediction change with the claim, six seeds on 250 complexes:
    single contacts, and the top-k residues against k random and the k nearest ones, ranked by the model (native) or
    by each residue's own deletion effect (occlusion)."""
    sample = cohort(graphs("test"), 250)
    nearest = {graph.pdb_id: nearest_residues(graph) for graph in sample}
    runs = []
    for name in SEEDS:
        rng = np.random.default_rng(
            20260912
        )  # the same draws for every seed: single contacts first, then residues
        base = dict(zip([g.pdb_id for g in sample], predict(name, sample)))
        single = single_contact_deletion(name, sample, base, rng)
        native = residue_deletion(name, sample, base, rng, nearest)
        occlusion = occlusion_deletion(name, sample, base, nearest)
        runs.append({"single": single, "native": native, "occlusion": occlusion})

    def across(values):
        return mean_sd(list(values))

    pooled = [run["native"]["pooled"] for run in runs]
    metrics = (
        "calibration_slope",
        "normalised_infidelity",
        "over_claim_pk",
        "sign_agreement",
    )
    out = {
        "n": len(sample),
        "single_contact": {
            m: across(run["single"][m] for run in runs) for m in metrics
        },
        "pooled": {
            "effect": across(p["effect"] for p in pooled),
            "discrepancy": across(p["over_claim_pk"] for p in pooled),
            "relative": across(p["over_claim_pk"] / p["effect"] for p in pooled),
            "slope": across(p["calibration_slope"] for p in pooled),
            "sign": across(p["sign_agreement"] for p in pooled),
            "n": pooled[0]["n"],
        },
        "top_five": {
            m: across(run["native"][5][m] for run in runs)
            for m in ("calibration_slope", "normalised_infidelity", "over_claim_pk")
        },
        "k5_top_effect": across(run["native"][5]["top"] for run in runs),
    }
    out["single_contact"]["n"] = runs[0]["single"]["n"]
    for arm in ("native", "occlusion"):
        out[arm] = {
            k: {
                "top_over_nearest": across(
                    run[arm][k]["top"] / run[arm][k]["nearest"] for run in runs
                )
            }
            for k in K_VALUES
        }
        out[arm][5].update(  # the paper prints these two at k=5 only
            top_over_random=across(
                run[arm][5]["top"] / run["native"][5]["random"] for run in runs
            ),
            over_claim=across(run[arm][5]["over_claim_pk"] for run in runs),
        )
    out["pooled"]["rfscore_over_discrepancy"] = (
        RF_SCORE_DISCREPANCY / out["pooled"]["discrepancy"]["mean"]
    )
    out["pooled"]["slope_minus_cheapnet"] = (
        out["pooled"]["slope"]["mean"] - CHEAPNET_SLOPE
    )
    return out


def deletion_curve():
    """Seed 0, first 120 complexes of the 250 cohort: area under the prediction as the top-ranked residues are deleted
    (0, 20, ..., 100%), scaled so the intact complex is 1 and the fully deleted one 0; and the same for insertion."""
    sample = cohort(graphs("test"), 250)
    claims = residues(infer("lp0", sample)[1])
    sample = [g for g in sample[:120] if g.pdb_id in claims]
    deleted, inserted = [], []
    for graph in sample:
        ids, claim = claims[graph.pdb_id]
        ids = ids[np.argsort(-np.abs(claim))]
        for fraction in np.linspace(0, 1, 6):
            k = int(round(fraction * len(ids)))
            deleted.append(drop(graph, ids[:k]))
            inserted.append(drop(graph, ids[k:]))
    deletion = predict("lp0", deleted).reshape(len(sample), 6)
    insertion = predict("lp0", inserted).reshape(len(sample), 6)

    def area(values, floor):
        values = np.asarray(values, float)
        return float(np.trapezoid((values - floor) / (values[0] - floor), dx=0.2))

    deletion_area = np.array([area(d, d[-1]) for d in deletion])
    insertion_area = np.array(
        [area(i[::-1], d[-1]) for i, d in zip(insertion, deletion)]
    )
    normaliser = deletion[:, 0] - deletion[:, -1]
    return {
        "n": len(deletion_area),
        "mean": float(deletion_area.mean()),
        "median": float(np.median(deletion_area)),
        "sd": float(deletion_area.std(ddof=1)),
        "negative_area_frac": float((deletion_area < 0).mean()),
        "negative_normaliser_count": int((normaliser <= 0).sum()),
        "smallest_normaliser_pk": float(np.abs(normaliser).min()),
        "interaction_term": float(deletion_area.mean() - np.nanmean(insertion_area)),
    }


def selection():
    """Seed 0, first 250 test complexes: the attributable part left after deleting, or keeping only, the top, next or
    random 5-30% of residues (five random draws)."""
    sample = graphs("test")[:250]
    claims = residues(infer("lp0", sample)[1])
    base = dict(zip([g.pdb_id for g in sample], predict("lp0", sample)))
    rng = np.random.default_rng(20260921)
    jobs, meta, smallest = [], [], np.inf
    for graph in sample:
        ids, claim = claims[graph.pdb_id]
        ranked, attributable = ids[np.argsort(-np.abs(claim))], float(claim.sum())
        smallest = min(smallest, attributable)
        for f in (0.05, 0.10, 0.20, 0.30):
            n = math.ceil(f * len(ids))
            picks = [("", ranked[:n]), ("_next", ranked[n : 2 * n])]
            picks += [
                ("_random", rng.choice(ranked, size=n, replace=False)) for _ in range(5)
            ]
            for name, chosen in picks:
                for mode, dropped in (
                    ("delete", chosen),
                    ("preserve", np.setdiff1d(ids, chosen)),
                ):
                    jobs.append(drop(graph, dropped))
                    meta.append((graph.pdb_id, str(f), mode + name, attributable))
    cells = defaultdict(lambda: defaultdict(list))
    for (pdb, f, column, attributable), p in zip(meta, predict("lp0", jobs)):
        cells[f, column][pdb].append(1 - (base[pdb] - p) / attributable)
    table = defaultdict(dict)
    for (f, column), per_complex in cells.items():
        table[f][column] = float(np.mean([np.mean(v) for v in per_complex.values()]))
    return {
        "n": len(sample),
        "table": table,
        "min_A_pk": smallest,
        "text_pct": {
            k: 100 * (1 - table["0.05"]["delete" + s])
            for k, s in (("top", ""), ("next", "_next"), ("random", "_random"))
        },
    }


def sparsity():
    """Seed 0, 1,860 complexes: how concentrated |contribution| is over (ligand atom, residue) groups."""
    keep = set(paths.need("cohorts/sparsity_1860.txt").read_text().split())
    groups = infer("lp0", [g for g in graphs("test") if g.pdb_id in keep])[1]
    rows = []
    for _, complex_groups in groups.groupby("pdb_id", sort=False):
        mass = np.abs(complex_groups.claim.to_numpy(float))
        if len(mass) < 2 or mass.sum() <= 0:
            continue
        share = mass / mass.sum()
        entropy = -np.sum(share[share > 0] * np.log(share[share > 0]))
        cumulative = np.cumsum(np.sort(share)[::-1])
        rows.append(
            [
                len(mass),
                entropy / np.log(len(mass)),
                np.exp(entropy) / len(mass),
                *(
                    (np.searchsorted(cumulative, q) + 1) / len(mass)
                    for q in (0.5, 0.8, 0.9)
                ),
                mass.sum(),
            ]
        )
    names = (
        "n_residues",
        "entropy_normalised",
        "effective_support",
        "f50",
        "f80",
        "f90",
        "mass_pk",
    )
    return dict(zip(names, map(float, np.mean(rows, axis=0))))


def share():
    """Six-seed mean over test complexes of |physics| / |prediction - baseline|."""
    per_seed = []
    for name in SEEDS:
        c = infer(name, graphs("test"))[0]
        per_seed.append(
            float(
                (
                    c.physics.abs() / (c.prediction - c.baseline).abs().clip(lower=1e-9)
                ).mean()
            )
        )
    return {"share": float(np.mean(per_seed))}


# chemistry


def roc_auc(score, positive):
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n, m = positive.sum(), (~positive).sum()
    return float((stats.rankdata(score)[positive].sum() - n * (n + 1) / 2) / (n * m))


def average_precision(score, positive):
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    hit = positive[np.lexsort((np.arange(len(score)), -score))]
    return float(
        (np.cumsum(hit) / np.arange(1, len(hit) + 1) * hit).sum() / positive.sum()
    )


@lru_cache(maxsize=1)
def residue_map():
    """{pdb_id: {graph residue id: "chain:RES:number:..."}}."""
    return json.loads(paths.need("prolif/residue_map.json").read_text())


def candidate_pairs(graph):
    """{(ligand atom ordinal, residue id): shortest contact distance} for every pair joined by a contact."""
    ordinal = {
        int(atom): i for i, atom in enumerate(torch.where(graph.ligand.bool())[0])
    }
    ligand_atom, protein_atom = graph.contact_edge_index
    distance = (graph.pos[ligand_atom] - graph.pos[protein_atom]).norm(dim=1)
    pairs = {}
    for atom, residue, d in zip(
        ligand_atom.tolist(), graph.residue_id[protein_atom].tolist(), distance.tolist()
    ):
        key = (ordinal[atom], residue)
        pairs[key] = min(d, pairs.get(key, d))
    return pairs


def prolif_labels(graph, keys):
    """{family: whether ProLIF detects that interaction for each candidate pair}."""
    labels = {family: np.zeros(len(keys), dtype=bool) for family in FAMILIES}
    files = sorted(paths.need("prolif/labels/test").glob(f"{graph.pdb_id}-*.csv.gz"))
    if not files:
        return labels
    residue_of = {
        name: int(residue)
        for residue, name in residue_map().get(graph.pdb_id, {}).items()
    }
    position = {key: i for i, key in enumerate(keys)}
    events = pd.read_csv(files[0])
    for family, atom, residue in zip(
        events.mapped_family, events.ligand_atom_index, events.stable_residue_id
    ):
        i = position.get((int(atom), residue_of.get(residue)))
        if family in labels and i is not None:
            labels[family][i] = True
    return labels


def prolif(sample, groups):
    """Over the candidate pairs of the sample: each pair's row in `groups` (-1 if none), its shortest contact
    distance, and its ProLIF label per family."""
    row = {
        (pdb, int(atom), int(residue)): i
        for i, (pdb, atom, residue) in enumerate(
            zip(groups["pdb_id"], groups["ligand_atom"], groups["residue_id"])
        )
    }
    rows, distance, labels = [], [], defaultdict(list)
    for graph in sample:
        pairs = candidate_pairs(graph)
        keys = sorted(pairs)
        rows += [row.get((graph.pdb_id, *key), -1) for key in keys]
        distance += [pairs[key] for key in keys]
        for family, found in prolif_labels(graph, keys).items():
            labels[family].append(found)
    return (
        np.array(rows),
        np.array(distance),
        {family: np.concatenate(v) for family, v in labels.items()},
    )


def chemistry():
    """Seed 0, 600 complexes: AUROC of each named channel's |contribution|, and of closeness, against its ProLIF
    interaction family over the candidate pairs. The model never sees an interaction label."""
    keep = set(paths.need("cohorts/ids_pignet_intersect_1862.txt").read_text().split())
    sample = cohort([g for g in graphs("test") if g.pdb_id.lower() in keep], 600)
    groups = infer("lp0", sample)[1]
    rows, distance, labels = prolif(sample, groups)
    out = {"n_pairs": len(distance), "n_complexes": len(sample)}
    for family in FAMILIES:
        if labels[family].sum():
            score = np.where(rows >= 0, np.abs(groups[family].to_numpy()[rows]), 0.0)
            out[family] = {
                "pact": roc_auc(score, labels[family]),
                "distance": roc_auc(-distance, labels[family]),
            }
    high = (
        "aromatic_proximity",
        "cationic_aromatic_proximity",
        "halogen_acceptor_proximity",
        "hydrophobic",
        "metal",
    )
    out["lowest_high"] = min(out[f]["pact"] for f in high)
    return out


@lru_cache(maxsize=None)
@torch.inference_mode()
def probe(name):
    """Per group of the 600-complex cohort: the raw basis (raw), the local coefficient (loc), the model's own channel
    contributions (shp) and three ledgers with corrections removed: all of them (fix), all but the local coefficient
    (lco), all but the contact context and group MLP (cco). Per contact: the context multiplier and the basis."""
    model = load(name)
    weights = model.pair_readout.physical_weights().numpy()[None]
    std = model.std
    groups, contacts = defaultdict(list), defaultdict(list)
    for batch in DataLoader(cohort(graphs("test"), 600), 32):
        out = model(batch)
        basis, local = (
            out["basis"].numpy(),
            np.asarray(out["local_coef"].numpy(), float),
        )
        scale = np.ones_like(basis)
        scale[:, 0] = out["vdw_share"].numpy()
        groups["pdb_id"].append(
            np.array([batch.pdb_id[i] for i in out["group_graph"].tolist()])
        )
        groups["residue_id"].append(out["group_residue"].numpy())
        groups["ligand_atom"].append(ligand_ordinal(batch)[out["group_atom"]].numpy())
        groups["raw"].append(basis)
        groups["loc"].append(local)
        groups["shp"].append((out["channels"] * std).numpy())
        groups["fix"].append(basis * weights * scale * std)
        groups["lco"].append(basis * weights * local * scale * std)
        groups["cco"].append(out["scaled"].numpy() * weights * scale * std)
        contacts["ctx"].append(out["context"].numpy())
        contacts["basis"].append(out["contact_basis"].numpy())
    return {k: np.concatenate(v) for k, v in groups.items()}, {
        k: np.concatenate(v) for k, v in contacts.items()
    }


def range_use(values, basis, half_width):
    """Over entries with a nonzero basis: live fraction, mean, SD, and the 5-95% span as a fraction of the range."""
    live = np.abs(basis) > 1e-6
    v = np.asarray(values, float)[live]
    p5, p95 = np.percentile(v, [5, 95])
    return {
        "live_frac": live.mean(),
        "mean": v.mean(),
        "sd": v.std(),
        "span90_over_range": (p95 - p5) / (2 * half_width),
    }


def coefficient_use():
    """Per channel, six-seed mean of how much of its range the contact multiplier and the local coefficient use, over
    the 600-complex cohort."""
    per_seed = defaultdict(list)
    for name in SEEDS:
        groups, contacts = probe(name)
        for j, channel in enumerate(CHANNELS):
            sides = {
                "contact": range_use(
                    contacts["ctx"][:, j], contacts["basis"][:, j], CONTEXT_RANGE
                ),
                "local": range_use(groups["loc"][:, j], groups["raw"][:, j], 1.0),
            }
            for side, use in sides.items():
                for key, value in use.items():
                    per_seed[channel, f"{side}_{key}"].append(float(value))
    out = defaultdict(dict, n=len(np.unique(groups["pdb_id"])))
    for (channel, key), values in per_seed.items():
        out[channel][key] = float(np.mean(values))
    return out


@torch.inference_mode()
def corrections():
    """Seed 0, 250 random test complexes: median local coefficient over the groups with a nonzero basis, for the four
    channels the paper names; the contact count covers every channel."""
    test = graphs("test")
    sample = [
        test[i]
        for i in sorted(
            np.random.default_rng(0).choice(len(test), size=250, replace=False)
        )
    ]
    model = load("lp0")
    weights = model.pair_readout.physical_weights().numpy()[None]
    coefficients = defaultdict(list)
    for batch in DataLoader(sample, 1):
        out = model(batch)
        live = np.abs(out["basis"].numpy() * weights) > 1e-9
        for j, channel in enumerate(CHANNELS):
            coefficients[channel].append(out["local_coef"].numpy()[live[:, j], j])
    coefficients = {c: np.concatenate(v) for c, v in coefficients.items()}
    return {
        "n_complexes": len(sample),
        "n_rows": sum(len(v) for v in coefficients.values()),
        "median": {
            c: float(np.median(coefficients[c].astype(float)))
            for c in ("apolar_area_times_occlusion", "vdw", "hydrophobic", "aromatic_proximity")
        },
    }


def rerank(groups, against):
    """How often the model's residue ranking differs from the `against` ledger's, over complexes with more than five
    residues: (complexes, top-1 changed, top-5 set changed, top-5 Jaccard, Spearman rho)."""
    frame = pd.DataFrame(
        {
            "pdb_id": groups["pdb_id"],
            "residue_id": groups["residue_id"],
            "model": groups["shp"].sum(axis=1),
            "other": groups[against].sum(axis=1),
        }
    )
    per_residue = (
        frame.groupby(["pdb_id", "residue_id"], sort=False).sum().reset_index()
    )
    top1, top5, jaccard, spearman = [], [], [], []
    for _, rows in per_residue.groupby("pdb_id", sort=False):
        if len(rows) < 6:
            continue
        by_model, by_other = (
            rows.sort_values("model", ascending=False),
            rows.sort_values("other", ascending=False),
        )
        a, b = (
            set(by_model.residue_id.to_numpy()[:5]),
            set(by_other.residue_id.to_numpy()[:5]),
        )
        top1.append(int(by_model.residue_id.iloc[0] != by_other.residue_id.iloc[0]))
        top5.append(int(a != b))
        jaccard.append(len(a & b) / len(a | b))
        rho = pd.Series(rows.model.to_numpy()).corr(
            pd.Series(rows.other.to_numpy()), method="spearman"
        )
        spearman.append(float(rho) if np.isfinite(rho) else 1.0)
    return len(top1), np.mean(top1), np.mean(top5), np.mean(jaccard), np.mean(spearman)


def reranking():
    """How often removing corrections changes the residue ranking (see probe), six-seed mean and SD."""
    out = {}
    for against in ("fix", "lco", "cco"):
        per_seed = np.array([rerank(probe(name)[0], against) for name in SEEDS])
        out["n"] = int(per_seed[0, 0])  # the same complexes for every ledger
        out[against] = {
            "top1": per_seed[:, 1].mean(),
            "top1_sd": per_seed[:, 1].std(),
            "top5": per_seed[:, 2].mean(),
            "top5_sd": per_seed[:, 2].std(),
            "jaccard": per_seed[:, 3].mean(),
            "spearman": per_seed[:, 4].mean(),
        }
    return out


def raw_basis():
    """Six-seed mean average precision of each channel's raw basis, and of the ledgers of probe, against its ProLIF
    family over the candidate pairs of the 600-complex cohort, with closeness and random scores as nulls."""
    sample = cohort(graphs("test"), 600)
    rows, distance, labels = prolif(sample, probe("lp0")[0])
    random = np.random.default_rng(0).random(len(distance))
    per_seed = defaultdict(list)
    for name in SEEDS:
        groups = probe(name)[0]
        for family in FAMILIES:
            positive, j = labels[family], CHANNELS.index(family)
            if not positive.sum():
                continue
            for column, key in (
                ("raw", "raw_basis"),
                ("fix", "fixed"),
                ("lco", "local_only"),
                ("cco", "context_only"),
            ):
                score = np.where(rows >= 0, np.abs(groups[column][rows, j]), 0.0)
                per_seed[family, key].append(average_precision(score, positive))
            per_seed[family, "distance_null"].append(
                average_precision(-distance, positive)
            )
            per_seed[family, "random_null"].append(average_precision(random, positive))
    out = defaultdict(dict, n_pairs=len(distance), n_complexes=len(sample))
    for (family, key), values in per_seed.items():
        out[family][key] = float(np.mean(values))
        out[family]["n_positive"] = int(labels[family].sum())
    return out


def recall(pose, reference, family=None):
    """Fraction of the reference pose's ProLIF interactions (of one family, or all) that the pose recovers."""
    pose = {e for e in pose if family is None or e[0] == family}
    reference = {e for e in reference if family is None or e[0] == family}
    return len(pose & reference) / len(reference) if reference else np.nan


def preference(pairs, score):
    """Fraction of (better, worse) pose pairs that `score` orders the same way, ties counting 0.5."""
    wins = [
        0.5 if score[a] == score[b] else float(score[a] > score[b])
        for a, b in pairs
        if a in score and b in score and np.isfinite(score[a]) and np.isfinite(score[b])
    ]
    return float(np.mean(wins))


def poses():
    """Redocked CASF-2016 poses, pairs within 0.5 A RMSD whose ProLIF fingerprint recovery differs by at least 0.2:
    how often the six-seed score, the contact count or -RMSD prefers the better pose; and per interaction family
    (1,500 pairs at most, seed 0) the same for the named channel, the total and the contact count."""
    prints = defaultdict(set)
    for pdb, code, family, atom, residue in pd.read_csv(
        paths.need("poses/pose_plif.csv.gz")
    ).itertuples(index=False):
        prints[code].add((family, int(atom), residue))
    rmsd = pd.read_csv(paths.need("poses/casf_rmsd.csv"))
    rmsd = dict(zip(rmsd.code, rmsd.rmsd))
    folder = paths.need("poses/casf_docking")
    ids = sorted(p.stem for p in folder.glob("*.pt"))

    def load_poses(pdb):
        return torch.load(folder / f"{pdb}.pt", weights_only=False)["graphs"]

    score, contacts, recovered, all_pairs, targets = {}, {}, {}, [], 0
    for pdb in ids:
        reference = prints.get(f"{pdb}_ligand")
        pose_graphs = [
            g
            for g in load_poses(pdb)
            if g.pdb_id != f"{pdb}_ligand" and g.pdb_id in rmsd
        ]
        codes = [g.pdb_id for g in pose_graphs]
        if not reference or len(codes) < 4:
            continue
        recovered.update({c: recall(prints.get(c, set()), reference) for c in codes})
        contacts.update(
            {g.pdb_id: int(g.contact_edge_index.shape[1]) for g in pose_graphs}
        )
        score.update(zip(codes, mean_seed_prediction(pose_graphs)))
        pairs = [
            (a, b) if recovered[a] > recovered[b] else (b, a)
            for i, a in enumerate(codes)
            for b in codes[i + 1 :]
            if abs(rmsd[a] - rmsd[b]) <= 0.5 and abs(recovered[a] - recovered[b]) >= 0.2
        ]
        targets += bool(pairs)
        all_pairs += pairs
    out = {
        "n_poses": len(recovered),
        "n_targets": targets,
        "n_pairs": len(all_pairs),
        "total": preference(all_pairs, score),
        "contact_count": preference(all_pairs, contacts),
        "rmsd": preference(all_pairs, {c: -rmsd[c] for c in recovered}),
    }

    family_pairs = defaultdict(list)
    for pdb in ids:
        reference = prints.get(f"{pdb}_ligand")
        codes = [
            c
            for c in prints
            if c.startswith(pdb + "_") and c in rmsd and c != f"{pdb}_ligand"
        ]
        if not reference or len(codes) < 4:
            continue
        rec = {
            c: {f: recall(prints.get(c, set()), reference, f) for f in FAMILIES}
            for c in codes
        }
        for i, a in enumerate(codes):
            for b in codes[i + 1 :]:
                if abs(rmsd[a] - rmsd[b]) > 0.5:
                    continue
                for f in FAMILIES:
                    if (
                        np.isfinite(rec[a][f])
                        and np.isfinite(rec[b][f])
                        and abs(rec[a][f] - rec[b][f]) >= 0.2
                    ):
                        family_pairs[f].append(
                            (a, b) if rec[a][f] > rec[b][f] else (b, a)
                        )
    rng = np.random.default_rng(20260912)
    for f in list(family_pairs):
        if len(family_pairs[f]) > 1500:
            family_pairs[f] = [
                family_pairs[f][i] for i in rng.permutation(len(family_pairs[f]))[:1500]
            ]
    needed = defaultdict(set)
    for pairs in family_pairs.values():
        for pair in pairs:
            for code in pair:
                needed[code.rsplit("_", 1)[0]].add(code)
    mass, total, contacts = defaultdict(dict), {}, {}
    for pdb, wanted in sorted(needed.items()):
        pose_graphs = [g for g in load_poses(pdb) if g.pdb_id in wanted]
        contacts.update(
            {g.pdb_id: int(g.contact_edge_index.shape[1]) for g in pose_graphs}
        )
        complexes, groups = infer("lp0", pose_graphs)
        for family in FAMILIES:
            mass[family].update(groups.groupby("pdb_id")[family].sum())
        total.update(zip(complexes.pdb_id, complexes.prediction))
    for f, pairs in family_pairs.items():
        out[f] = [
            preference(pairs, mass[f]),
            preference(pairs, total),
            preference(pairs, contacts),
        ]
    return out


def mmp():
    """3D matched molecular pairs: whether the six-seed mean prediction orders each pair like the measured affinities,
    overall and by stratum; and how far our graphs of shared structures predict from the cached LP-PDBbind ones."""
    mmp_graphs = torch.load(paths.need("mmp/mmp3d_graphs.pt"), weights_only=False)
    by_id = {g.pdb_id: g for g in mmp_graphs}
    mean = dict(zip([g.pdb_id for g in mmp_graphs], mean_seed_prediction(mmp_graphs)))
    trained = {g.pdb_id for g in graphs("train")}
    records = json.loads(paths.need("mmp/pairs.json").read_text())
    left, right = (
        [r["pdb_L"].lower() for r in records],
        [r["pdb_R"].lower() for r in records],
    )
    observed = np.array([r["delta_pK"] for r in records])
    correct = np.sign(
        np.array([mean[b] - mean[a] for a, b in zip(left, right)])
    ) == np.sign(observed)
    nonzero = observed != 0
    cluster = np.array([r["cluster"] for r in records])
    member = np.array([a in trained or b in trained for a, b in zip(left, right)])
    heavy = {pdb: int(g.ligand.bool().sum()) for pdb, g in by_id.items()}
    same_size = np.array([heavy[a] == heavy[b] for a, b in zip(left, right)])
    sub_micromolar = np.array(
        [r["aff_L_uM"] < 1 and r["aff_R_uM"] < 1 for r in records]
    )

    out = {
        "n_structures": len({pdb for r in records for pdb in (r["pdb_L"], r["pdb_R"])}),
        "train_touching": int(member.sum()),
    }
    strata = {
        "all": np.ones(len(records), bool),
        "no_train_member": ~member,
        "equal_heavy_atoms": same_size,
        "cliffs_ge_1pK": np.abs(observed) >= 1.0,
    }
    for name, keep in strata.items():
        out[name] = {
            "n": int(keep.sum()),
            "n_dir": int((keep & nonzero).sum()),
            "clusters": len(set(cluster[keep])),
            "direction": float(correct[keep & nonzero].mean()),
        }
    for threshold in (0.5, 1.0):
        keep = sub_micromolar & (np.abs(observed) >= threshold) & nonzero
        out[f"ge_{threshold}"] = {
            "n": int(keep.sum()),
            "hits": int(correct[keep].sum()),
            "pct": 100 * float(correct[keep].mean()),
        }

    cached = {g.pdb_id: g for split in ("train", "val", "test") for g in graphs(split)}
    shared = sorted(set(by_id) & set(cached))
    ours = [by_id[p] for p in shared]
    difference = np.abs(
        predict("lp0", ours) - predict("lp0", [cached[p] for p in shared])
    )
    out["two_pipeline"] = {
        "median_abs_diff": float(np.median(difference)),
        "p95_abs_diff": float(np.percentile(difference, 95)),
        "mean_seed_sd": float(
            np.stack([predict(s, ours) for s in SEEDS]).std(0, ddof=1).mean()
        ),
    }
    return out


def protonation():
    """Alternative protonation states of the same complexes: [complexes, mean |shift| of the six-seed mean prediction,
    Exact@5, Overlap@5 of the seed-0 top-5 residues] per state change."""
    built = torch.load(paths.need("protonation/states.pt"), weights_only=False)
    ids = sorted(built)
    rebuilt = [built[p]["rebuilt"] for p in ids]
    base = dict(zip(ids, mean_seed_prediction(rebuilt)))
    base_claims = residues(infer("lp0", rebuilt)[1])

    def top5(claims):
        ids_, values = claims
        return set(ids_[np.argsort(-np.abs(values), kind="stable")[:5]])

    out = {}
    for state in sorted({name for p in ids for name in built[p]} - {"rebuilt"}):
        subset = [p for p in ids if state in built[p]]
        if len(subset) < 12:
            continue
        alternative = [built[p][state] for p in subset]
        shift = np.abs(
            mean_seed_prediction(alternative) - np.array([base[p] for p in subset])
        )
        claims = residues(infer("lp0", alternative)[1])
        pairs = [(top5(base_claims[p]), top5(claims[p])) for p in subset]
        out[state] = [
            len(subset),
            float(shift.mean()),
            float(np.mean([float(a == b) for a, b in pairs])),
            float(np.mean([len(a & b) / max(1, len(a)) for a, b in pairs])),
        ]
    return out


def example():
    """The worked example 1c7e, seed 0: prediction and observed affinity, the five largest residue contributions each
    with the prediction change when its contacts are removed, and the seed-0 test RMSE."""
    test = graphs("test")
    graph = next(g for g in test if g.pdb_id == "1c7e")
    complexes, groups = infer("lp0", [graph])
    ids, claims = residues(groups)["1c7e"]
    full = float(predict("lp0", [graph])[0])
    out = {
        "prediction": float(complexes.prediction.iloc[0]),
        "observed": float(complexes.target.iloc[0]),
    }
    out["error"] = out["prediction"] - out["observed"]
    for i in np.argsort(-np.abs(claims))[:5]:
        _, amino_acid, number, *_ = residue_map()["1c7e"][str(ids[i])].split(":")
        removal = full - float(predict("lp0", [drop(graph, [int(ids[i])])])[0])
        out[amino_acid.title() + number] = [float(claims[i]), removal]
    out["tyr100_gap"] = abs(out["Tyr100"][0] - out["Tyr100"][1])
    out["gly61_gap"] = abs(out["Gly61"][0] - out["Gly61"][1])
    y = np.concatenate([g.y.numpy() for g in test])
    out["rmse"] = float(np.sqrt(np.mean((predict("lp0", test).astype(float) - y) ** 2)))
    return out


PORTS = {
    "accuracy": accuracy,
    "external": external,
    "parameters": parameters,
    "splits": splits,
    "stability": stability,
    "rigid": rigid,
    "invariance": invariance,
    "interventions": interventions,
    "deletion_curve": deletion_curve,
    "selection": selection,
    "sparsity": sparsity,
    "share": share,
    "chemistry": chemistry,
    "coefficient_use": coefficient_use,
    "corrections": corrections,
    "reranking": reranking,
    "raw_basis": raw_basis,
    "poses": poses,
    "mmp": mmp,
    "protonation": protonation,
    "example": example,
}
