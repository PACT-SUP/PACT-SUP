"""
PACT: Protein–ligand Affinity from Contact Types source code.
"""

import math
from functools import lru_cache

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

import paths

# the twelve channels, in checkpoint order
CHANNELS = (
    "vdw",
    "repulsion",
    "hydrophobic",
    "hbond",
    "ionic",
    "aromatic_proximity",
    "halogen_acceptor_proximity",
    "metal",
    "cationic_aromatic_proximity",
    "apolar_area_times_occlusion",
    "polar_area_times_occlusion",
    "zinc_coordination",
)
INITIAL_WEIGHTS = (
    0.02,
    0.05,
    0.02,
    0.05,
    0.05,
    0.04,
    0.03,
    0.10,
    0.04,
    0.02,
    0.05,
    0.10,
)

# columns of graph.atom_features, and graph.node_type values
CHARGE, AROMATIC, DONOR, ACCEPTOR, HYDROPHOBE, POSITIVE, NEGATIVE, HALOGEN = (
    2,
    4,
    6,
    7,
    8,
    9,
    10,
    11,
)
ATOM_FEATURES = 12
PROTEIN, METAL = 0, 3
AA_TYPES = 22
ZINC = 30

CUTOFF = 5.0
SWITCH = 0.8
COVALENT_MAX = 1.95
TEMPERATURE = 1.5
HBOND_CONE = 0.5
PROBE = 1.4
SASA_SCALE = 20.0
CONTEXT_RANGE = 0.25
GATE_REFERENCE = 4.0
NULL_LOGIT = 0.5
SHARPNESS = 0.5
GLOBAL_BOUND = 0.3


def gaussian(x, center, width):
    return torch.exp(-torch.square((x - center) / width))


def ramp_down(x, full_at, zero_at):
    """1 up to `full_at`, falling linearly to 0 at `zero_at`."""
    return ((zero_at - x) / (zero_at - full_at)).clamp(0.0, 1.0)


def contact_geometry(data):
    """Unit vector (protein to ligand), distance and smooth cutoff envelope of every contact."""
    ligand_atom, protein_atom = data.contact_edge_index
    delta = data.pos[ligand_atom] - data.pos[protein_atom]
    distance = delta.norm(dim=1).clamp_min(1e-8)
    scaled = distance / CUTOFF
    taper = ((scaled - SWITCH) / (1.0 - SWITCH)).clamp(0.0, 1.0)
    envelope = (1.0 - taper.square() * (3.0 - 2.0 * taper)) * (scaled < 1).to(
        distance.dtype
    )
    return delta / distance[:, None], distance, envelope


def atom_axes(data):
    """Unit vector per atom pointing away from its covalent neighbours, and whether it is defined.
    Hydrogens are not in the graph, so this is where a donor's H or an acceptor's lone pair sits."""
    pos = data.pos
    total = torch.zeros_like(pos)
    for (source, target), keep in (
        (data.ligand_edge_index, data.ligand_bond_type > 0),
        (data.protein_edge_index, None),
    ):
        bond = pos[source] - pos[target]
        length = bond.norm(dim=1).clamp_min(1e-8)
        if keep is None:
            keep = length <= COVALENT_MAX
        total.index_add_(0, target[keep], bond[keep] / length[keep, None])
    norm = total.norm(dim=1, keepdim=True)
    return -total / norm.clamp_min(1e-8), norm[:, 0] > 1e-6


def cone(axis, has_axis, towards):
    """Ramp on the angle between an atom's axis and `towards`; 1 for an atom without an axis."""
    ramp = (((axis * towards).sum(dim=1) - HBOND_CONE) / (1.0 - HBOND_CONE)).clamp(
        0.0, 1.0
    )
    return torch.where(has_axis, ramp, 1.0)


def contact_basis(data, radii, direction, distance):
    """The twelve channel values of every contact, one column per channel."""
    ligand_atom, protein_atom = data.contact_edge_index
    ligand, protein = data.atom_features[ligand_atom], data.atom_features[protein_atom]
    ligand_radius, protein_radius = (
        radii[data.z[ligand_atom]],
        radii[data.z[protein_atom]],
    )
    surface = distance - ligand_radius - protein_radius
    kind = data.node_type[protein_atom]
    direct = (kind == PROTEIN).to(distance.dtype)
    metal_site = (kind == METAL).to(distance.dtype)
    not_metal = (kind != METAL).to(distance.dtype)
    axis, has_axis = atom_axes(data)
    toward_protein = cone(axis[ligand_atom], has_axis[ligand_atom], -direction)
    toward_ligand = cone(axis[protein_atom], has_axis[protein_atom], direction)

    vdw = gaussian(surface, 0.5, 0.5) * not_metal
    repulsion = -F.relu(-surface).square() * not_metal
    hydrophobic = (
        ligand[:, HYDROPHOBE]
        * protein[:, HYDROPHOBE]
        * ramp_down(surface, 0.5, 1.5)
        * direct
    )
    donor_pair = (
        ligand[:, DONOR] * protein[:, ACCEPTOR] * toward_protein
        + ligand[:, ACCEPTOR] * protein[:, DONOR] * toward_ligand
    ).clamp(max=1.0)
    hbond = donor_pair * gaussian(distance, 2.9, 0.55) * direct
    charge_pair = (
        ligand[:, POSITIVE] * protein[:, NEGATIVE]
        + ligand[:, NEGATIVE] * protein[:, POSITIVE]
    ).clamp(max=1.0)
    ionic = charge_pair * ramp_down(distance, 3.0, 5.0) * direct
    aromatic = (
        ligand[:, AROMATIC]
        * protein[:, AROMATIC]
        * gaussian(distance, 4.2, 1.0)
        * direct
    )
    halogen = (
        ligand[:, HALOGEN]
        * protein[:, ACCEPTOR]
        * gaussian(surface, -0.2, 0.5)
        * direct
    )
    coordinator = (ligand[:, ACCEPTOR] > 0) | torch.isin(
        data.z[ligand_atom], data.z.new_tensor([7, 8, 16])
    ) & (ligand[:, CHARGE] <= 0)
    metal = coordinator.to(distance.dtype) * gaussian(distance, 2.4, 0.8) * metal_site
    zinc = metal * (data.z[protein_atom] == ZINC).to(distance.dtype)
    cation_pair = (
        ligand[:, POSITIVE] * protein[:, AROMATIC]
        + ligand[:, AROMATIC] * protein[:, POSITIVE]
    ).clamp(max=1.0)
    cation_pi = cation_pair * gaussian(distance, 4.3, 1.1) * direct

    # the fraction of the ligand atom's solvent shell that this protein atom hides
    ligand_shell, protein_shell = ligand_radius + PROBE, protein_radius + PROBE
    height = ligand_shell - (
        distance.square() + ligand_shell.square() - protein_shell.square()
    ) / (2.0 * distance)
    occluded = (height / (2.0 * ligand_shell)).clamp(0.0, 1.0) * direct
    apolar = data.buried_nonpolar[ligand_atom] / SASA_SCALE * occluded
    polar = data.buried_polar[ligand_atom] / SASA_SCALE * occluded
    return torch.stack(
        (
            vdw,
            repulsion,
            hydrophobic,
            hbond,
            ionic,
            aromatic,
            halogen,
            metal,
            cation_pi,
            apolar,
            polar,
            zinc,
        ),
        dim=1,
    )


def pool(basis, group, n):
    """Max of each channel over a group's contacts; min for repulsion, which is negative."""
    pooled = scatter(basis, group, dim=0, dim_size=n, reduce="max")
    strongest = scatter(basis[:, 1], group, dim=0, dim_size=n, reduce="min")
    return pooled.index_copy(
        1, torch.tensor([1], device=basis.device), strongest[:, None]
    )


def sparsemax(score, segment):
    """Sparsemax of SHARPNESS * score within each segment (the residues one ligand atom shares its vdW
    weight between), against a null column that can keep some of the weight back."""
    logits = SHARPNESS * score
    segments = int(segment.max()) + 1
    counts = torch.bincount(segment, minlength=segments)
    order = torch.argsort(segment)
    rows = segment[order]
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(logits), device=logits.device) - torch.repeat_interleave(
        starts, counts
    )
    dense = logits.new_full((segments, int(counts.max()) + 1), -torch.inf)
    dense[rows, rank] = logits[order]
    dense[:, -1] = SHARPNESS * NULL_LOGIT
    ranked = dense.sort(dim=1, descending=True).values
    cumulative = ranked.cumsum(dim=1)
    position = torch.arange(
        1, dense.shape[1] + 1, device=logits.device, dtype=logits.dtype
    )[None]
    support = (1 + position * ranked > cumulative).sum(dim=1).clamp_min(1)
    threshold = (cumulative.gather(1, (support - 1)[:, None])[:, 0] - 1) / support.to(
        logits.dtype
    )
    weights = torch.empty_like(logits)
    weights[order] = torch.relu(dense - threshold[:, None])[rows, rank]
    return weights


def ligand_terms(descriptors):
    """Six named ligand features from the cached descriptors
    [log1p(heavy) / 4, rotatable / heavy, aromatic / heavy, hetero / heavy, charge / 3, |charge| / 3]."""
    heavy = torch.expm1(4.0 * descriptors[:, 0]).clamp_min(1.0)
    rotatable = descriptors[:, 1] * heavy
    return torch.stack(
        (
            rotatable / 10.0,
            descriptors[:, 0],
            descriptors[:, 3],
            descriptors[:, 2],
            descriptors[:, 4],
            descriptors[:, 5],
        ),
        dim=1,
    )


def global_prior(graph_list, mean, std):
    """Least squares of the standardised target on the six ligand terms: (weights, intercept, mean term)."""
    x = (
        ligand_terms(
            torch.cat([g.ligand_descriptors.reshape(1, 6) for g in graph_list])
        )
        .double()
        .numpy()
    )
    y = torch.cat([g.y for g in graph_list]).double().numpy()
    fitted = np.linalg.lstsq(
        np.column_stack([np.ones(len(x)), x]), (y - mean) / std, rcond=None
    )[0]
    return tuple(fitted[1:]), float(fitted[0]), float((x @ fitted[1:]).mean())


class ResidueTable(nn.Module):
    def __init__(self):
        super().__init__()
        self.side = nn.Parameter(torch.zeros(AA_TYPES, len(CHANNELS)))

    def forward(self, aa):
        """Bounded multiplier per residue type and channel, read at the protein end of each contact."""
        return 1.0 + CONTEXT_RANGE * torch.tanh(self.side[aa])


class GroupMLP(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.embed = nn.Linear(len(CHANNELS) + 1, width)
        self.message = nn.Sequential(
            nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.update = nn.Sequential(
            nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.out = nn.Linear(width, len(CHANNELS))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, basis):
        """Bounded per-channel correction of each group from its own pooled basis. The layer was written for
        message passing; in PACT the neighbour messages are always zero."""
        if len(basis) < 2:
            return torch.zeros_like(basis)
        state = self.embed(torch.cat((basis, torch.ones_like(basis[:, :1])), dim=1))
        message = self.message(torch.cat((state, torch.zeros_like(state)), dim=1))
        state = state + self.update(torch.cat((state, message), dim=1))
        return CONTEXT_RANGE * torch.tanh(self.out(state))


class PairReadout(nn.Module):
    def __init__(self):
        super().__init__()
        radii = [0.0] + [Chem.GetPeriodicTable().GetRvdw(z) for z in range(1, 119)]
        self.register_buffer("vdw_radii", torch.tensor(radii))
        self.raw_physical_weight = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(INITIAL_WEIGHTS)))
        )
        self.local_coef_weight = nn.Parameter(
            torch.zeros(2 * ATOM_FEATURES, len(CHANNELS))
        )
        self.group_gate_weight = nn.Parameter(torch.zeros(len(CHANNELS)))
        self.atom_gate_weight = nn.Parameter(torch.zeros(ATOM_FEATURES))
        self.group_gate_bias = nn.Parameter(torch.zeros(()))
        self.residue_table = ResidueTable()
        self.interaction_graph = GroupMLP()

    def physical_weights(self):
        """Channel coefficients. Softplus keeps them positive, so a channel's sign is fixed by its basis."""
        return F.softplus(self.raw_physical_weight)

    def forward(self, data):
        """Every contact's contribution, and the per-group values the explanation reports."""
        ligand_atom, protein_atom = data.contact_edge_index
        features = data.atom_features
        direction, distance, envelope = contact_geometry(data)
        basis = (
            contact_basis(data, self.vdw_radii, direction, distance) * envelope[:, None]
        )

        # pool the contacts onto (ligand atom, residue) groups
        groups, edge_group = torch.unique(
            torch.stack((ligand_atom, data.residue_id[protein_atom]), dim=1),
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        n, group_atom = len(groups), groups[:, 0]
        closeness = torch.exp(-distance / TEMPERATURE) * envelope
        group_closeness = scatter(
            closeness, edge_group, dim=0, dim_size=n, reduce="sum"
        ).clamp_min(1e-12)
        share = closeness / group_closeness[edge_group]
        context = self.residue_table(data.aa[protein_atom])
        contextual = basis * context
        group_basis = pool(basis, edge_group, n)
        pre_mlp = pool(contextual, edge_group, n)

        # the local chemistry coefficient: the ligand atom's features and the protein side's, closeness-weighted
        protein_side = scatter(
            features[protein_atom] * share[:, None],
            edge_group,
            dim=0,
            dim_size=n,
            reduce="sum",
        )
        local_coef = 1.0 + torch.tanh(
            torch.cat((features[group_atom], protein_side), dim=1)
            @ self.local_coef_weight
        )

        # the vdW weight of a ligand atom is shared between the residues it touches
        gate = (
            group_basis @ self.group_gate_weight
            + self.group_gate_bias
            + features[group_atom] @ self.atom_gate_weight
        )
        distance_prior = torch.log(group_closeness) + GATE_REFERENCE / TEMPERATURE
        segment = torch.unique(group_atom, sorted=True, return_inverse=True)[1]
        vdw_scale = torch.ones_like(group_basis)
        vdw_scale[:, 0] = sparsemax(distance_prior + gate, segment)

        scaled = pre_mlp * (1.0 + self.interaction_graph(group_basis))
        channels = scaled * (self.physical_weights()[None] * local_coef) * vdw_scale

        # split each group's value back onto its contacts
        score = contextual.abs() * closeness[:, None]
        normalizer = scatter(score, edge_group, dim=0, dim_size=n, reduce="sum")[
            edge_group
        ]
        split = torch.where(
            normalizer > 1e-12, score / normalizer.clamp_min(1e-12), share[:, None]
        )
        contact = (channels[edge_group] * split).sum(dim=1)
        return {
            "contact": contact,
            "contact_basis": basis,
            "context": context,
            "distance": distance,
            "edge_group": edge_group,
            "group_atom": group_atom,
            "group_residue": groups[:, 1],
            "segment": segment,
            "basis": group_basis,
            "pre_mlp": pre_mlp,
            "scaled": scaled,
            "vdw_share": vdw_scale[:, 0],
            "local_coef": local_coef,
            "distance_prior": distance_prior,
            "channels": channels,
        }


class PACT(nn.Module):
    def __init__(self, prior):
        """`prior` is (weights, intercept, centre) of the ligand term, from `global_prior`."""
        super().__init__()
        self.pair_readout = PairReadout()
        weights, intercept, centre = prior
        self.global_centre = round(centre, 4)
        self.named_global_weight = nn.Parameter(
            torch.tensor(weights, dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.full(
                (1,),
                intercept + 1.5 * math.tanh(self.global_centre),
                dtype=torch.float32,
            )
        )

    def forward(self, data):
        """Standardised pK = bias + ligand_global + sum over contacts, with every intermediate."""
        out = self.pair_readout(data)
        terms = (
            ligand_terms(data.ligand_descriptors.reshape(data.num_graphs, 6))
            * self.named_global_weight
        )
        ligand_global = GLOBAL_BOUND * torch.tanh(
            (terms.sum(dim=1) - self.global_centre) / GLOBAL_BOUND
        )
        baseline = self.bias.expand(data.num_graphs)
        contact_graph = data.batch[data.contact_edge_index[0]]
        physics = scatter(
            out["contact"], contact_graph, dim=0, dim_size=data.num_graphs, reduce="sum"
        )
        return {
            "prediction": baseline + ligand_global + physics,
            "baseline": baseline,
            "contact_graph": contact_graph,
            "group_graph": data.batch[out["group_atom"]],
            **out,
        }


@lru_cache(maxsize=3)
def graphs(split):
    """The featurised LP-PDBbind complexes of one split (train, val or test)."""
    return torch.load(paths.need(f"lp_pdbbind/{split}_all.pt"), weights_only=False)[
        "graphs"
    ]


def normalisation(graph_list):
    """Mean and SD of the target: the model predicts in these units."""
    y = torch.cat([g.y for g in graph_list])
    return float(y.mean()), float(y.std(unbiased=False))


@lru_cache(maxsize=None)
def load(name):
    """Checkpoint `name` (lp0-lp5, cleansplit0-cleansplit4) on the CPU, with the mean and SD it predicts in."""
    saved = torch.load(
        paths.CHECKPOINTS / f"{name}.pt", map_location="cpu", weights_only=False
    )
    model = PACT(saved["prior"])
    model.load_state_dict(saved["state_dict"])
    model.mean, model.std = saved["mean"], saved["std"]
    return model.eval()


@torch.inference_mode()
def predict(name, graph_list, batch_size=64):
    """Predictions in pK, in the order given."""
    model = load(name)
    return np.concatenate(
        [
            (model(batch)["prediction"] * model.std + model.mean).numpy()
            for batch in DataLoader(graph_list, batch_size)
        ]
    )
