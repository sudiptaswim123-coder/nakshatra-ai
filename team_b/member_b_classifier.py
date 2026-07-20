"""Member B classifier for ISRO BAH 2026 Challenge 7.

Consumes Member A's manifest/NPZ contract and exposes:
    classify(global_view, local_view, stellar_metadata, tpf)

The model inputs are strictly global_view, local_view, stellar_metadata, and
optional TPF aperture pixels. flattened_flux/time are retained only for
debugging and transit-forwarding payloads.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - exercised on machines without torch
    raise SystemExit(
        "PyTorch is required for Member B classifier training/inference. "
        "Install with: pip install -r requirements-member-b.txt"
    ) from exc

try:
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import train_test_split
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for metrics/splitting. "
        "Install with: pip install -r requirements-member-b.txt"
    ) from exc


CLASS_NAMES = ("transit", "eclipse", "blend", "noise")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
TARGET_F1 = {"transit": 0.92, "eclipse": 0.94, "blend": 0.88, "noise": 0.96}


@dataclass(frozen=True)
class LoadedArrays:
    global_view: np.ndarray
    local_view: np.ndarray
    metadata: np.ndarray
    labels: np.ndarray
    tic_ids: list[str]
    files: list[Path]


class ManifestDataset(Dataset):
    def __init__(self, arrays: LoadedArrays, indices: Iterable[int] | None = None):
        self.arrays = arrays
        self.indices = list(range(len(arrays.labels))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        idx = self.indices[i]
        return {
            "global_view": torch.tensor(self.arrays.global_view[idx], dtype=torch.float32).unsqueeze(0),
            "local_view": torch.tensor(self.arrays.local_view[idx], dtype=torch.float32).unsqueeze(0),
            "metadata": torch.tensor(self.arrays.metadata[idx], dtype=torch.float32),
            "label": torch.tensor(self.arrays.labels[idx], dtype=torch.long),
        }


def load_manifest(manifest_csv: str | Path) -> LoadedArrays:
    """Load Member A's manifest without touching rejected stars or reprocessing."""
    manifest_path = Path(manifest_csv)
    base_dir = manifest_path.parent
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="") as f:
        rows.extend(csv.DictReader(f))

    globals_, locals_, meta, labels, tic_ids, files = [], [], [], [], [], []
    for row in rows:
        star_path = base_dir / row["file"]
        d = np.load(star_path)
        label = str(d["label"])
        if label not in CLASS_TO_ID:
            raise ValueError(f"TIC {row.get('tic_id', '?')} has unknown label {label!r}; ping Member A.")
        global_view = np.asarray(d["global_view"], dtype=np.float32)
        local_view = np.asarray(d["local_view"], dtype=np.float32)
        if global_view.shape != (2000,) or local_view.shape != (200,):
            raise ValueError(f"TIC {row.get('tic_id', '?')} has wrong view shapes; ping Member A.")
        globals_.append(global_view)
        locals_.append(local_view)
        meta.append([float(d["teff"]), float(d["radius"]), float(d["logg"]), float(d["mass"])])
        labels.append(CLASS_TO_ID[label])
        tic_ids.append(row.get("tic_id", ""))
        files.append(star_path)

    return LoadedArrays(
        global_view=np.stack(globals_),
        local_view=np.stack(locals_),
        metadata=np.asarray(meta, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        tic_ids=tic_ids,
        files=files,
    )


class ConvBranch(nn.Module):
    def __init__(self, in_len: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4 if in_len > 500 else 2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4 if in_len > 500 else 2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _extract_tpf_cube(tpf: Any) -> np.ndarray | None:
    if tpf is None:
        return None
    if isinstance(tpf, dict):
        for key in ("flux", "tpf_flux", "pixel_flux", "cube"):
            if key in tpf:
                return np.asarray(tpf[key], dtype=np.float32)
        return None
    return np.asarray(tpf, dtype=np.float32)


def tpf_to_graph(
    tpf: Any,
    k_neighbors: int = 4,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Build graph nodes from TPF pixels using spatial and contamination cues.

    Expected tpf shape is (time, rows, cols) or (rows, cols, time). The returned
    edge_attr is a scalar weight combining inverse pixel distance and similarity
    of the pixel's excess flux above the aperture median.
    """
    cube = _extract_tpf_cube(tpf)
    if cube is None or cube.ndim != 3:
        return None
    if cube.shape[0] <= 16 and cube.shape[-1] > 16:
        cube = np.moveaxis(cube, -1, 0)
    n_time, n_rows, n_cols = cube.shape
    pixels = cube.reshape(n_time, n_rows * n_cols).T
    finite = np.nan_to_num(pixels, nan=np.nanmedian(pixels), posinf=0.0, neginf=0.0)
    mean = finite.mean(axis=1)
    std = finite.std(axis=1)
    peak = finite.max(axis=1)
    excess = np.maximum(mean - np.median(mean), 0.0)

    coords = np.array([(r, c) for r in range(n_rows) for c in range(n_cols)], dtype=np.float32)
    coord_scale = np.maximum(np.array([n_rows - 1, n_cols - 1], dtype=np.float32), 1.0)
    coord_features = coords / coord_scale
    raw_phot_features = np.stack([mean, std, peak], axis=1)
    denom = np.maximum(np.std(raw_phot_features, axis=0), 1e-6)
    phot_features = (raw_phot_features - raw_phot_features.mean(axis=0)) / denom
    node_features = np.concatenate([coord_features, phot_features], axis=1).astype(np.float32)

    edges, weights = [], []
    for i, xy in enumerate(coords):
        dist = np.linalg.norm(coords - xy, axis=1)
        neighbor_ids = np.argsort(dist)[1 : k_neighbors + 1]
        for j in neighbor_ids:
            spatial_w = math.exp(-float(dist[j]))
            contam_w = 1.0 / (1.0 + abs(float(excess[i] - excess[j])))
            edges.append((i, j))
            weights.append(spatial_w * contam_w)

    x = torch.tensor(node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).T.contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    return x, edge_index, edge_weight


class WeightedGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(out_dim * 2 + 1, 1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        h = self.proj(x)
        src, dst = edge_index
        logits = self.attn(torch.cat([h[src], h[dst], edge_weight[:, None]], dim=1)).squeeze(1)
        logits = F.leaky_relu(logits, 0.2)
        exp_logits = torch.exp(logits - logits.max())
        denom = torch.zeros(x.size(0), device=x.device).scatter_add_(0, dst, exp_logits).clamp_min(1e-8)
        alpha = exp_logits / denom[dst]
        out = torch.zeros_like(h).index_add_(0, dst, h[src] * alpha[:, None] * edge_weight[:, None])
        return F.elu(out + self.bias)


class ApertureGNN(nn.Module):
    """Small OmicsGAT-inspired attention encoder for aperture-pixel graphs."""

    def __init__(self, node_dim: int = 5, hidden_dim: int = 32, embedding_dim: int = 32):
        super().__init__()
        self.gat1 = WeightedGraphAttentionLayer(node_dim, hidden_dim)
        self.gat2 = WeightedGraphAttentionLayer(hidden_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.ReLU())

    def forward(self, graph: tuple[Tensor, Tensor, Tensor] | None, batch_size: int, device: torch.device) -> Tensor:
        if graph is None:
            return torch.zeros(batch_size, self.head[0].out_features, device=device)
        x, edge_index, edge_weight = graph
        h = self.gat1(x, edge_index, edge_weight)
        h = self.gat2(h, edge_index, edge_weight)
        pooled = h.mean(dim=0, keepdim=True)
        return self.head(pooled).repeat(batch_size, 1)


class DualViewApertureClassifier(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.global_branch = ConvBranch(2000, 64)
        self.local_branch = ConvBranch(200, 64)
        self.meta_branch = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 16), nn.ReLU())
        self.aperture_gnn = ApertureGNN(embedding_dim=32)
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 16 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        global_view: Tensor,
        local_view: Tensor,
        metadata: Tensor,
        tpf_graph: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> Tensor:
        batch_size = global_view.shape[0]
        device = global_view.device
        emb = torch.cat(
            [
                self.global_branch(global_view),
                self.local_branch(local_view),
                self.meta_branch(metadata),
                self.aperture_gnn(tpf_graph, batch_size=batch_size, device=device),
            ],
            dim=1,
        )
        return self.head(emb)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def fit(self, logits: Tensor, labels: Tensor, steps: int = 250) -> None:
        if logits.numel() == 0:
            return
        opt = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=steps)

        def closure() -> Tensor:
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)


def validate_training_labels(labels: np.ndarray, allow_tiny_sample: bool) -> None:
    present = {CLASS_NAMES[int(i)] for i in np.unique(labels)}
    missing = [name for name in CLASS_NAMES if name not in present]
    if missing and not allow_tiny_sample:
        raise ValueError(
            "Training set is missing classes "
            f"{missing}. Request the full ISRO-curated dataset from Member A before real training. "
            "Use --allow-tiny-sample only for pipeline wiring/smoke tests."
        )


def train_model(args: argparse.Namespace) -> None:
    arrays = load_manifest(args.manifest)
    validate_training_labels(arrays.labels, args.allow_tiny_sample)

    if len(arrays.labels) < 8 or args.allow_tiny_sample:
        train_idx = val_idx = np.arange(len(arrays.labels))
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(arrays.labels)),
            test_size=args.val_fraction,
            stratify=arrays.labels,
            random_state=args.seed,
        )

    device = torch.device(args.device)
    model = DualViewApertureClassifier().to(device)
    scaler = TemperatureScaler().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(ManifestDataset(arrays, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ManifestDataset(arrays, val_idx), batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            opt.zero_grad()
            logits = model(
                batch["global_view"].to(device),
                batch["local_view"].to(device),
                batch["metadata"].to(device),
            )
            loss = F.cross_entropy(logits, batch["label"].to(device))
            loss.backward()
            opt.step()
            running += float(loss.detach())
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train_loss={running / max(len(train_loader), 1):.4f}")

    logits, labels = collect_logits(model, val_loader, device)
    scaler.fit(logits, labels)
    probs = F.softmax(scaler(logits), dim=1).detach().cpu().numpy()
    pred = probs.argmax(axis=1)
    truth = labels.cpu().numpy()
    print(classification_report(truth, pred, labels=list(range(4)), target_names=CLASS_NAMES, zero_division=0))
    f1s = f1_score(truth, pred, labels=list(range(4)), average=None, zero_division=0)
    for name, score in zip(CLASS_NAMES, f1s):
        print(f"{name}: f1={score:.3f} target={TARGET_F1[name]:.2f}")

    checkpoint = {
        "model_state": model.state_dict(),
        "temperature_state": scaler.state_dict(),
        "class_names": CLASS_NAMES,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved_checkpoint={args.output}")


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    model.eval()
    logits_out, labels_out = [], []
    for batch in loader:
        logits_out.append(
            model(
                batch["global_view"].to(device),
                batch["local_view"].to(device),
                batch["metadata"].to(device),
            ).cpu()
        )
        labels_out.append(batch["label"])
    return torch.cat(logits_out), torch.cat(labels_out)


def load_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[DualViewApertureClassifier, TemperatureScaler]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DualViewApertureClassifier().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = TemperatureScaler().to(device)
    scaler.load_state_dict(ckpt.get("temperature_state", {}), strict=False)
    scaler.eval()
    return model, scaler


@torch.no_grad()
def classify(
    global_view: np.ndarray,
    local_view: np.ndarray,
    stellar_metadata: np.ndarray | list[float],
    tpf: Any = None,
    *,
    checkpoint_path: str | Path = "models/member_b_classifier.pt",
    device: str = "cpu",
) -> dict[str, Any]:
    """Return {class_label, confidence_scores[4]} for Member C/D callers."""
    model, scaler = load_checkpoint(checkpoint_path, device=device)
    dev = torch.device(device)
    g = torch.tensor(np.asarray(global_view, dtype=np.float32), device=dev).view(1, 1, 2000)
    l = torch.tensor(np.asarray(local_view, dtype=np.float32), device=dev).view(1, 1, 200)
    m = torch.tensor(np.asarray(stellar_metadata, dtype=np.float32), device=dev).view(1, 4)
    graph = tpf_to_graph(tpf, device=dev)
    probs = F.softmax(scaler(model(g, l, m, graph)), dim=1).squeeze(0).cpu().numpy()
    return {
        "class_label": CLASS_NAMES[int(np.argmax(probs))],
        "confidence_scores": {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)},
    }


def transit_payloads(
    manifest_csv: str | Path,
    checkpoint_path: str | Path,
    min_confidence: float = 0.0,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Forward only Transit-classified candidates with confidence and debug curves."""
    manifest_path = Path(manifest_csv)
    payloads: list[dict[str, Any]] = []
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            d = np.load(manifest_path.parent / row["file"])
            metadata = np.array([d["teff"], d["radius"], d["logg"], d["mass"]], dtype=np.float32)
            result = classify(d["global_view"], d["local_view"], metadata, None, checkpoint_path=checkpoint_path, device=device)
            confidence = result["confidence_scores"]["transit"]
            if result["class_label"] == "transit" and confidence >= min_confidence:
                payloads.append(
                    {
                        "tic_id": row["tic_id"],
                        "class_label": "transit",
                        "confidence": confidence,
                        "confidence_scores": result["confidence_scores"],
                        "flattened_flux": d["flattened_flux"],
                        "time": d["time"],
                    }
                )
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate Member B dual-view aperture classifier.")
    parser.add_argument("--manifest", default="member_a_output_sample/manifest.csv")
    parser.add_argument("--output", default="models/member_b_classifier.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--allow-tiny-sample", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_model(args)


if __name__ == "__main__":
    main()
