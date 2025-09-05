"""
Deep Knowledge Tracing with Graph Attention over a Skill Graph (ASSISTments)
----------------------------------------------------------------------------
- Frameworks: PyTorch, PyTorch Geometric
- Idea: Build a skill-to-skill co-occurrence graph; learn skill embeddings via GAT; feed
        (skill,response) embeddings through a GRU to model student state; predict
        probability of correctness for the next exercised skill.

Assumptions about the dataset (CSV):
  Required columns (any case-insensitive alias works):
    - user_id       (aliases: user, student_id, anon_student_id, student)
    - skill_id      (aliases: skill, skill_name, kc, kc_id, knowledge_component, problem_skill)
    - correct       (aliases: is_correct, correctness, outcome; values in {0,1,true/false})
    - timestamp     (aliases: time, start_time, end_time, order_id; if absent, we sort by index)

Usage (example):
  python dkt_gat_assistments.py \
      --data_csv ./assistments_2009_2010.csv \
      --min_seq_len 5 --max_seq_len 200 \
      --batch_size 64 --epochs 10 \
      --gat_hidden 128 --gat_heads 4 --rnn_hidden 256

Notes:
  - If you don't have timestamps, the script will keep the original row order per student.
  - Edges: we connect consecutive *different* skills encountered by the same student;
           weights = co-occurrence counts; you can enable thresholding.
  - Evaluation: AUC and accuracy on held-out students (80/10/10 split by student).
  - This is a clean reference implementation meant to run on a single GPU/CPU.
"""

import argparse
import math
import os
import random
import networkx as nx
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import plotly.io as pio
from pyvis.network import Network


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

# Optional but recommended: PyTorch Geometric for GAT
try:
    from torch_geometric.data import Data as GeoData
    from torch_geometric.utils import coalesce
    from torch_geometric.nn import GATv2Conv
    PYG_AVAILABLE = True
except Exception as e:
    PYG_AVAILABLE = False
    print("[Warning] PyTorch Geometric is not available. Install with:\n"
          "pip install torch-geometric\n"
          "and follow the install instructions for torch-scatter/torch-sparse.")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def auto_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
    # try relaxed contains
    for c in df.columns:
        lc = c.lower()
        for name in candidates:
            if name in lc:
                return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")

@dataclass
class Config:
    data_csv: str
    min_seq_len: int = 5
    max_seq_len: int = 200
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # GAT
    emb_dim: int = 128
    gat_hidden: int = 128
    gat_heads: int = 4
    gat_layers: int = 2
    edge_threshold: int = 0  # keep edges with weight > threshold
    # RNN
    rnn_hidden: int = 256
    rnn_layers: int = 1
    dropout: float = 0.2
    # Misc
    num_workers: int = 2

class AssistSeqDataset(Dataset):
    """Sequences of (skill, response) per student, padded per-batch.
       We'll predict the correctness of the *next* skill.
    """
    def __init__(self, seqs: List[Tuple[List[int], List[int]]], max_len: int):
        self.samples = []
        self.max_len = max_len
        for skills, resps in seqs:
            # Trim very long sequences into chunks of max_len
            s = np.array(skills, dtype=np.int64)
            r = np.array(resps, dtype=np.int64)
            n = len(s)
            if n < 2:
                continue
            start = 0
            while start < n - 1:
                end = min(start + max_len, n)
                # inputs are up to end-1, targets from 1..end-1
                s_chunk = s[start:end]
                r_chunk = r[start:end]
                if len(s_chunk) >= 2:
                    self.samples.append((s_chunk.tolist(), r_chunk.tolist()))
                start += max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s, r = self.samples[idx]
        # inputs: up to T-1; targets: from 1..T-1
        skills_in = s[:-1]
        resp_in = r[:-1]
        skills_tgt = s[1:]
        resp_tgt = r[1:]
        length = len(skills_in)
        return (
            torch.tensor(skills_in, dtype=torch.long),
            torch.tensor(resp_in, dtype=torch.long),
            torch.tensor(skills_tgt, dtype=torch.long),
            torch.tensor(resp_tgt, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )

def collate_fn(batch):
    # sort by length desc
    batch = sorted(batch, key=lambda x: x[4], reverse=True)
    skills_in, resp_in, skills_tgt, resp_tgt, lengths = zip(*batch)
    lengths = torch.stack(lengths)
    max_len = lengths.max().item()
    pad_val = 0

    def pad1d(seqs, pad_val):
        out = torch.full((len(seqs), max_len), pad_val, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
        return out

    skills_in = pad1d(skills_in, pad_val)
    resp_in = pad1d(resp_in, 0)
    skills_tgt = pad1d(skills_tgt, pad_val)
    resp_tgt = pad1d(resp_tgt, 0)

    return skills_in, resp_in, skills_tgt, resp_tgt, lengths

class SkillGraphBuilder:
    def __init__(self, edge_threshold: int = 0, strategy: str = "cooccurrence"):
        """
        strategy:
          - 'cooccurrence': connect consecutive different skills in sequences (default).
          - 'transition': weighted directed edges based on conditional probability P(next|current).
          - 'similarity': cosine similarity between skill response patterns (requires matrix).
        """
        self.edge_threshold = edge_threshold
        self.strategy = strategy

    def build(self, sequences: List[Tuple[List[int], List[int]]], num_skills: int):
        if self.strategy == "cooccurrence":
            counts = {}
            for skills, _ in sequences:
                for a, b in zip(skills[:-1], skills[1:]):
                    if a == b:  # skip self
                        continue
                    u, v = int(a), int(b)
                    if u > v:  # undirected
                        u, v = v, u
                    counts[(u, v)] = counts.get((u, v), 0) + 1

            edges_u, edges_v, weights = [], [], []
            for (u, v), w in counts.items():
                if w > self.edge_threshold:
                    edges_u += [u, v]
                    edges_v += [v, u]
                    weights += [w, w]

        elif self.strategy == "transition":
            # conditional transition counts
            counts = {}
            totals = {}
            for skills, _ in sequences:
                for a, b in zip(skills[:-1], skills[1:]):
                    if a == b:
                        continue
                    u, v = int(a), int(b)
                    counts[(u, v)] = counts.get((u, v), 0) + 1
                    totals[u] = totals.get(u, 0) + 1

            edges_u, edges_v, weights = [], [], []
            for (u, v), c in counts.items():
                prob = c / totals[u]
                if prob > (self.edge_threshold / 100):  # threshold as percentage
                    edges_u.append(u)
                    edges_v.append(v)
                    weights.append(prob)

        elif self.strategy == "similarity":
            # build skill × response matrix
            mat = np.zeros((num_skills, num_skills))
            for skills, resps in sequences:
                for s, r in zip(skills, resps):
                    mat[s, s] += r  # simplistic: accumulate correctness
            # cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(mat)
            edges_u, edges_v, weights = [], [], []
            for u in range(num_skills):
                for v in range(num_skills):
                    if u < v and sim[u, v] > (self.edge_threshold / 100):
                        edges_u += [u, v]
                        edges_v += [v, u]
                        weights += [sim[u, v], sim[u, v]]

        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

        # fallback: if no edges, add self-loops
        if len(edges_u) == 0:
            edges_u = list(range(num_skills))
            edges_v = list(range(num_skills))
            weights = [1] * num_skills

        edge_index = torch.tensor([edges_u, edges_v], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)

        if PYG_AVAILABLE:
            edge_index, edge_weight = coalesce(edge_index, edge_weight, num_skills, num_skills)
        return edge_index, edge_weight


class GATSkillEncoder(nn.Module):
    def __init__(self, num_skills: int, emb_dim: int, heads: int = 2, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.num_skills = num_skills
        self.emb_dim = emb_dim

        layers = []
        for i in range(num_layers):
            in_dim = emb_dim if i == 0 else emb_dim
            out_dim = emb_dim
            layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=True,
                    edge_dim=1   # 👈 Added so edge weights are used as features
                )
            )
        self.gnn = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

        # skill embedding (input to GAT)
        self.skill_emb = nn.Embedding(num_skills, emb_dim)

    def forward(self, edge_index, edge_weight):
        # h: (num_skills, emb_dim)
        h = self.skill_emb(torch.arange(self.num_skills, device=edge_index.device))
        for layer in self.gnn:
            h = layer(h, edge_index, edge_weight.view(-1, 1))  # 👈 Pass weights as edge features
            h = F.elu(h)
            h = self.dropout(h)
        return h

class DKT_GAT(nn.Module):
    def __init__(self, num_skills: int, cfg: Config):
        super().__init__()
        self.num_skills = num_skills
        self.encoder = GATSkillEncoder(num_skills, cfg.emb_dim,cfg.gat_heads, cfg.dropout, cfg.gat_layers)
        self.resp_emb = nn.Embedding(2, cfg.emb_dim)  # response 0/1
        self.rnn = nn.GRU(input_size=cfg.emb_dim, hidden_size=cfg.rnn_hidden, num_layers=cfg.rnn_layers, batch_first=True, dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0)
        self.out_proj = nn.Linear(cfg.rnn_hidden, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, skills_in, resp_in, lengths, edge_index, edge_weight):
        # skills_in: (B, T), resp_in: (B, T)
        B, T = skills_in.shape
        skill_embs = self.encoder(edge_index, edge_weight)  # (K, d)
        # gather current skill embeddings
        s_emb = skill_embs[skills_in]  # (B, T, d)
        r_emb = self.resp_emb(resp_in)  # (B, T, d)
        x = s_emb + r_emb  # (B, T, d)
        # pack and run GRU
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        h, h_n = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True)
        h = self.dropout(h)  # (B, T, H)
        # map to embedding space and score via dot with target skill embedding
        h_proj = self.out_proj(h)  # (B, T, d)
        return h_proj, skill_embs

    def predict_next(self, h_proj_t, next_skill_ids, skill_embs):
        # h_proj_t: (B, d) for each timestep t; next_skill_ids: (B,), skill_embs: (K, d)
        w = skill_embs[next_skill_ids]  # (B, d)
        logits = (h_proj_t * w).sum(-1)
        probs = torch.sigmoid(logits)
        return probs, logits


def compute_loss(model: DKT_GAT, batch, edge_index, edge_weight, device):
    skills_in, resp_in, skills_tgt, resp_tgt, lengths = batch
    skills_in = skills_in.to(device)
    resp_in = resp_in.to(device)
    skills_tgt = skills_tgt.to(device)
    resp_tgt = resp_tgt.to(device).float()
    lengths = lengths.to(device)

    h_proj, skill_embs = model(skills_in, resp_in, lengths, edge_index, edge_weight)
    B, T = skills_in.shape

    # For each time step t in [0..L-1], predict correctness for target skill at t (which is skills_tgt[:, t])
    mask = torch.arange(T, device=device)[None, :] < lengths[:, None]

    h_proj_flat = h_proj[mask]  # (N, d)
    target_skills_flat = skills_tgt[mask]  # (N,)
    target_resp_flat = resp_tgt[mask]  # (N,)

    probs, logits = model.predict_next(h_proj_flat, target_skills_flat, skill_embs)
    loss = F.binary_cross_entropy(probs, target_resp_flat)

    with torch.no_grad():
        pred = (probs >= 0.5).float()
        acc = (pred == target_resp_flat).float().mean().item()
    return loss, acc, probs.detach().cpu(), target_resp_flat.detach().cpu()


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Lightweight AUC to avoid pulling sklearn as a heavy dep
    # Computes ROC-AUC using rank statistic (Mann–Whitney U)
    y_true = y_true.astype(np.int64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    # ranking
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def split_by_student(df: pd.DataFrame, user_col: str, ratios=(0.8, 0.1, 0.1)):
    users = df[user_col].drop_duplicates().values
    rng = np.random.default_rng(SEED)
    rng.shuffle(users)
    n = len(users)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_users = set(users[:n_train])
    val_users = set(users[n_train:n_train + n_val])
    test_users = set(users[n_train + n_val:])
    train = df[df[user_col].isin(train_users)].copy()
    val = df[df[user_col].isin(val_users)].copy()
    test = df[df[user_col].isin(test_users)].copy()
    return train, val, test


def prepare_sequences(df: pd.DataFrame, user_col: str, skill_col: str, correct_col: str, time_col: str, min_seq_len: int) -> Tuple[List[Tuple[List[int], List[int]]], Dict[int, int]]:
    # Map skills to contiguous ids
    skill_ids = df[skill_col].astype('category').cat.codes.values
    df = df.copy()
    df['skill_idx'] = skill_ids
    # sort per user by time (or row order)
    if time_col in df.columns:
        df = df.sort_values([user_col, time_col])
    else:
        df = df.sort_values([user_col]).reset_index(drop=True)
    seqs = []
    for uid, g in df.groupby(user_col, sort=False):
        s = g['skill_idx'].tolist()
        r_raw = g[correct_col].values
        r = []
        for v in r_raw:
            if isinstance(v, str):
                v = v.strip().lower()
                if v in ("true", "t", "yes", "y"): v = 1
                elif v in ("false", "f", "no", "n"): v = 0
            r.append(int(v))
        if len(s) >= min_seq_len:
            seqs.append((s, r))
    num_skills = int(df['skill_idx'].max()) + 1
    return seqs, {i: i for i in range(num_skills)}


def infer_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    user_col = auto_col(df, ["user_id", "user", "student_id", "anon_student_id", "student"]) 
    skill_col = auto_col(df, ["skill_id", "skill", "kc", "kc_id", "skill_name", "knowledge_component", "problem_id"]) 
    correct_col = auto_col(df, ["correct", "is_correct", "outcome", "correctness"])
    try:
        time_col = auto_col(df, ["timestamp", "time", "time_on_task", "end_time", "order_id"]) 
    except KeyError:
        time_col = "__no_time__"
    return user_col, skill_col, correct_col, time_col


def save_predictions(model, dataloader, edge_index, edge_weight, device,
                     save_path="predictions.csv", cm_path="confusion_matrix.png"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            loss, acc, preds, labels = compute_loss(model, batch, edge_index, edge_weight, device)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_labels = (all_preds >= 0.5).astype(int)

    # Save predictions CSV
    df = pd.DataFrame({
        "y_true": all_labels,
        "y_pred_prob": all_preds,
        "y_pred_label": pred_labels
    })
    df.to_csv(save_path, index=False)
    print(f"[+] Predictions saved to {save_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Plot confusion matrix using Plotly
    z = cm
    x = ["Positive", "Negative"]
    y = ["Positive", "Negative"]
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale="Blues", showscale=True)

    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="Actual")
    )

    # Save as PNG (needs kaleido) or HTML fallback
    try:
        pio.write_image(fig, cm_path)  # saves PNG
        print(f"[+] Confusion matrix saved to {cm_path}")
    except Exception as e:
        html_path = cm_path.replace(".png", ".html")
        pio.write_html(fig, file=html_path, auto_open=False)
        print(f"[!] PNG export failed, saved as HTML instead: {html_path}")

    return cm


def plot_interactive_skill_graph(edge_index, edge_weight, num_nodes, 
                                 top_k=100, top_edges=500, save_path="skill_graph_interactive.html"):
    G = nx.Graph()

    # Degree filter
    node_degrees = {}
    for u, v in edge_index.t().tolist():
        node_degrees[u] = node_degrees.get(u, 0) + 1
        node_degrees[v] = node_degrees.get(v, 0) + 1

    top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:top_k]
    top_node_set = set(top_nodes)

    # Filter edges
    edges = []
    for (u, v), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        if u in top_node_set and v in top_node_set:
            edges.append((u, v, w))

    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_edges]
    G.add_weighted_edges_from(edges)

    # Build pyvis network
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    net.from_nx(G)

    # Customize nodes
    for node in net.nodes:
        node_id = node["id"]
        degree = G.degree[node_id]
        node["size"] = 20 + degree * 3    # bigger node size
        node["title"] = f"Skill {node_id} (deg={degree})"
        node["label"] = str(node_id)      # show skill number on node

    # Customize edges (make thinner)
    for edge in net.edges:
        edge["width"] = 0.5 + edge.get("value", 1) * 0.1   # thinner edges

    net.write_html(save_path)
    print(f"✅ Interactive graph saved: {save_path}")


def main(cfg: Config):
    print(cfg)
    assert os.path.isfile(cfg.data_csv), f"CSV not found: {cfg.data_csv}"
    df = pd.read_csv(cfg.data_csv)
    df = df.dropna().reset_index(drop=True)
    user_col, skill_col, correct_col, time_col = infer_columns(df)
    print(f"Using columns -> user: {user_col} | skill: {skill_col} | correct: {correct_col} | time: {time_col}")

    # Split by student
    train_df, val_df, test_df = split_by_student(df, user_col)

    # Build sequences
    train_seqs, _ = prepare_sequences(train_df, user_col, skill_col, correct_col, time_col, cfg.min_seq_len)
    val_seqs, _ = prepare_sequences(val_df, user_col, skill_col, correct_col, time_col, cfg.min_seq_len)
    test_seqs, _ = prepare_sequences(test_df, user_col, skill_col, correct_col, time_col, cfg.min_seq_len)

    # Determine num_skills from full set
    all_skills = pd.concat([train_df[[skill_col]], val_df[[skill_col]], test_df[[skill_col]]])
    num_skills = all_skills[skill_col].astype('category').cat.categories.size
    print(f"#students: train={train_df[user_col].nunique()}, val={val_df[user_col].nunique()}, test={test_df[user_col].nunique()}")
    print(f"#skills: {num_skills}")

    # Build skill graph on training sequences
    graph_builder = SkillGraphBuilder(
        edge_threshold=cfg.edge_threshold,
        strategy="cooccurrence"   # change to "transition" or "similarity"
    )
    edge_index, edge_weight = graph_builder.build(train_seqs, num_skills)
    edge_index = edge_index.to(cfg.device)
    edge_weight = edge_weight.to(cfg.device)

    # Datasets and loaders
    train_ds = AssistSeqDataset(train_seqs, cfg.max_seq_len)
    val_ds = AssistSeqDataset(val_seqs, cfg.max_seq_len)
    test_ds = AssistSeqDataset(test_seqs, cfg.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

    # Model
    model = DKT_GAT(num_skills, cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_auc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses, accs = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            loss, acc, p, y = compute_loss(model, batch, edge_index, edge_weight, cfg.device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            accs.append(acc)
        train_loss = float(np.mean(losses)) if losses else float('nan')
        train_acc = float(np.mean(accs)) if accs else float('nan')

        # Validation
        model.eval()
        with torch.no_grad():
            all_p, all_y = [], []
            v_losses, v_accs = [], []
            for batch in val_loader:
                loss, acc, p, y = compute_loss(model, batch, edge_index, edge_weight, cfg.device)
                v_losses.append(loss.item())
                v_accs.append(acc)
                all_p.append(p.numpy())
                all_y.append(y.numpy())
            if all_p:
                ps = np.concatenate(all_p)
                ys = np.concatenate(all_y)
                val_auc = auc_score(ys, ps)
            else:
                val_auc = float('nan')
        val_loss = float(np.mean(v_losses)) if v_losses else float('nan')
        val_acc = float(np.mean(v_accs)) if v_accs else float('nan')

        print(f"Epoch {epoch:02d} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f} AUC {val_auc:.4f}")

        os.makedirs("models", exist_ok=True)

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_state': model.state_dict(),
                'cfg': cfg.__dict__,
                'num_skills': num_skills,
            }, 'models/best_dkt_gat.pt')
            print("  -> Saved best checkpoint: best_dkt_gat.pt")

    # Test
    model.eval()
    with torch.no_grad():
        all_p, all_y = [], []
        t_losses, t_accs = [], []
        for batch in test_loader:
            loss, acc, p, y = compute_loss(model, batch, edge_index, edge_weight, cfg.device)
            t_losses.append(loss.item())
            t_accs.append(acc)
            all_p.append(p.numpy())
            all_y.append(y.numpy())
        if all_p:
            ps = np.concatenate(all_p)
            ys = np.concatenate(all_y)
            test_auc = auc_score(ys, ps)
        else:
            test_auc = float('nan')
    test_loss = float(np.mean(t_losses)) if t_losses else float('nan')
    test_acc = float(np.mean(t_accs)) if t_accs else float('nan')
    print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f} AUC {test_auc:.4f}")

    cm = save_predictions(
        model,
        test_loader,
        edge_index,
        edge_weight,
        cfg.device,
        save_path="outputs/test_predictions.csv",
        cm_path="outputs/confusion_matrix.png"
    )

    plot_interactive_skill_graph(edge_index, edge_weight, 
                                num_nodes=num_skills, 
                                top_k=100, top_edges=500, 
                                save_path="outputs/skill_graph_interactive.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, required=True)
    parser.add_argument('--min_seq_len', type=int, default=5)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--gat_hidden', type=int, default=128)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--gat_layers', type=int, default=2)
    parser.add_argument('--edge_threshold', type=int, default=0)
    parser.add_argument('--rnn_hidden', type=int, default=256)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    cfg = Config(**vars(args))
    main(cfg)
