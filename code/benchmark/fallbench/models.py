"""RF, SVM and LSTM behind one interface.

    model = make_model(name, params, cfg, seed)
    model.fit(X, y, groups)        # groups = trial ids (LSTM early stopping)
    model.score(X)                 # continuous FALL score for ranking / thresholding
    model.describe()               # structure for the cost table
    model.save(path) / load(path)

All learned preprocessing (imputation, scaling) lives inside the model and
is fitted on the training data passed to fit().
"""
from __future__ import annotations

import copy
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

NAMES = {"rf": "RF", "svm": "SVM", "lstm": "LSTM"}


class Model:
    input = "sum"                                  # "sum" (window summaries) or "seq" (ordered frames)

    def __init__(self, params: dict, cfg: dict, seed: int):
        self.params, self.cfg, self.seed = dict(params), cfg, seed
        self.weighted = cfg["class_weighting"] == "balanced"

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "Model":
        return joblib.load(path)


class SklearnModel(Model):
    def fit(self, X, y, groups=None):
        t0 = time.perf_counter()
        self.pipe = self.build().fit(X, y)
        self.train_seconds = time.perf_counter() - t0
        return self


class RF(SklearnModel):
    def build(self):
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             RandomForestClassifier(random_state=self.seed, n_jobs=-1,
                                                    class_weight="balanced" if self.weighted else None,
                                                    **self.params))

    def score(self, X):
        return self.pipe.predict_proba(X)[:, 1]

    def describe(self):
        rf = self.pipe[-1]
        nodes = [t.tree_.node_count for t in rf.estimators_]
        depth = max(t.tree_.max_depth for t in rf.estimators_)
        return dict(structure=f"{len(nodes)} trees, {sum(nodes)} nodes, max depth {depth}",
                    n_trees=len(nodes), total_nodes=sum(nodes), max_depth=depth)

    def ops(self, X):
        """Measured: mean number of tree nodes visited per window."""
        visits = self.pipe[-1].decision_path(self.pipe[0].transform(X))[0].sum(axis=1).mean()
        return float(visits), "tree-node visits (measured)"


class SVM(SklearnModel):
    def build(self):
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             SVC(cache_size=1000, class_weight="balanced" if self.weighted else None,
                                 **self.params))

    def score(self, X):
        # decision value: enough for ROC/PR, and the threshold is set on training-subject scores
        return self.pipe.decision_function(X)

    def describe(self):
        svc = self.pipe[-1]
        return dict(structure=f"{svc.kernel} kernel, {svc.n_support_.sum()} support vectors",
                    kernel=svc.kernel, n_support_vectors=int(svc.n_support_.sum()))

    def ops(self, X):
        svc = self.pipe[-1]
        return float(svc.n_support_.sum() * svc.shape_fit_[1]), "kernel multiply-adds (theoretical)"


# ---------------------------------------------------------------- LSTM
import torch                                    # noqa: E402
from torch import nn                            # noqa: E402


class LSTMNet(nn.Module):
    def __init__(self, n_in, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):                        # B x T x F, fixed T (no padding needed)
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def device(pref: str) -> str:
    return "cuda" if pref in ("auto", "cuda") and torch.cuda.is_available() else "cpu"


class LSTM(Model):
    input = "seq"

    def _scale(self, X):
        Z = (X - self.mean) / self.std
        return torch.tensor(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)

    def fit(self, X, y, groups):
        from .config import set_seed
        set_seed(self.seed)
        m, p = self.cfg["models"]["lstm"], self.params
        self.device = device(m["device"])
        # early stopping on held-out training trials (never the test subject)
        k = round(1 / m["val_fraction"])
        tr, va = next(StratifiedGroupKFold(k, shuffle=True, random_state=self.seed).split(X, y, groups))
        flat = X[tr].reshape(-1, X.shape[-1])
        self.mean = np.nan_to_num(np.nanmean(flat, 0))
        sd = np.nan_to_num(np.nanstd(flat, 0))
        self.std = np.where(sd > 1e-8, sd, 1.0)
        Xt, yt = self._scale(X[tr]), torch.tensor(y[tr], dtype=torch.float32)
        self.net = LSTMNet(X.shape[-1], p["hidden"], p["layers"], p["dropout"]).to(self.device)
        pos = y[tr].sum()
        weight = (len(tr) - pos) / pos if self.weighted and pos else 1.0
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(weight), device=self.device))
        opt = torch.optim.Adam(self.net.parameters(), lr=p["lr"], weight_decay=1e-4)
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, bad, self.history = -1.0, None, 0, []
        t0 = time.perf_counter()
        for epoch in range(m["max_epochs"]):
            self.net.train()
            for b in torch.randperm(len(Xt), generator=gen).split(p["batch"]):
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[b].to(self.device)), yt[b].to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            ap = average_precision_score(y[va], self.score(X[va])) if 0 < y[va].sum() < len(va) else 0.0
            self.history.append(ap)
            if ap > best + 1e-4:
                best, bad, best_state, self.best_epoch = ap, 0, copy.deepcopy(self.net.state_dict()), epoch + 1
            elif (bad := bad + 1) >= m["patience"]:
                break
        self.net.load_state_dict(best_state)
        self.train_seconds = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def score(self, X, batch: int = 4096):
        self.net.eval()
        Z = self._scale(X)
        return torch.cat([torch.sigmoid(self.net(Z[i:i + batch].to(self.device))).cpu()
                          for i in range(0, len(Z), batch)]).numpy()

    def to(self, dev: str):
        self.device = dev
        self.net.to(dev)
        return self

    def save(self, path):
        dev = self.device
        joblib.dump(self.to("cpu"), path)
        self.to(dev)

    def describe(self):
        p = self.params
        return dict(structure=f"hidden {p['hidden']}, {p['layers']} layer(s), best epoch {self.best_epoch}",
                    n_parameters=sum(t.numel() for t in self.net.parameters()))

    def ops(self, X):
        T, F, h = X.shape[1], X.shape[2], self.params["hidden"]
        macs = sum(T * 4 * h * ((F if i == 0 else h) + h) for i in range(self.params["layers"])) + h
        return float(macs), "multiply-accumulates (theoretical)"


def make_model(name: str, params: dict, cfg: dict, seed: int) -> Model:
    return {"rf": RF, "svm": SVM, "lstm": LSTM}[name](params, cfg, seed)
