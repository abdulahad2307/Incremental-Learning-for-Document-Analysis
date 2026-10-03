import torch
import numpy as np
from scipy.stats import weibull_min
from sklearn.metrics import pairwise_distances

class IncrementalEVM:
    """
    Incremental Extreme Value Machine (iEVM)
    PyTorch version based on the TensorFlow-iEVM.
    """

    def __init__(self, tailsize=0.3, ev_budget=10, cover_threshold=0.7, distance_metric="euclidean"):
        self.tailsize = tailsize
        self.ev_budget = ev_budget
        self.cover_threshold = cover_threshold
        self.distance_metric = distance_metric
        self.class_evs = {}    
        self.class_features = {}
        self.initialized = False

    ###################### UTILITY FUNCTIONS ######################

    @staticmethod
    def np_pairwise_distances(A, B):
        return pairwise_distances(A, B)

    @staticmethod
    def torch_pairwise_distances(A, B):
        if not torch.is_tensor(A):
            A = torch.tensor(A, dtype=torch.float32)
        if not torch.is_tensor(B):
            B = torch.tensor(B, dtype=torch.float32)
        return torch.cdist(A, B)

    
    ################### GREEDY K-COVER ###################

    def greedy_k_set_cover(self, evs, k):
        """Greedily select up to k extreme vectors that cover the most class points.
        EV i covers point j if its inclusion probability Psi_i(x_j) >= cover_threshold."""
        if len(evs) <= k:
            return evs
        mat = np.array([ev[0] for ev in evs])
        covers = np.zeros((len(evs), len(evs)), dtype=bool)
        for i, ev in enumerate(evs):
            dists = pairwise_distances(mat, ev[0][None, :], metric=self.distance_metric).flatten()
            with np.errstate(over="ignore"):
                psi = np.exp(-((dists / (ev[1] + 1e-8)) ** (ev[2] + 1e-8)))  # Avoid division by zero
            covers[i] = psi >= self.cover_threshold
        uncovered = np.ones(len(evs), dtype=bool)
        idxs = []
        while len(idxs) < k:
            gains = (covers & uncovered).sum(axis=1)
            gains[idxs] = -1
            best = int(np.argmax(gains))
            if gains[best] <= 0:
                break
            idxs.append(best)
            uncovered &= ~covers[best]
        return [evs[i] for i in idxs]

    
    #################### WEIBULL FITTING ####################

    def fit_weibull(self, point, negatives):
        dists = pairwise_distances([point], negatives, metric=self.distance_metric).flatten()
        t = max(int(len(dists) * self.tailsize), 1) if self.tailsize < 1.0 else min(int(self.tailsize), len(dists))
        tails = np.sort(dists)[:t] / 2.0  # half-distances (EVM margin)
        try:
            shape, loc, scale = weibull_min.fit(tails, floc=0)
        except Exception:
            shape, scale = 1., np.mean(tails)
        max_tail = np.max(tails)
        return scale, shape, max_tail


    def fit(self, features_dict):
        """Fit all classes from scratch."""
        self.class_features = {}
        for label, feats in features_dict.items():
            feats = np.array(feats).astype(np.float32)
            self.class_features[label] = feats
        for label in self.class_features:
            feats = self.class_features[label]
            negatives = np.vstack([self.class_features[lab] for lab in self.class_features if lab != label]) if len(self.class_features) > 1 else feats
            evs = []
            for point in feats:
                scale, shape, max_tail = self.fit_weibull(point, negatives)
                evs.append((point, scale, shape, max_tail))
            self.class_evs[label] = self.greedy_k_set_cover(evs, self.ev_budget)
        self.initialized = True
        return self

    def _negatives_for(self, label):
        if len(self.class_features) > 1:
            return np.vstack([self.class_features[lab] for lab in self.class_features if lab != label])
        return self.class_features[label]

    def incremental_update(self, new_features_dict):
        """Incrementally update (add new classes or add samples to existing) and refit efficiently."""
        new_features_dict = {label: np.array(feats).astype(np.float32) for label, feats in new_features_dict.items()}
        # Updating local database
        for label, feats in new_features_dict.items():
            if label in self.class_features:
                self.class_features[label] = np.vstack((self.class_features[label], feats))
            else:
                self.class_features[label] = feats

        for label in self.class_features:
            # New samples of other labels are new negatives for this label
            new_negs = [f for lab, f in new_features_dict.items() if lab != label]
            new_negs = np.vstack(new_negs) if new_negs else None
            new_pos = new_features_dict.get(label)
            if new_negs is None and new_pos is None:
                continue
            negatives = self._negatives_for(label)
            existing_evs = self.class_evs.get(label, [])
            # An EV is affected if a new negative falls inside its margin tail
            affected = new_negs is not None and any(
                np.any(pairwise_distances([ev[0]], new_negs, metric=self.distance_metric).flatten() / 2.0 < ev[3])
                for ev in existing_evs
            )
            if affected:
                # Margins shrank: re-fit every stored point of this class and redo the set cover,
                # otherwise the few retained EVs no longer cover the class
                candidates = self.class_features[label]
                new_evs = []
            else:
                candidates = new_pos if new_pos is not None else []
                new_evs = list(existing_evs)
            for point in candidates:
                scale, shape, max_tail = self.fit_weibull(point, negatives)
                new_evs.append((point, scale, shape, max_tail))
            self.class_evs[label] = self.greedy_k_set_cover(new_evs, self.ev_budget)
        self.initialized = True
        return self

    
    ################### PREDICTION API ###################

    def predict_proba(self, features):
        features = np.array(features, dtype=np.float32)
        results = {}
        for label, evs in self.class_evs.items():
            scores = np.zeros(features.shape[0], dtype=np.float32)
            for ev in evs:
                d = pairwise_distances(features, ev[0][None, :], metric=self.distance_metric).flatten()
                with np.errstate(over="ignore"):  # (d/scale)^shape -> inf means Psi = 0
                    prob = np.exp(-((d / (ev[1] + 1e-8)) ** (ev[2] + 1e-8)))
                scores = np.maximum(scores, prob)
            results[label] = scores  # max inclusion probability; higher for closer samples
        return results

    def predict_proba_tensor(self, features):
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)
        device = features.device
        cols = []
        for cname, evs in self.class_evs.items():
            points = torch.tensor(np.stack([ev[0] for ev in evs]), dtype=torch.float32, device=device)  # (M, D)
            scales = torch.tensor([float(ev[1]) + 1e-8 for ev in evs], dtype=torch.float32, device=device)
            shapes = torch.tensor([float(ev[2]) + 1e-8 for ev in evs], dtype=torch.float32, device=device)
            d = torch.cdist(features, points)  # (N, M)
            # Out-of-place (autograd-safe): max over extreme vectors of the inclusion probability
            cols.append(torch.exp(-(d / scales) ** shapes).max(dim=1).values)
        return torch.stack(cols, dim=1).cpu()

    def predict(self, features, threshold=None):
        if threshold is None:
            threshold = self.cover_threshold
        probs = self.predict_proba(features)
        class_names = list(probs.keys())
        prob_matrix = np.column_stack([probs[c] for c in class_names])
        max_probs = np.max(prob_matrix, axis=1)
        max_indices = np.argmax(prob_matrix, axis=1)
        labels = [class_names[idx] if p >= threshold else "unknown"
                  for p, idx in zip(max_probs, max_indices)]
        return labels, max_probs

    def state_dict(self):
        return {
            "tailsize": self.tailsize,
            "ev_budget": self.ev_budget,
            "cover_threshold": self.cover_threshold,
            "distance_metric": self.distance_metric,
            "class_evs": self.class_evs,
            "class_features": self.class_features,
        }

    def load_state_dict(self, state):
        self.tailsize = state["tailsize"]
        self.ev_budget = state["ev_budget"]
        self.cover_threshold = state["cover_threshold"]
        self.distance_metric = state["distance_metric"]
        self.class_evs = state["class_evs"]
        self.class_features = state["class_features"]
        self.initialized = True
