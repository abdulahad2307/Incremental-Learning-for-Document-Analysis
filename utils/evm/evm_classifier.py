import numpy as np
import gc
import torch
from sklearn.metrics import pairwise_distances
from scipy.stats import weibull_min


class EVMClassifier:
    """Extreme Value Machine for open-world classification (model-agnostic)."""

    def __init__(self, tailsize: float = 0.3, cover_threshold: float = 0.7,
                 distance_metric: str = 'euclidean',batch_size=128, max_fit_samples=50):
        self.tailsize = tailsize
        self.cover_threshold = cover_threshold
        self.distance_metric = distance_metric
        self.weibull_models = {}
        self.class_means = {}
        self.class_features = {}
        self.initialized = False
        self.batch_size = batch_size
        self.max_fit_samples = max_fit_samples
    
    def _fit_weibull(self, tail):
        """(scale, shape) of the Weibull fitted to one extreme vector's tail of half-distances (maximum likelihood)."""
        try:
            shape, _, scale = weibull_min.fit(tail, floc=0)
            return scale, shape
        except Exception:
            return float(np.mean(tail)), 1.0

    def _prune_features(self, features):
        if self.max_fit_samples and features.shape[0] > self.max_fit_samples:
            indices = np.linspace(0, features.shape[0] - 1, self.max_fit_samples).astype(int)
            return features[indices]
        return features

    def fit(self, features: dict):
        """Fit EVM to feature vectors for each class/domain (memory efficient)."""
        classes = list(features.keys())
        # Storing features and means in float32 to save memory
        for class_name, class_features in features.items():
            arr = np.asarray(class_features, dtype=np.float32)
            self.class_features[class_name] = arr
            self.class_means[class_name] = np.mean(arr, axis=0)

        # Fitting Weibull models per class
        for target_class in classes:
            target_features = self._prune_features(self.class_features[target_class])
            if len(classes) > 1:
                negatives = [self._prune_features(self.class_features[c]) for c in classes if c != target_class]
                negative_features = np.vstack(negatives)
            else:
                negative_features = target_features

            negative_features = negative_features.astype(np.float32, copy=False)

            weibull_models = []
            # Processing features in batches to avoid huge distance matrix in memory
            for start in range(0, target_features.shape[0], self.batch_size):
                end = min(start + self.batch_size, target_features.shape[0])
                batch_feats = target_features[start:end]

                # Pairwise distances between this batch and all negatives
                dists = pairwise_distances(batch_feats, negative_features, metric=self.distance_metric)

                for i in range(dists.shape[0]):
                    point_distances = np.sort(dists[i])
                    t = (max(int(len(point_distances) * self.tailsize), 1)
                         if isinstance(self.tailsize, float) and self.tailsize < 1.0
                         else min(int(self.tailsize), len(point_distances)))
                    # EVM fits the Weibull on half-distances to the nearest negatives (margin)
                    tailsize_distances = point_distances[:t] / 2.0
                    scale, shape = self._fit_weibull(tailsize_distances)
                    weibull_models.append((batch_feats[i], scale, shape))
                del dists
                gc.collect()

            self.weibull_models[target_class] = weibull_models

        self.initialized = True
        gc.collect()
        return self
    
    def fit_cuda(self, features: dict, device=torch.device('cuda')):
        """
        Similar to fit(), but leverages PyTorch on CUDA for distance computations,
        minimizing CPU-GPU data copying. Weibull fitting still on CPU.
        """
        classes = list(features.keys())
        self.class_features = {}
        self.class_means = {}

        # Converting features to cuda tensors to speed pairwise dist calc
        for class_name, feats in features.items():
            arr = np.asarray(feats, dtype=np.float32)
            self.class_features[class_name] = arr
            self.class_means[class_name] = np.mean(arr, axis=0)

        for target_class in classes:
            target_feats = self._prune_features(self.class_features[target_class])
            target_feats_tensor = torch.tensor(target_feats, dtype=torch.float32, device=device)

            if len(classes) > 1:
                negatives_np = []
                for c in classes:
                    if c != target_class:
                        negatives_np.append(self._prune_features(self.class_features[c]))
                negative_feats = np.vstack(negatives_np)
            else:
                negative_feats = target_feats

            negative_feats_tensor = torch.tensor(negative_feats, dtype=torch.float32, device=device)

            weibull_models = []
            for start in range(0, target_feats_tensor.size(0), self.batch_size):
                end = min(start + self.batch_size, target_feats_tensor.size(0))
                batch_feats = target_feats_tensor[start:end]

                # Computing pairwise distances on GPU
                dists = torch.cdist(batch_feats, negative_feats_tensor, p=2).cpu().numpy()

                for i in range(dists.shape[0]):
                    point_distances = np.sort(dists[i])
                    t = (max(int(len(point_distances) * self.tailsize), 1)
                         if isinstance(self.tailsize, float) and self.tailsize < 1.0
                         else min(int(self.tailsize), len(point_distances)))
                    # EVM fits the Weibull on half-distances to the nearest negatives (margin)
                    tailsize_distances = point_distances[:t] / 2.0
                    scale, shape = self._fit_weibull(tailsize_distances)
                    weibull_models.append((batch_feats[i].cpu().numpy(), scale, shape))
                del dists
                torch.cuda.empty_cache()

            self.weibull_models[target_class] = weibull_models

        self.initialized = True
        return self

    def predict_proba(self, features: np.ndarray) -> dict:
        """Predict class probabilities for input features (memory efficient)."""
        if not self.initialized or len(self.weibull_models) == 0:
            raise ValueError("Model not fitted yet")

        features = np.asarray(features, dtype=np.float32)
        probabilities = {}

        # For each class, computing max-probabilities using its Weibull models
        for class_name, models in self.weibull_models.items():
            class_probs = np.zeros(features.shape[0], dtype=np.float32)

            # Processing Weibull models in batches for efficiency
            for point, scale, shape in models:
                # Chunk-chunk features for distance computation
                for start in range(0, features.shape[0], self.batch_size):
                    end = min(start + self.batch_size, features.shape[0])
                    feat_batch = features[start:end]
                    dists = pairwise_distances(
                        feat_batch, point.reshape(1, -1), metric=self.distance_metric
                    ).flatten().astype(np.float32)
                    with np.errstate(over="ignore"):  # (d/scale)^shape -> inf means Psi = 0
                        point_probs = np.exp(-((dists / scale) ** shape))
                    class_probs[start:end] = np.maximum(class_probs[start:end], point_probs)

            # Psi_class(x) = max over extreme vectors of their inclusion probability
            probabilities[class_name] = class_probs
            gc.collect()

        return probabilities
    
    def predict_proba_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """
        CUDA-compatible probability prediction for a batch of features.

        Args:
            features (torch.Tensor): shape (N, D), on the same device as model.

        Returns:
            torch.Tensor: shape (N, num_classes), probabilities on CPU float32.
        """
        if not self.initialized or len(self.weibull_models) == 0:
            raise RuntimeError("EVM not fitted yet.")

        device = features.device
        cols = []
        for class_name in self.weibull_models:
            models = self.weibull_models[class_name]
            points = torch.tensor(np.stack([m[0] for m in models]), dtype=torch.float32, device=device)  # (M, D)
            scales = torch.tensor([float(m[1]) for m in models], dtype=torch.float32, device=device)
            shapes = torch.tensor([float(m[2]) for m in models], dtype=torch.float32, device=device)
            # Out-of-place (autograd-safe): inclusion probability exp(-(d/scale)^shape), max over extreme vectors
            chunks = []
            for start in range(0, features.size(0), self.batch_size):
                dists = torch.cdist(features[start:start + self.batch_size], points, p=2)  # (n, M)
                chunks.append(torch.exp(-(dists / scales) ** shapes).max(dim=1).values)
            cols.append(torch.cat(chunks))
        return torch.stack(cols, dim=1).cpu()

    def predict(self, features: np.ndarray, threshold: float = None):
        """Predict class labels for input features."""
        if threshold is None:
            threshold = self.cover_threshold
        probabilities = self.predict_proba(features)
        class_names = list(probabilities.keys())
        prob_matrix = np.column_stack([probabilities[c] for c in class_names])
        max_probs = np.max(prob_matrix, axis=1)
        max_indices = np.argmax(prob_matrix, axis=1)
        labels = []
        for prob, idx in zip(max_probs, max_indices):
            if prob >= threshold:
                labels.append(class_names[idx])
            else:
                labels.append("unknown")
        return labels, max_probs

    def incremental_update(self, new_features: dict):
        """Update the model with new classes or examples (re-fit)."""
        for class_name, class_features in new_features.items():
            arr = np.asarray(class_features, dtype=np.float32)
            if class_name in self.class_features:
                self.class_features[class_name] = np.concatenate(
                    (self.class_features[class_name], arr), axis=0
                )
            else:
                self.class_features[class_name] = arr
            # Updating mean
            self.class_means[class_name] = np.mean(self.class_features[class_name], axis=0)
        # Re-fitting on updated dataset
        self.fit(self.class_features)
        return self
