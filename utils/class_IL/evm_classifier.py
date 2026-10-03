import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from scipy.stats import weibull_min
from typing import List, Dict, Tuple, Optional

class EVMClassifier:
    """Extreme Value Machine for open-world classification"""
    def __init__(self, 
                 tailsize: float = 0.5, 
                 cover_threshold: float = 0.7,
                 distance_metric: str = 'euclidean'):
        self.tailsize = tailsize
        self.cover_threshold = cover_threshold
        self.distance_metric = distance_metric
        self.weibull_models = {}  # class_name -> list of (mean, scale, shape)
        self.class_means = {}     # class_name -> feature vector
        self.class_features = {}  # class_name -> list of feature vectors
        
    def fit(self, features: Dict[str, np.ndarray]):
        """Fit EVM to feature vectors for each class"""
        classes = list(features.keys())
        
        # Store class features and compute means
        for class_name, class_features in features.items():
            self.class_features[class_name] = class_features
            self.class_means[class_name] = np.mean(class_features, axis=0)
        
        # Fit Weibull models for each class
        for target_class in classes:
            target_features = features[target_class]
            
            # Combine features from all other classes as negative examples
            negative_features = np.vstack([
                features[c] for c in classes if c != target_class
            ])
            
            # Compute distances between target class and negative examples
            distances = pairwise_distances(
                target_features, 
                negative_features, 
                metric=self.distance_metric
            )
            
            # Fit Weibull distribution for each point in target class
            weibull_models = []
            for i, point_distances in enumerate(distances):
                # Sort distances in ascending order
                point_distances = np.sort(point_distances)
                
                # Determine tailsize
                if isinstance(self.tailsize, float) and self.tailsize < 1.0:
                    t = max(int(len(point_distances) * self.tailsize), 1)
                else:
                    t = min(int(self.tailsize), len(point_distances))
                
                # Use only the smallest t half-distances (margin) for fitting
                tailsize_distances = point_distances[:t] / 2.0
                
                # Fit Weibull distribution
                try:
                    shape, loc, scale = weibull_min.fit(tailsize_distances, floc=0)
                    weibull_models.append((target_features[i], scale, shape))
                except:
                    # If fitting fails, use default parameters
                    weibull_models.append((target_features[i], np.mean(tailsize_distances), 1.0))
            
            self.weibull_models[target_class] = weibull_models
            
        return self
    
    def predict_proba(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict class probabilities for input features"""
        if len(self.weibull_models) == 0:
            raise ValueError("Model not fitted yet")
        
        # Initialize probabilities
        probabilities = {}
        
        # Compute probabilities for each class
        for class_name, weibull_models in self.weibull_models.items():
            class_probs = np.zeros(len(features))
            
            for point, scale, shape in weibull_models:
                # Compute distances to the current point
                distances = pairwise_distances(
                    features, 
                    point.reshape(1, -1), 
                    metric=self.distance_metric
                ).flatten()
                
                # Inclusion probability: exp(-(d/scale)^shape)
                with np.errstate(over="ignore"):  # (d/scale)^shape -> inf means Psi = 0
                    point_probs = np.exp(-((distances / scale) ** shape))
                
                # Update class probabilities (take maximum probability)
                class_probs = np.maximum(class_probs, point_probs)
            
            probabilities[class_name] = class_probs
        
        return probabilities
    
    def predict(self, features: np.ndarray, threshold: float = 0.5) -> Tuple[List[str], np.ndarray]:
        """Predict class labels for input features"""
        probabilities = self.predict_proba(features)
        
        # Get class with highest probability for each sample
        class_names = list(probabilities.keys())
        prob_matrix = np.column_stack([probabilities[c] for c in class_names])
        
        # Get max probability and corresponding class
        max_probs = np.max(prob_matrix, axis=1)
        max_indices = np.argmax(prob_matrix, axis=1)
        
        # Assign labels based on threshold
        labels = []
        for i, (prob, idx) in enumerate(zip(max_probs, max_indices)):
            if prob >= threshold:
                labels.append(class_names[idx])
            else:
                labels.append("unknown")
        
        return labels, max_probs
    
    def incremental_update(self, new_features: Dict[str, np.ndarray]):
        """Update the model with new classes or examples"""
        # Update class features and means
        for class_name, class_features in new_features.items():
            if class_name in self.class_features:
                # Combine old and new features
                self.class_features[class_name] = np.vstack([
                    self.class_features[class_name],
                    class_features
                ])
            else:
                self.class_features[class_name] = class_features
            
            # Update class mean
            self.class_means[class_name] = np.mean(self.class_features[class_name], axis=0)
        
        # Refit the model with updated features
        self.fit(self.class_features)
        
        return self
