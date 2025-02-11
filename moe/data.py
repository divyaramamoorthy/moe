"""
Synthetic Data Generation Module for MOE (Mixture of Experts)

This module provides utilities for generating complex synthetic datasets that are
suitable for testing and evaluating Mixture of Experts (MOE) models. The generated
datasets contain various patterns that would benefit from expert specialization,
including linear clusters, Gaussian clusters, nonlinear manifolds, and noise.

The main function `generate_complex_synthetic_data` creates a multi-pattern dataset
that can be used to evaluate how well MOE models learn to specialize different
experts for different types of data patterns.
"""

import numpy as np

def generate_synthetic_data(n_samples_per_pattern=200, dim=10, random_state=42):
    """
    Generates a complex synthetic dataset with multiple patterns that would benefit from expert specialization.
    
    The dataset includes:
    1. Linear clusters with different orientations: Three distinct linear patterns with
       random orientations in the high-dimensional space, each with small Gaussian noise.
    2. Gaussian clusters with different variances: Three Gaussian clusters centered at
       random points with increasing variance (0.1, 0.5, 1.0).
    3. Nonlinear manifolds: Two spiral patterns projected from 2D to higher dimensions,
       representing complex nonlinear relationships.
    4. Noise pattern: Uniform random noise to test robustness of the model.
    
    Args:
        n_samples_per_pattern (int, optional): Number of samples to generate for each pattern.
            Defaults to 200.
        dim (int, optional): Dimensionality of the output space. Must be >= 2.
            Defaults to 10.
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 42.
        
    Returns:
        tuple:
            - X (np.ndarray): Generated points with shape (n_total_samples, dim),
              where n_total_samples = n_samples_per_pattern * n_patterns.
            - labels (np.ndarray): Integer labels for each point indicating which pattern
              generated it. Shape (n_total_samples,). Labels range from 0 to 8,
              corresponding to:
                - 0-2: Linear clusters
                - 3-5: Gaussian clusters
                - 6-7: Spiral patterns
                - 8: Noise pattern
    
    Example:
        >>> X, labels = generate_complex_synthetic_data(n_samples_per_pattern=100, dim=5)
        >>> print(X.shape)
        (900, 5)  # 9 patterns * 100 samples each
        >>> print(np.unique(labels))
        [0 1 2 3 4 5 6 7 8]
    """
    np.random.seed(random_state)
    all_data = []
    all_labels = []
    label_counter = 0
    
    # 1. Linear clusters with different orientations
    for _ in range(3):
        direction = np.random.randn(dim)
        direction /= np.linalg.norm(direction)
        base_points = np.random.randn(n_samples_per_pattern, 1) * 2
        noise = np.random.randn(n_samples_per_pattern, dim) * 0.1
        points = base_points @ direction.reshape(1, -1) + noise
        all_data.append(points)
        all_labels.append(np.full(n_samples_per_pattern, label_counter))
        label_counter += 1
    
    # 2. Gaussian clusters with different variances
    for variance in [0.1, 0.5, 1.0]:
        center = np.random.randn(dim) * 3
        points = np.random.randn(n_samples_per_pattern, dim) * variance + center
        all_data.append(points)
        all_labels.append(np.full(n_samples_per_pattern, label_counter))
        label_counter += 1
    
    # 3. Nonlinear manifolds (spiral-like patterns projected to higher dimensions)
    t = np.linspace(0, 4*np.pi, n_samples_per_pattern)
    for i in range(2):
        # Create spiral in 2D
        spiral_2d = np.column_stack([
            t * np.cos(t + i*np.pi),
            t * np.sin(t + i*np.pi)
        ])
        # Project to higher dimensions using random projection matrix
        projection = np.random.randn(2, dim)
        points = spiral_2d @ projection
        # Add noise
        points += np.random.randn(*points.shape) * 0.2
        all_data.append(points)
        all_labels.append(np.full(n_samples_per_pattern, label_counter))
        label_counter += 1
    
    # 4. Uniform noise pattern
    noise_points = np.random.uniform(-5, 5, size=(n_samples_per_pattern, dim))
    all_data.append(noise_points)
    all_labels.append(np.full(n_samples_per_pattern, label_counter))
    
    # Combine all patterns
    X = np.vstack(all_data)
    labels = np.concatenate(all_labels)
    
    # Shuffle the dataset
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    labels = labels[shuffle_idx]
    
    return X, labels