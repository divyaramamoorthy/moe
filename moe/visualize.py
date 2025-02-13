import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score
import seaborn as sns

def plot_pca_projection(X, labels, ax=None, title='PCA Projection', cmap='tab10', alpha=0.6):
    """Plot PCA projection of the data."""
    if ax is None:
        fig, ax = plt.subplots()
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    legend = ax.legend(*scatter.legend_elements(), title="Classes", 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
    return scatter, X_pca

def plot_tsne_projection(X, labels, ax=None, title='t-SNE Projection', cmap='tab10', alpha=0.6):
    """Plot t-SNE projection of the data."""
    if ax is None:
        fig, ax = plt.subplots()
    
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap=cmap, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel('First t-SNE Component')
    ax.set_ylabel('Second t-SNE Component')
    legend = ax.legend(*scatter.legend_elements(), title="Classes", 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
    return scatter, X_tsne

def plot_expert_specialization(y_true, expert_assignments, n_experts, ax=None, 
                             title='Expert Specialization', show_metrics=True):
    """Plot expert specialization heatmap."""
    if ax is None:
        fig, ax = plt.subplots()
    
    expert_pattern_counts = np.zeros((n_experts, len(np.unique(y_true))))
    for expert_idx in range(n_experts):
        for pattern_idx in range(len(np.unique(y_true))):
            mask = (expert_assignments == expert_idx) & (y_true == pattern_idx)
            expert_pattern_counts[expert_idx, pattern_idx] = mask.sum()
    
    # Normalize by row, handling zero-sum rows
    row_sums = expert_pattern_counts.sum(axis=1, keepdims=True)
    # Replace zero sums with 1 to avoid division by zero
    row_sums[row_sums == 0] = 1
    expert_pattern_counts = expert_pattern_counts / row_sums
    
    im = ax.imshow(expert_pattern_counts, cmap='YlOrRd')
    
    if show_metrics:
        metrics = calculate_expert_agreement(y_true, expert_assignments)
        title = f"{title}\nNMI: {metrics['nmi']:.3f}, ARI: {metrics['ari']:.3f}"
    
    ax.set_title(title)
    ax.set_xlabel('Pattern Type')
    ax.set_ylabel('Expert ID')
    return im

def plot_pattern_distribution(labels, ax=None, title='Pattern Distribution'):
    """Plot pattern type distribution."""
    if ax is None:
        fig, ax = plt.subplots()
    
    pattern_counts = np.bincount(labels)
    ax.bar(range(len(pattern_counts)), pattern_counts)
    ax.set_title(title)
    ax.set_xlabel('Pattern Type')
    ax.set_ylabel('Count')

def plot_confusion_matrix(y_true, y_pred, ax=None, title='Confusion Matrix', cmap='Blues'):
    """Plot confusion matrix using sklearn for computation and seaborn for visualization."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # Compute confusion matrix and normalize
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Plot using seaborn
    sns.heatmap(cm, 
                annot=True, 
                fmt='.1f', 
                cmap=cmap,
                square=True,
                cbar=True,
                ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    
    return ax.collections[0]  # Return the mappable for colorbar compatibility

def plot_expert_distribution(expert_assignments, ax=None, title='Expert Distribution'):
    """Plot distribution of samples across experts."""
    if ax is None:
        fig, ax = plt.subplots()
    
    expert_counts = np.bincount(expert_assignments)
    ax.bar(range(len(expert_counts)), expert_counts)
    ax.set_title(title)
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Number of Assigned Samples')

def create_visualization_grid(plots_config, figsize=(20, 15)):
    """
    Create a flexible grid of visualizations.
    
    Args:
        plots_config: List of dictionaries, each containing:
            - 'plot_func': Function to create the plot
            - 'grid_pos': Tuple of (row, col) for placement
            - 'kwargs': Dictionary of keyword arguments for the plot function
        figsize: Tuple specifying figure size
    """
    # Calculate grid dimensions based on plot positions
    max_row = max(pos['grid_pos'][0] for pos in plots_config) + 1
    max_col = max(pos['grid_pos'][1] for pos in plots_config) + 1
    
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(max_row, max_col, figure=fig)
    
    for plot_spec in plots_config:
        ax = fig.add_subplot(gs[plot_spec['grid_pos']])
        result = plot_spec['plot_func'](ax=ax, **plot_spec['kwargs'])
        
        # # Add colorbar if the plot returns a mappable object
        # if isinstance(result, tuple) and hasattr(result[0], 'colorbar'):
        #     plt.colorbar(result[0], ax=ax)
        # elif hasattr(result, 'colorbar'):
        #     plt.colorbar(result, ax=ax)
    
    plt.tight_layout()
    return fig

# Example usage:
def data_visualization(X, y, figsize=(10, 7.5)):
    plots_config = [
        {
            'plot_func': plot_pca_projection,
            'grid_pos': (0, 0),
            'kwargs': {'X': X, 'labels': y, 'title': 'PCA Projection by Pattern Type'}
        },
        {
            'plot_func': plot_tsne_projection,
            'grid_pos': (0, 1),
            'kwargs': {'X': X, 'labels': y, 'title': 't-SNE Projection by Pattern Type'}
        },
    ]   
    return create_visualization_grid(plots_config, figsize=figsize)

def training_visualization(X, y_true, y_hat_pre, y_hat_post, figsize=(12,5)):
    plots_config = [
        {
            'plot_func': plot_confusion_matrix,
            'grid_pos': (0, 0),
            'kwargs': {
                'y_true': y_true,
                'y_pred': y_hat_pre,
                'title': 'Confusion Matrix (Pre-training)'
            }
        },
        {
            'plot_func': plot_confusion_matrix,
            'grid_pos': (0, 1),
            'kwargs': {
                'y_true': y_true,
                'y_pred': y_hat_post,
                'title': 'Confusion Matrix (Post-training)'
            }
        },
    ]
    return create_visualization_grid(plots_config, figsize=figsize)

def expert_visualization(X, y_true, pre_assignments, post_assignments, n_experts):
    """Creates a comprehensive visualization combining all plot types."""
    plots_config = [
        {
            'plot_func': plot_tsne_projection,
            'grid_pos': (0, 0),
            'kwargs': {
                'X': X,
                'labels': pre_assignments,
                'title': 'TSNE Projection: Expert Assignments (Pre-training)'
            }
        },
        {
            'plot_func': plot_tsne_projection, 
            'grid_pos': (1, 0),
            'kwargs': {
                'X': X,
                'labels': post_assignments,
                'title': 'TSNE Projection: Expert Assignments (Post-training)'
            }
        },

        {
            'plot_func': plot_expert_specialization,
            'grid_pos': (0, 1),
            'kwargs': {
                'y_true': y_true,
                'expert_assignments': pre_assignments,
                'n_experts': n_experts,
                'title': 'Expert Specialization Heatmap (Pre-training)',
                'show_metrics': True
            }
        },
        {
            'plot_func': plot_expert_specialization,
            'grid_pos': (1, 1),
            'kwargs': {
                'y_true': y_true,
                'expert_assignments': post_assignments,
                'n_experts': n_experts,
                'title': 'Expert Specialization Heatmap (Post-training)',
                'show_metrics': True
            }
        },
        {
            'plot_func': plot_expert_distribution,
            'grid_pos': (0, 2),
            'kwargs': {
                'expert_assignments': post_assignments,
                'title': 'Expert Distribution (Post-training)'
            }
        }
    ]
    
    return create_visualization_grid(plots_config)

def calculate_expert_agreement(y_true, expert_assignments):
    """
    Calculate agreement metrics between true labels and expert assignments.
    
    Args:
        y_true: True labels
        expert_assignments: Expert assignments for each sample
        
    Returns:
        dict: Dictionary containing NMI and ARI scores
    """
    nmi = normalized_mutual_info_score(y_true, expert_assignments)
    ari = adjusted_rand_score(y_true, expert_assignments)
    
    return {
        'nmi': nmi,
        'ari': ari
    }

# # Use the visualization
# fig = create_comprehensive_visualization(
#     X_test, 
#     y_test, 
#     pre_model_idx[:, 0],  # Take first expert assignment
#     post_model_idx[:, 0], 
#     n_clusters
# )
# plt.show()