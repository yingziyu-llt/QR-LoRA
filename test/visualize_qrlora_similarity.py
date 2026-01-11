import torch
import safetensors.torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
import argparse
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the similarity between two DeltaR-LoRA weight files')
    parser.add_argument('--lora1_path', type=str, required=True, help='Path to the first LoRA weight file')
    parser.add_argument('--lora2_path', type=str, required=True, help='Path to the second LoRA weight file')
    parser.add_argument('--lora1_name', type=str, default="style", help='Name of the first LoRA')
    parser.add_argument('--lora2_name', type=str, default="content", help='Name of the second LoRA')
    parser.add_argument('--output_dir', type=str, default="deltaR_similarity_analysis", help='Output directory')
    parser.add_argument('--fixed_scale', action='store_true', help='Whether to fix the similarity axis range to 0-1')
    return parser.parse_args()


def load_lora_weights(path: str) -> dict:
    """Load LoRA weight file"""
    if path.endswith('.safetensors'):
        return safetensors.torch.load_file(path)
    return torch.load(path)


def get_matrices_by_type(state_dict: dict, matrix_type: str) -> dict:
    """Extract matrices of the specified type"""
    matrices = {}
    type_patterns = {'Q': ['lora.q.weight', 'lora.up.weight'], 'deltaR': ['lora.delta_r.weight', 'lora.down.weight'], 'baseR': ['lora.base_r.weight']}
    patterns = type_patterns.get(matrix_type, [])

    for key in state_dict.keys():
        for pattern in patterns:
            if pattern in key:
                matrices[key] = state_dict[key]
                break
    return matrices


# --- Metric Calculators (Refined) ---


def calculate_cosine_tensor(X, Y):
    """Cosine similarity for flattened tensors."""
    # 加上 eps 防止除以零
    return F.cosine_similarity(X.view(-1), Y.view(-1), dim=0, eps=1e-8)


def calculate_linear_cka(X, Y):
    """
    Computes Linear CKA.
    X, Y should be shape (n_samples, n_features).
    For LoRA weights, we recommend passing Transposed weights: (dim, rank).
    This ensures we compare the correlation structure of the 'dim' (neurons).
    """
    # Ensure 2D
    if X.dim() == 1: X = X.view(-1, 1)
    if Y.dim() == 1: Y = Y.view(-1, 1)

    # Center columns (features)
    # Centering matrix H = I - 1/n * 11^T is equivalent to subtracting mean
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Linear HSIC = ||Y^T X||_F^2 / (n-1)^2
    # But for CKA normalization, the denominator cancels out.
    # Formula: CKA = ||Y^T X||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F )

    numerator = torch.norm(torch.mm(Y.t(), X))**2
    denominator = torch.norm(torch.mm(X.t(), X)) * torch.norm(torch.mm(Y.t(), Y))

    return (numerator / (denominator + 1e-8)).item()


def calculate_nmi_tensor(X, Y, bins=64):
    """
    Computes Normalized Mutual Information (NMI).
    NMI = I(X;Y) / sqrt(H(X) * H(Y))
    Range: [0, 1]
    """
    X_flat, Y_flat = X.view(-1), Y.view(-1)

    # 1. Discretize (Binning)
    def get_bins(t, num_bins):
        t_min, t_max = t.min(), t.max()
        # Handle constant array case
        if t_max - t_min < 1e-9:
            return torch.zeros_like(t, dtype=torch.long)
        step = (t - t_min) / (t_max - t_min)
        # Clamp to ensure indices are within [0, num_bins-1]
        indices = (step * num_bins).long().clamp(0, num_bins - 1)
        return indices

    x_ids = get_bins(X_flat, bins)
    y_ids = get_bins(Y_flat, bins)

    # 2. Compute Joint & Marginal Probabilities
    # Trick: use flat indices for 2D histogram
    flat_indices = x_ids * bins + y_ids
    c_xy = torch.bincount(flat_indices, minlength=bins * bins).float()

    # Add smoothing (epsilon) to avoid log(0) issues purely
    c_xy = c_xy + 1e-10

    p_xy = c_xy / c_xy.sum()
    p_xy_mat = p_xy.view(bins, bins)

    p_x = p_xy_mat.sum(dim=1)
    p_y = p_xy_mat.sum(dim=0)

    # 3. Compute Entropies H(X), H(Y)
    h_x = -torch.sum(p_x * torch.log(p_x))
    h_y = -torch.sum(p_y * torch.log(p_y))

    # 4. Compute Mutual Information I(X;Y)
    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    # H(X,Y) = -sum(p_xy * log(p_xy))
    h_xy = -torch.sum(p_xy * torch.log(p_xy))
    mi = h_x + h_y - h_xy

    # 5. Normalize
    # Use geometric mean of entropies
    denominator = torch.sqrt(h_x * h_y)

    if denominator < 1e-8:
        return 0.0

    return (mi / denominator).item()


# --- Aggregator ---


def calculate_all_metrics(m1_matrices: dict, m2_matrices: dict) -> dict:
    results = {'cosine': {}, 'cka': {}, 'mi': {}}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for key in m1_matrices:
        if key not in m2_matrices:
            continue

        t1 = m1_matrices[key].to(device).float()
        t2 = m2_matrices[key].to(device).float()

        with torch.no_grad():
            # 1. Cosine (Flattened)
            results['cosine'][key] = calculate_cosine_tensor(t1, t2).item()

            # 2. Linear CKA (Transposed!)
            # t1 shape: (rank, dim). We want to compare feature correlations (dim).
            # So pass as (dim, rank) -> (samples, features) logic
            # This makes CKA independent of rank permutation but sensitive to feature activation
            if t1.ndim == 2 and t1.shape[0] < t1.shape[1]:
                results['cka'][key] = calculate_linear_cka(t1.t(), t2.t())
            else:
                # Fallback for 1D biases etc.
                results['cka'][key] = calculate_linear_cka(t1, t2)

            # 3. NMI (Flattened)
            results['mi'][key] = calculate_nmi_tensor(t1, t2)

            del t1, t2

    return results


def plot_similarities(similarities: dict, output_dir: str, lora1_name: str, lora2_name: str, matrix_type: str, metric_name: str, fixed_scale: bool = False):
    """Plot similarity visualization for a specific metric"""
    if not similarities:
        return

    plt.rcParams['font.size'] = 12

    layer_names = list(similarities.keys())
    sim_values = list(similarities.values())
    layer_indices = list(range(len(sim_values)))

    # 1. Line plot
    plt.figure(figsize=(24, 6))
    plt.plot(layer_indices, sim_values, marker='o', linewidth=1.5, markersize=3)
    plt.title(f'{metric_name} of {matrix_type} matrices\n({lora1_name} vs {lora2_name})', fontsize=14, pad=15)

    step = max(len(layer_indices) // 20, 1)
    plt.xticks(layer_indices[::step], layer_indices[::step], rotation=45)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)

    if fixed_scale and metric_name in ['Cosine', 'CKA']:
        plt.ylim(-0.1, 1.1)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    scale_type = "fixed" if fixed_scale else "auto"
    plt.savefig(os.path.join(output_dir, f'{matrix_type}_{metric_name}_{scale_type}_line.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(sim_values, bins=20)
    plt.title(f'Distribution of {metric_name} ({matrix_type})\n({lora1_name} vs {lora2_name})', fontsize=14, pad=15)
    plt.xlabel(metric_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    if fixed_scale and metric_name in ['Cosine', 'CKA']:
        plt.xlim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{matrix_type}_{metric_name}_{scale_type}_hist.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    lora1_weights = load_lora_weights(args.lora1_path)
    lora2_weights = load_lora_weights(args.lora2_path)

    matrix_types = ['Q', 'deltaR', 'baseR']
    metrics_map = {'Cosine': 'cosine', 'CKA': 'cka', 'MI': 'mi'}

    all_stats = {}

    for matrix_type in matrix_types:
        m1_matrices = get_matrices_by_type(lora1_weights, matrix_type)
        m2_matrices = get_matrices_by_type(lora2_weights, matrix_type)

        if not m1_matrices or not m2_matrices:
            print(f"Skipping {matrix_type}: matrices not found.")
            continue

        print(f"\nProcessing {matrix_type}...")

        # Calculate all 3 metrics
        results_bundle = calculate_all_metrics(m1_matrices, m2_matrices)

        all_stats[matrix_type] = {}

        for metric_display_name, metric_key in metrics_map.items():
            metric_scores = results_bundle[metric_key]

            if not metric_scores:
                continue

            # Plot
            plot_similarities(metric_scores, args.output_dir, args.lora1_name, args.lora2_name, matrix_type, metric_display_name, args.fixed_scale)

            # Stats
            values = list(metric_scores.values())
            stats = {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values))}
            all_stats[matrix_type][metric_display_name] = stats

            print(f"  [{metric_display_name}] Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f}")

            # Save raw data per metric
            with open(os.path.join(args.output_dir, f'{matrix_type}_{metric_key}_data.json'), 'w') as f:
                json.dump(metric_scores, f, indent=4)

    # Save overall summary
    with open(os.path.join(args.output_dir, 'all_statistics_summary.json'), 'w') as f:
        json.dump(all_stats, f, indent=4)


if __name__ == "__main__":
    main()
