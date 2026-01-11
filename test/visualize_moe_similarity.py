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
    parser = argparse.ArgumentParser(description='Analyze the similarity between two Freq-MoE LoRA weight files')
    parser.add_argument('--lora1_path', type=str, required=True, help='Path to the first LoRA weight file')
    parser.add_argument('--lora2_path', type=str, required=True, help='Path to the second LoRA weight file')
    parser.add_argument('--lora1_name', type=str, default="Model A", help='Name of the first LoRA')
    parser.add_argument('--lora2_name', type=str, default="Model B", help='Name of the second LoRA')
    parser.add_argument('--output_dir', type=str, default="moe_similarity_analysis", help='Output directory')
    parser.add_argument('--fixed_scale', action='store_true', help='Whether to fix the similarity axis range to 0-1')
    return parser.parse_args()

def load_lora_weights(path: str) -> dict:
    if path.endswith('.safetensors'):
        return safetensors.torch.load_file(path)
    return torch.load(path, map_location="cpu")

def get_matrices_by_type(state_dict: dict, matrix_type: str) -> dict:
    matrices = {}
    # Definitions based on train_qrlora_SD3_MoE.py save_model_hook
    type_patterns = {
        'Gate':   ['gate.weight'],
        'Q':      ['lora.q.weight'],
        'DeltaR': ['lora.delta_r.weight'],
        'BaseR':  ['lora.base_r.weight']
    }
    patterns = type_patterns.get(matrix_type, [])

    for key in state_dict.keys():
        for pattern in patterns:
            if pattern in key:
                matrices[key] = state_dict[key]
                break
    return matrices

# --- Metric Calculators ---

def calculate_cosine_tensor(X, Y):
    return F.cosine_similarity(X.view(-1), Y.view(-1), dim=0, eps=1e-8)

def calculate_linear_cka(X, Y):
    # Ensure 2D
    if X.dim() == 1: X = X.view(-1, 1)
    if Y.dim() == 1: Y = Y.view(-1, 1)

    # Center columns
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    numerator = torch.norm(torch.mm(Y.t(), X))**2
    denominator = torch.norm(torch.mm(X.t(), X)) * torch.norm(torch.mm(Y.t(), Y))

    return (numerator / (denominator + 1e-8)).item()

def calculate_nmi_tensor(X, Y, bins=64):
    X_flat, Y_flat = X.view(-1), Y.view(-1)

    def get_bins(t, num_bins):
        t_min, t_max = t.min(), t.max()
        if t_max - t_min < 1e-9:
            return torch.zeros_like(t, dtype=torch.long)
        step = (t - t_min) / (t_max - t_min)
        indices = (step * num_bins).long().clamp(0, num_bins - 1)
        return indices

    x_ids = get_bins(X_flat, bins)
    y_ids = get_bins(Y_flat, bins)

    flat_indices = x_ids * bins + y_ids
    c_xy = torch.bincount(flat_indices, minlength=bins * bins).float()
    c_xy = c_xy + 1e-10

    p_xy = c_xy / c_xy.sum()
    p_xy_mat = p_xy.view(bins, bins)

    p_x = p_xy_mat.sum(dim=1)
    p_y = p_xy_mat.sum(dim=0)

    h_x = -torch.sum(p_x * torch.log(p_x))
    h_y = -torch.sum(p_y * torch.log(p_y))
    h_xy = -torch.sum(p_xy * torch.log(p_xy))
    
    mi = h_x + h_y - h_xy
    denominator = torch.sqrt(h_x * h_y)

    if denominator < 1e-8:
        return 0.0

    return (mi / denominator).item()

# --- Aggregator ---

def calculate_all_metrics(m1_matrices: dict, m2_matrices: dict) -> dict:
    results = {'cosine': {}, 'cka': {}, 'mi': {}}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Sort keys to ensure layer order
    common_keys = sorted([k for k in m1_matrices.keys() if k in m2_matrices])

    for key in common_keys:
        t1 = m1_matrices[key].to(device).float()
        t2 = m2_matrices[key].to(device).float()

        # Simplify key name for display
        short_key = key.replace("transformer.", "").replace(".lora.q.weight", "") \
                        .replace(".lora.delta_r.weight", "").replace(".lora.base_r.weight", "") \
                        .replace(".gate.weight", "")

        with torch.no_grad():
            # 1. Cosine
            results['cosine'][short_key] = calculate_cosine_tensor(t1, t2).item()

            # 2. Linear CKA (Transpose for (rank, dim) -> (dim, rank))
            # For Gate weights (experts, dim), transpose means comparing expert activation patterns
            if t1.ndim == 2 and t1.shape[0] < t1.shape[1]:
                results['cka'][short_key] = calculate_linear_cka(t1.t(), t2.t())
            else:
                results['cka'][short_key] = calculate_linear_cka(t1, t2)

            # 3. NMI
            results['mi'][short_key] = calculate_nmi_tensor(t1, t2)

            del t1, t2

    return results

def plot_similarities(similarities: dict, output_dir: str, lora1_name: str, lora2_name: str, matrix_type: str, metric_name: str, fixed_scale: bool = False):
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
    plt.xticks(layer_indices[::step], layer_names[::step], rotation=90, fontsize=8)
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
    plt.title(f'Distribution of {metric_name} ({matrix_type})', fontsize=14, pad=15)
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

    print(f"Loading weights from:\n1. {args.lora1_path}\n2. {args.lora2_path}")
    lora1_weights = load_lora_weights(args.lora1_path)
    lora2_weights = load_lora_weights(args.lora2_path)

    # Added 'Gate' to the list of types to analyze
    matrix_types = ['Gate', 'DeltaR', 'Q', 'BaseR']
    metrics_map = {'Cosine': 'cosine', 'CKA': 'cka', 'MI': 'mi'}

    all_stats = {}

    for matrix_type in matrix_types:
        m1_matrices = get_matrices_by_type(lora1_weights, matrix_type)
        m2_matrices = get_matrices_by_type(lora2_weights, matrix_type)

        if not m1_matrices or not m2_matrices:
            print(f"Skipping {matrix_type}: matrices not found.")
            continue

        print(f"\nProcessing {matrix_type} ({len(m1_matrices)} layers)...")

        results_bundle = calculate_all_metrics(m1_matrices, m2_matrices)
        all_stats[matrix_type] = {}

        for metric_display_name, metric_key in metrics_map.items():
            metric_scores = results_bundle[metric_key]
            if not metric_scores:
                continue

            plot_similarities(metric_scores, args.output_dir, args.lora1_name, args.lora2_name, matrix_type, metric_display_name, args.fixed_scale)

            values = list(metric_scores.values())
            stats = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
            all_stats[matrix_type][metric_display_name] = stats
            print(f"  [{metric_display_name}] Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f}")

            with open(os.path.join(args.output_dir, f'{matrix_type}_{metric_key}_data.json'), 'w') as f:
                json.dump(metric_scores, f, indent=4)

    with open(os.path.join(args.output_dir, 'moe_statistics_summary.json'), 'w') as f:
        json.dump(all_stats, f, indent=4)
        
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()