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
    parser = argparse.ArgumentParser(description='Analyze the similarity between two SD3 ICA-LoRA weight files')
    parser.add_argument('--lora1_path', type=str, required=True,
                        help='Path to the first LoRA weight file (e.g., Content LoRA)')
    parser.add_argument('--lora2_path', type=str, required=True,
                        help='Path to the second LoRA weight file (e.g., Style LoRA)')
    parser.add_argument('--lora1_name', type=str, default="Content",
                        help='Name label for the first LoRA')
    parser.add_argument('--lora2_name', type=str, default="Style",
                        help='Name label for the second LoRA')
    parser.add_argument('--output_dir', type=str, default="ica_similarity_analysis",
                        help='Output directory for plots and stats')
    parser.add_argument('--fixed_scale', action='store_true',
                        help='Whether to fix the similarity axis range to -0.1 to 1.1')
    return parser.parse_args()

def load_lora_weights(path: str) -> dict:
    """Load LoRA weight file (.safetensors or .pt)"""
    if path.endswith('.safetensors'):
        return safetensors.torch.load_file(path)
    return torch.load(path, map_location="cpu")

def get_matrices_by_type(state_dict: dict, matrix_type: str) -> dict:
    """
    Extract matrices of the specified type.
    For ICA-LoRA:
    - 'S_Matrix':  Saved as 'lora.q.weight' (The Source Matrix)
    - 'Delta_A':   Saved as 'lora.delta_r.weight' (The Delta Mixing Matrix)
    - 'Base_A':    Saved as 'lora.base_r.weight' (The Base Mixing Matrix Transposed)
    """
    matrices = {}
    
    # Mapping ICA concepts to the keys used in save_model_hook
    type_patterns = {
        'S_Matrix': 'lora.q.weight',       # ICA: Source Matrix (Shared/Frozen)
        'Delta_A':  'lora.delta_r.weight', # ICA: Delta Mixing Matrix (Trainable)
        'Base_A':   'lora.base_r.weight'   # ICA: Base Mixing Matrix (Frozen)
    }
    
    pattern = type_patterns.get(matrix_type, matrix_type)
    
    for key, value in state_dict.items():
        if pattern in key:
            # Remove prefix to make keys comparable between models if needed
            matrices[key] = value
    return matrices

# --- Metric Calculators (Integrated from second script) ---

def calculate_cosine_tensor(X, Y):
    """Cosine similarity for flattened tensors."""
    return F.cosine_similarity(X.view(-1), Y.view(-1), dim=0, eps=1e-8)

def calculate_linear_cka(X, Y):
    """
    Computes Linear CKA.
    X, Y should be shape (n_samples, n_features).
    """
    # Ensure 2D
    if X.dim() == 1: X = X.view(-1, 1)
    if Y.dim() == 1: Y = Y.view(-1, 1)

    # Center columns (features)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    numerator = torch.norm(torch.mm(Y.t(), X))**2
    denominator = torch.norm(torch.mm(X.t(), X)) * torch.norm(torch.mm(Y.t(), Y))

    return (numerator / (denominator + 1e-8)).item()

def calculate_nmi_tensor(X, Y, bins=64):
    """
    Computes Normalized Mutual Information (NMI).
    """
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
    """Calculate Cosine, CKA, and MI between two sets of matrices layer by layer"""
    results = {'cosine': {}, 'cka': {}, 'mi': {}}
    
    # Sort keys to ensure consistent order
    common_keys = sorted([k for k in m1_matrices.keys() if k in m2_matrices])
    
    if not common_keys:
        print("Warning: No common keys found for this matrix type!")
        return results

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for metric calculation")

    for key in common_keys:
        t1 = m1_matrices[key].to(device).float()
        t2 = m2_matrices[key].to(device).float()
        
        # Simplify key name for display
        short_key = key.replace("transformer.", "").replace(".lora.q.weight", "") \
                        .replace(".lora.delta_r.weight", "").replace(".lora.base_r.weight", "")

        with torch.no_grad():
            # 1. Cosine
            results['cosine'][short_key] = calculate_cosine_tensor(t1, t2).item()

            # 2. Linear CKA
            # For LoRA weights (usually [rank, dim]), we transpose to [dim, rank]
            # so we compare feature correlation across the rank dimension.
            if t1.ndim == 2 and t1.shape[0] < t1.shape[1]:
                 results['cka'][short_key] = calculate_linear_cka(t1.t(), t2.t())
            else:
                 results['cka'][short_key] = calculate_linear_cka(t1, t2)

            # 3. NMI
            results['mi'][short_key] = calculate_nmi_tensor(t1, t2)

            del t1, t2
            
    return results

def plot_similarities(similarities: dict, output_dir: str, lora1_name: str, 
                      lora2_name: str, matrix_type: str, metric_name: str, fixed_scale: bool = False):
    """Generate Line Plot and Histogram for a specific metric"""
    if not similarities:
        return

    plt.rcParams['font.size'] = 12
    
    layer_names = list(similarities.keys())
    sim_values = list(similarities.values())
    layer_indices = list(range(len(sim_values)))
    
    # 1. Line Plot
    plt.figure(figsize=(24, 8)) 
    plt.plot(layer_indices, sim_values, marker='o', linewidth=1.5, markersize=4, alpha=0.8)
    
    plt.title(f'{metric_name} of {matrix_type}\n({lora1_name} vs {lora2_name})', 
              fontsize=16, pad=15)
    
    step = max(len(layer_indices) // 30, 1)
    plt.xticks(layer_indices[::step], layer_names[::step], rotation=90, fontsize=8)
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    
    # Reference line for Cosine
    if metric_name == 'Cosine':
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1, label="Orthogonal")
        plt.legend()

    if fixed_scale and metric_name in ['Cosine', 'CKA']:
        plt.ylim(-1.1 if metric_name == 'Cosine' else -0.1, 1.1)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    scale_str = "fixed" if fixed_scale else "auto"
    filename_line = f'{matrix_type}_{metric_name}_{scale_str}_line.png'
    plt.savefig(os.path.join(output_dir, filename_line), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(sim_values, bins=30, kde=True)
    if metric_name == 'Cosine':
        plt.axvline(x=0, color='r', linestyle='--', linewidth=1.5)
    
    plt.title(f'Distribution of {metric_name} for {matrix_type}', fontsize=14, pad=15)
    plt.xlabel(metric_name, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    if fixed_scale and metric_name in ['Cosine', 'CKA']:
        plt.xlim(-1.1 if metric_name == 'Cosine' else -0.1, 1.1)
    
    plt.tight_layout()
    filename_hist = f'{matrix_type}_{metric_name}_{scale_str}_hist.png'
    plt.savefig(os.path.join(output_dir, filename_hist), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading weights from:\n1. {args.lora1_path}\n2. {args.lora2_path}")
    lora1_weights = load_lora_weights(args.lora1_path)
    lora2_weights = load_lora_weights(args.lora2_path)
    
    # Analyze three key components of ICA-LoRA (Preserving original structure)
    matrix_types = ['S_Matrix', 'Delta_A', 'Base_A']
    metrics_map = {'Cosine': 'cosine', 'CKA': 'cka', 'MI': 'mi'}
    
    all_summary = {}
    
    for m_type in matrix_types:
        print(f"\n--- Analyzing {m_type} ---")
        
        # 1. Extract
        m1 = get_matrices_by_type(lora1_weights, m_type)
        m2 = get_matrices_by_type(lora2_weights, m_type)
        print(f"Found {len(m1)} layers in LoRA 1, {len(m2)} layers in LoRA 2")
        
        if not m1 or not m2:
            print(f"Skipping {m_type}: matrices missing.")
            continue

        # 2. Calculate All Metrics
        results_bundle = calculate_all_metrics(m1, m2)
        
        all_summary[m_type] = {}

        # 3. Plot & Print Stats for each metric
        for metric_display_name, metric_key in metrics_map.items():
            sims = results_bundle[metric_key]
            
            if not sims:
                continue
                
            plot_similarities(sims, args.output_dir, args.lora1_name, args.lora2_name, 
                              m_type, metric_display_name, args.fixed_scale)
            
            values = list(sims.values())
            avg_val = float(np.mean(values))
            std_val = float(np.std(values))
            
            print(f"  [{metric_display_name}] Mean: {avg_val:.4f} | Std: {std_val:.4f}")
            
            all_summary[m_type][metric_display_name] = {
                "mean": avg_val,
                "std": std_val,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "layer_values": sims
            }

    # Save summary of all types and metrics
    with open(os.path.join(args.output_dir, 'overall_summary.json'), 'w') as f:
        json.dump(all_summary, f, indent=4)
        
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()