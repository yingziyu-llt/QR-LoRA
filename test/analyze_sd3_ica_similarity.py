import torch
import safetensors.torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from torch.nn.functional import cosine_similarity
import numpy as np
import argparse

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

def calculate_cosine_similarities(m1_matrices: dict, m2_matrices: dict) -> dict:
    """Calculate cosine similarity between two sets of matrices layer by layer"""
    similarities = {}
    
    # Sort keys to ensure consistent order
    common_keys = sorted([k for k in m1_matrices.keys() if k in m2_matrices])
    
    if not common_keys:
        print("Warning: No common keys found for this matrix type!")
        return {}

    for key in common_keys:
        # Flatten tensors to vectors for cosine similarity
        m1_flat = m1_matrices[key].float().view(-1)
        m2_flat = m2_matrices[key].float().view(-1)
        
        sim = cosine_similarity(m1_flat, m2_flat, dim=0)
        
        # Simplify key name for display (e.g., remove transformer. and suffixes)
        short_key = key.replace("transformer.", "").replace(".lora.q.weight", "") \
                       .replace(".lora.delta_r.weight", "").replace(".lora.base_r.weight", "")
        similarities[short_key] = sim.item()
        
    return similarities

def plot_similarities(similarities: dict, output_dir: str, lora1_name: str, 
                     lora2_name: str, matrix_type: str, fixed_scale: bool = False):
    """Generate Line Plot and Histogram"""
    if not similarities:
        return

    plt.rcParams['font.size'] = 12
    
    layer_names = list(similarities.keys())
    sim_values = list(similarities.values())
    layer_indices = list(range(len(sim_values)))
    
    # 1. Line Plot (Layer-wise Similarity)
    plt.figure(figsize=(24, 8)) # Wider figure for SD3's many layers
    plt.plot(layer_indices, sim_values, marker='o', linewidth=1.5, markersize=4, alpha=0.8)
    
    plt.title(f'Cosine Similarity of {matrix_type}\n({lora1_name} vs {lora2_name})', 
             fontsize=16, pad=15)
    
    # X-axis formatting: Show fewer labels to avoid crowding
    step = max(len(layer_indices) // 30, 1)
    plt.xticks(layer_indices[::step], layer_names[::step], rotation=90, fontsize=8)
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    
    # Add a reference line at 0 (Orthogonality)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1, label="Orthogonal")
    plt.legend()

    if fixed_scale:
        plt.ylim(-1.1, 1.1) # Cosine similarity range
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    scale_str = "fixed" if fixed_scale else "auto"
    filename_line = f'{matrix_type}_similarity_{scale_str}_line.png'
    plt.savefig(os.path.join(output_dir, filename_line), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Histogram (Distribution)
    plt.figure(figsize=(10, 6))
    sns.histplot(sim_values, bins=30, kde=True)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=1.5)
    
    plt.title(f'Distribution of {matrix_type} Similarity\n({lora1_name} vs {lora2_name})', 
             fontsize=14, pad=15)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    if fixed_scale:
        plt.xlim(-1.1, 1.1)
    
    plt.tight_layout()
    filename_hist = f'{matrix_type}_similarity_{scale_str}_hist.png'
    plt.savefig(os.path.join(output_dir, filename_hist), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Save Statistics to JSON
    stats = {
        "mean": float(np.mean(sim_values)),
        "std": float(np.std(sim_values)),
        "min": float(np.min(sim_values)),
        "max": float(np.max(sim_values)),
        "abs_mean": float(np.mean(np.abs(sim_values))), # Mean of absolute similarity (closer to 0 is better for disentanglement)
        "layer_similarities": similarities
    }
    
    with open(os.path.join(output_dir, f'{matrix_type}_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading weights from:\n1. {args.lora1_path}\n2. {args.lora2_path}")
    lora1_weights = load_lora_weights(args.lora1_path)
    lora2_weights = load_lora_weights(args.lora2_path)
    
    # Analyze three key components of ICA-LoRA
    # S_Matrix: Source components (Should be identical if seed is same)
    # Delta_A:  Task-specific mixing weights (Should be orthogonal/dissimilar for different tasks)
    # Base_A:   Original mixing weights projection (Should be identical)
    matrix_types = ['S_Matrix', 'Delta_A', 'Base_A']
    
    all_summary = {}
    
    for m_type in matrix_types:
        print(f"\n--- Analyzing {m_type} ---")
        
        # 1. Extract
        m1 = get_matrices_by_type(lora1_weights, m_type)
        m2 = get_matrices_by_type(lora2_weights, m_type)
        print(f"Found {len(m1)} layers in LoRA 1, {len(m2)} layers in LoRA 2")
        
        # 2. Calculate
        sims = calculate_cosine_similarities(m1, m2)
        
        if not sims:
            continue
            
        # 3. Plot & Save
        plot_similarities(sims, args.output_dir, args.lora1_name, args.lora2_name, m_type, args.fixed_scale)
        
        # 4. Print Summary
        values = list(sims.values())
        avg_sim = np.mean(values)
        avg_abs_sim = np.mean(np.abs(values))
        
        print(f"Mean Similarity: {avg_sim:.4f}")
        print(f"Mean Absolute Similarity: {avg_abs_sim:.4f} (Lower is better for Delta_A disentanglement)")
        
        all_summary[m_type] = {
            "mean": avg_sim,
            "mean_abs": avg_abs_sim,
            "std": np.std(values)
        }

    # Save summary of all types
    with open(os.path.join(args.output_dir, 'overall_summary.json'), 'w') as f:
        json.dump(all_summary, f, indent=4)
        
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()