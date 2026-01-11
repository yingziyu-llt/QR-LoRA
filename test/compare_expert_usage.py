import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine, correlation

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Expert Usage between two LoRAs")
    parser.add_argument("--content_npy", type=str, required=True, help="Path to content expert_usage_matrix.npy")
    parser.add_argument("--style_npy", type=str, required=True, help="Path to style expert_usage_matrix.npy")
    parser.add_argument("--output_dir", type=str, default="analysis_result")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载数据 [Layers, Experts]
    # 假设形状是 (303, 4)
    mat_c = np.load(args.content_npy)
    mat_s = np.load(args.style_npy)
    
    if mat_c.shape != mat_s.shape:
        print(f"Warning: Shapes do not match! Content: {mat_c.shape}, Style: {mat_s.shape}")
        # 尝试截断到较小的形状进行对比 (如果层数不一致)
        min_layers = min(mat_c.shape[0], mat_s.shape[0])
        mat_c = mat_c[:min_layers]
        mat_s = mat_s[:min_layers]
    
    print(f"Analyzing matrices of shape: {mat_c.shape}")
    
    # 2. 全局相似度 (Global Similarity)
    # 将矩阵展平为向量，计算余弦相似度
    flat_c = mat_c.flatten()
    flat_s = mat_s.flatten()
    
    # Cosine Similarity = 1 - Cosine Distance
    # 值域 [0, 1]，越接近 0 表示越正交（解耦越好），越接近 1 表示越重叠
    global_sim = 1 - cosine(flat_c, flat_s)
    
    print("-" * 30)
    print(f"Global Cosine Similarity: {global_sim:.4f}")
    print(f"  -> 0.0 means perfect disentanglement (orthogonal usage)")
    print(f"  -> 1.0 means identical usage")
    print("-" * 30)
    
    # 3. 逐层相似度 (Layer-wise Similarity)
    # 看看哪些层解耦得好，哪些层“打架”
    layer_sims = []
    for i in range(mat_c.shape[0]):
        # 防止全0向量除零错误
        if np.all(mat_c[i] == 0) or np.all(mat_s[i] == 0):
            sim = 0.0 # 如果有一方不激活，则视为正交
        else:
            sim = 1 - cosine(mat_c[i], mat_s[i])
        layer_sims.append(sim)
        
    layer_sims = np.array(layer_sims)
    
    print(f"Layer-wise Statistics:")
    print(f"  Mean Similarity: {np.mean(layer_sims):.4f}")
    print(f"  Max Similarity:  {np.max(layer_sims):.4f} (Most conflicted layer)")
    print(f"  Min Similarity:  {np.min(layer_sims):.4f}")
    
    # 4. 专家偏好相关性 (Expert-wise Correlation)
    # 看看 Content 和 Style 是否倾向于使用不同的 Expert
    # 对列（Expert）求平均激活
    avg_c = np.mean(mat_c, axis=0) # [4,]
    avg_s = np.mean(mat_s, axis=0) # [4,]
    
    print("-" * 30)
    print("Average Activation per Expert:")
    print(f"Expert | Content | Style  | Diff (C-S)")
    for i in range(len(avg_c)):
        print(f"Exp {i}  | {avg_c[i]:.4f}  | {avg_s[i]:.4f} | {avg_c[i]-avg_s[i]:.4f}")
        
    # 5. 可视化差异矩阵 (Difference Heatmap)
    # 我们想看 mat_c 和 mat_s 在哪里不同
    # 简单的减法: Content - Style
    # 正值(红)表示 Content 主导，负值(蓝)表示 Style 主导
    diff_map = mat_c - mat_s
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 20))
    sns.heatmap(diff_map, cmap="coolwarm", center=0, cbar_kws={'label': 'Content (+) vs Style (-)'})
    plt.title(f"Usage Difference (Content - Style)\nGlobal Sim: {global_sim:.3f}")
    plt.xlabel("Experts")
    plt.ylabel("Layers")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "usage_difference.png"))
    print(f"Saved difference map to {os.path.join(args.output_dir, 'usage_difference.png')}")

if __name__ == "__main__":
    main()