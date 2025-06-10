import os
import pickle as pkl
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image

class CosineSimilarityKMeans:
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx]
        
        for _ in range(self.max_iter):
            similarities = cosine_similarity(X, self.cluster_centers_)
            new_labels = np.argmax(similarities, axis=1)
            
            old_centers = self.cluster_centers_.copy()
            for i in range(self.n_clusters):
                cluster_samples = X[new_labels == i]
                if len(cluster_samples) > 0:
                    self.cluster_centers_[i] = cluster_samples.mean(axis=0)
                    self.cluster_centers_[i] /= np.linalg.norm(self.cluster_centers_[i])
            
            if np.allclose(old_centers, self.cluster_centers_):
                break
                
        return new_labels, similarities, torch.from_numpy(self.cluster_centers_)

def cluster_features_per_class(k_list):
    with open("./features/image_features/bcss_features_pro.pkl", 'rb') as f:
        features_dict = pkl.load(f)
    
    if len(k_list) != 4:
        raise ValueError("k_list must contain 4 values for TUM, STR, LYM, and NEC respectively")
    
    all_centers = []
    class_order = ['TUM', 'STR', 'LYM', 'NEC']
    
    for class_name, k in zip(class_order, k_list):
        print(f"\n{'='*20} Class: {class_name} (k={k}) {'='*20}")
        
        class_features = []
        class_names = []
        
        for item in features_dict[class_name]:
            class_features.append(item['features'].squeeze())
            class_names.append(item['name'])
        
        features_array = np.array(class_features)
        features_norm = features_array / np.linalg.norm(features_array, axis=1, keepdims=True)
        
        kmeans = CosineSimilarityKMeans(n_clusters=k, random_state=42)
        cluster_labels, similarities, cluster_centers = kmeans.fit_predict(features_norm)
        
        all_centers.append(cluster_centers)
        
        for cluster_idx in range(k):
            cluster_mask = cluster_labels == cluster_idx
            cluster_similarities = similarities[cluster_mask][:, cluster_idx]
            
            top_5_indices = np.argsort(cluster_similarities)[-5:][::-1]
            cluster_sample_indices = np.where(cluster_mask)[0][top_5_indices]
            
            print(f"\nCluster {cluster_idx + 1} - Top 5 closest images to center:")
            for idx, sample_idx in enumerate(cluster_sample_indices, 1):
                similarity = similarities[sample_idx, cluster_idx]
                print(f"{idx}. {class_names[sample_idx]} (similarity: {similarity:.4f})")
            
            cluster_size = np.sum(cluster_labels == cluster_idx)
            print(f"Total samples in cluster: {cluster_size}")
    
    all_centers_tensor = torch.cat(all_centers, dim=0)
    
    save_info = {
        'features': all_centers_tensor,
        'k_list': k_list,
        'class_order': class_order,
        'cumsum_k': np.cumsum([0] + k_list) 
    }
    
    save_dir = "./features/image_features"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"bcss_label_fea_pro_{k_list[0]}{k_list[1]}{k_list[2]}{k_list[3]}.pkl")
    
    with open(save_path, 'wb') as f:
        pkl.dump(save_info, f)
    
    print(f"\nInformation saved to {save_path}")
    print(f"Feature tensor shape: {all_centers_tensor.shape}")
    print(f"K list: {k_list}")
    print(f"Cumulative sum of k: {save_info['cumsum_k']}")
    print("Class features index ranges:")
    for i, class_name in enumerate(class_order):
        start_idx = save_info['cumsum_k'][i]
        end_idx = save_info['cumsum_k'][i+1]
        print(f"{class_name}: {start_idx} to {end_idx}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_list', type=int, nargs=4, default=[3,3,3,3],
                      help='Number of clusters for each class [TUM, STR, LYM, NEC]')
    args = parser.parse_args()
    
    cluster_features_per_class(k_list=args.k_list) 