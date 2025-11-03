import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


def segmentar_kmeans(image_path, K, dim_type='RGB'):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Erro ao carregar imagem: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    
    if dim_type == 'RGB':
        pixels = image_rgb.reshape((-1, 3))
        features = pixels.astype(np.float32)
    
    elif dim_type == 'RGB+XY':
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        x_coords_norm = x_coords.flatten() / width
        y_coords_norm = y_coords.flatten() / height
        
        pixels = image_rgb.reshape((-1, 3))
        
        features = np.hstack([
            pixels.astype(np.float32),
            x_coords_norm.reshape(-1, 1) * 255,
            y_coords_norm.reshape(-1, 1) * 255
        ])
    
    else:
        raise ValueError("dim_type deve ser 'RGB' ou 'RGB+XY'")
    
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    if dim_type == 'RGB+XY':
        centers_rgb = kmeans.cluster_centers_[:, :3].astype(np.uint8)
    else:
        centers_rgb = kmeans.cluster_centers_.astype(np.uint8)
    
    labels_reshaped = labels.reshape(height, width)
    imagem_segmentada = np.zeros_like(image_rgb)
    
    for i in range(K):
        mask = labels_reshaped == i
        imagem_segmentada[mask] = centers_rgb[i]
    
    return imagem_segmentada, labels, centers_rgb


def plot_dispersao_3d(image_path, K, labels, centers):
    if '2or4objects.jpg' not in image_path:
        print(f"AVISO: plot_dispersao_3d deve ser usado apenas para images/2or4objects.jpg. Imagem recebida: {image_path}")
        return
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    
    pixels = image_rgb.reshape((-1, 3))
    
    if labels.ndim == 2:
        labels_flat = labels.flatten()
    else:
        labels_flat = labels
    
    sample_size = min(50000, len(pixels))
    indices = np.random.choice(len(pixels), sample_size, replace=False)
    pixels_sample = pixels[indices]
    labels_sample = labels_flat[indices]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    try:
        colors_map = plt.cm.get_cmap('tab10', K)
    except AttributeError:
        colors_map = plt.colormaps['tab10'].resampled(K)
    
    for i in range(K):
        mask = labels_sample == i
        if np.any(mask):
            cluster_pixels = pixels_sample[mask]
            ax.scatter(
                cluster_pixels[:, 0],
                cluster_pixels[:, 1],
                cluster_pixels[:, 2],
                c=[colors_map(i)] * len(cluster_pixels),
                label=f'Cluster {i+1}',
                alpha=0.3,
                s=1
            )
    
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c='black',
        marker='x',
        s=200,
        linewidths=3,
        label='Centros'
    )
    
    ax.set_xlabel('R (Red)', fontsize=12)
    ax.set_ylabel('G (Green)', fontsize=12)
    ax.set_zlabel('B (Blue)', fontsize=12)
    ax.set_title(f'Dispersão 3D RGB - K={K} Clusters\n{image_path}', fontsize=14)
    ax.legend()
    
    output_path = f"results/dispersao_3d_{K}clusters.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Gráfico 3D salvo em: {output_path}")
    
    plt.close()

