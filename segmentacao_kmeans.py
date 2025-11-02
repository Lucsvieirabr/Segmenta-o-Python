"""
Módulo para segmentação de imagens usando K-means clustering.
Implementa segmentação em espaço RGB e RGB+XY.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


def segmentar_kmeans(image_path, K, dim_type='RGB'):
    """
    Segmenta uma imagem usando K-means clustering.
    
    Args:
        image_path (str): Caminho para a imagem
        K (int): Número de clusters
        dim_type (str): 'RGB' para 3 dimensões ou 'RGB+XY' para 5 dimensões
    
    Returns:
        tuple: (imagem_segmentada, labels, centers)
            - imagem_segmentada: Imagem com pixels substituídos pelos valores dos centros dos clusters
            - labels: Array com os rótulos de cada pixel
            - centers: Array com os centros dos clusters (RGB)
    """
    # Carrega a imagem
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Erro ao carregar imagem: {image_path}")
    
    # Converte BGR para RGB (OpenCV usa BGR por padrão)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    
    # Caso A: RGB apenas (3 dimensões)
    if dim_type == 'RGB':
        # Redimensiona para (pixels, 3) - cada pixel tem valores RGB
        pixels = image_rgb.reshape((-1, 3))
        features = pixels.astype(np.float32)
    
    # Caso B: RGB + XY (5 dimensões)
    elif dim_type == 'RGB+XY':
        # Cria arrays de coordenadas X e Y
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Normaliza as coordenadas para o intervalo [0, 1] para balancear com valores RGB
        x_coords_norm = x_coords.flatten() / width
        y_coords_norm = y_coords.flatten() / height
        
        # Redimensiona RGB para (pixels, 3)
        pixels = image_rgb.reshape((-1, 3))
        
        # Combina RGB com coordenadas XY normalizadas
        features = np.hstack([
            pixels.astype(np.float32),
            x_coords_norm.reshape(-1, 1) * 255,  # Escala para similar ao RGB
            y_coords_norm.reshape(-1, 1) * 255
        ])
    
    else:
        raise ValueError("dim_type deve ser 'RGB' ou 'RGB+XY'")
    
    # Aplica K-means
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Obtém os centros (apenas RGB, mesmo se usamos RGB+XY para clustering)
    if dim_type == 'RGB+XY':
        # Os centros têm 5 dimensões, mas precisamos apenas das primeiras 3 (RGB)
        centers_rgb = kmeans.cluster_centers_[:, :3].astype(np.uint8)
    else:
        centers_rgb = kmeans.cluster_centers_.astype(np.uint8)
    
    # Cria a imagem segmentada substituindo cada pixel pelo centro do seu cluster
    labels_reshaped = labels.reshape(height, width)
    imagem_segmentada = np.zeros_like(image_rgb)
    
    for i in range(K):
        mask = labels_reshaped == i
        imagem_segmentada[mask] = centers_rgb[i]
    
    return imagem_segmentada, labels, centers_rgb


def plot_dispersao_3d(image_path, K, labels, centers):
    """
    Cria um gráfico de dispersão 3D para ilustrar o agrupamento RGB.
    
    Restrição: Este gráfico deve ser gerado apenas para a imagem images/2or4objects.jpg.
    
    Args:
        image_path (str): Caminho para a imagem
        K (int): Número de clusters
        labels (np.array): Rótulos de cada pixel
        centers (np.array): Centros dos clusters (RGB)
    """
    # Verifica se é a imagem correta
    if '2or4objects.jpg' not in image_path:
        print(f"AVISO: plot_dispersao_3d deve ser usado apenas para images/2or4objects.jpg. Imagem recebida: {image_path}")
        return
    
    # Carrega a imagem
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    
    # Redimensiona para array de pixels RGB
    pixels = image_rgb.reshape((-1, 3))
    
    # Garante que labels é um array 1D
    if labels.ndim == 2:
        labels_flat = labels.flatten()
    else:
        labels_flat = labels
    
    # Amostra uma fração dos pixels para visualização (para performance)
    sample_size = min(50000, len(pixels))
    indices = np.random.choice(len(pixels), sample_size, replace=False)
    pixels_sample = pixels[indices]
    labels_sample = labels_flat[indices]
    
    # Cria a figura 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define cores para cada cluster
    try:
        colors_map = plt.cm.get_cmap('tab10', K)
    except AttributeError:
        # Para versões mais recentes do matplotlib
        colors_map = plt.colormaps['tab10'].resampled(K)
    
    # Plota os pixels de cada cluster
    for i in range(K):
        mask = labels_sample == i
        if np.any(mask):
            cluster_pixels = pixels_sample[mask]
            ax.scatter(
                cluster_pixels[:, 0],  # R
                cluster_pixels[:, 1],  # G
                cluster_pixels[:, 2],  # B
                c=[colors_map(i)] * len(cluster_pixels),
                label=f'Cluster {i+1}',
                alpha=0.3,
                s=1
            )
    
    # Plota os centros dos clusters
    ax.scatter(
        centers[:, 0],  # R
        centers[:, 1],  # G
        centers[:, 2],  # B
        c='black',
        marker='x',
        s=200,
        linewidths=3,
        label='Centros'
    )
    
    # Configura os eixos
    ax.set_xlabel('R (Red)', fontsize=12)
    ax.set_ylabel('G (Green)', fontsize=12)
    ax.set_zlabel('B (Blue)', fontsize=12)
    ax.set_title(f'Dispersão 3D RGB - K={K} Clusters\n{image_path}', fontsize=14)
    ax.legend()
    
    # Salva o gráfico
    output_path = f"results/dispersao_3d_{K}clusters.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Gráfico 3D salvo em: {output_path}")
    
    plt.close()

