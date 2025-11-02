"""
Módulo para detecção de rosto usando técnicas de processamento de imagem.
Implementa thresholding de cor de pele e segmentação semeadora.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque


def skin_color_thresholding(image_path):
    """
    Realiza a detecção de pele usando limiar de cor.
    
    Para realizar esta operação, RGB é um bom espaço de cores ou HSV é melhor?
    
    RESPOSTA: HSV é MUITO MELHOR que RGB para detecção de cor de pele. 
    
    Razões:
    1. HSV separa a luminosidade (V) da cor (H, S), tornando mais robusto a variações de iluminação
    2. A cor da pele humana está em uma faixa relativamente estreita de matiz (Hue), 
       independentemente da tonalidade ou brilho
    3. Em RGB, a mesma cor de pele pode ter valores muito diferentes devido à iluminação,
       enquanto em HSV a matiz (H) permanece relativamente constante
    4. O thresholding em HSV é mais intuitivo: trabalhamos com faixas de matiz e saturação
       que são mais próximas da percepção humana de cor
    
    Args:
        image_path (str): Caminho para a imagem
    
    Returns:
        tuple: (imagem_original, imagem_segmentada, mascara_binaria)
            - imagem_original: Imagem RGB original
            - imagem_segmentada: Imagem com apenas pixels de pele destacados
            - mascara_binaria: Máscara binária (0 ou 255) indicando regiões de pele
    """
    # Carrega a imagem
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Erro ao carregar imagem: {image_path}")
    
    # Converte BGR para RGB (para exibição)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Converte BGR para HSV (melhor espaço de cores para detecção de cor)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define os intervalos de cor de pele em HSV
    # Valores típicos para pele humana:
    # Hue (Matiz): 0-50 (cores que vão de vermelho a amarelo)
    # Saturation (Saturação): 48-255 (evita cores muito esbranquiçadas)
    # Value (Brilho): 89-255 (evita sombras muito escuras)
    
    lower_skin = np.array([0, 48, 89], dtype=np.uint8)
    upper_skin = np.array([50, 255, 255], dtype=np.uint8)
    
    # Cria a máscara binária usando thresholding
    mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
    
    # Aplica operações morfológicas para melhorar a máscara
    # Remove ruídos pequenos
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Preenche buracos
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Cria a imagem segmentada aplicando a máscara
    imagem_segmentada = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    return image_rgb, imagem_segmentada, mask


def seeded_segmentation(image_path, seed_point, tolerance=30):
    """
    Implementa Segmentação Semeada (Seeded segmentation).
    
    A segmentação semeada usa um ponto inicial (seed) e expande a região
    baseado na similaridade de pixels vizinhos. Neste caso, usamos a intensidade
    média em execução da pele e uma tolerância para determinar se um pixel vizinho
    deve ser incluído na região.
    
    Args:
        image_path (str): Caminho para a imagem
        seed_point (tuple): Ponto inicial (x, y) na face
        tolerance (int): Tolerância para similaridade de pixels (padrão: 30)
    
    Returns:
        tuple: (imagem_original, imagem_segmentada, mascara_binaria)
            - imagem_original: Imagem RGB original
            - imagem_segmentada: Imagem com apenas a região segmentada destacada
            - mascara_binaria: Máscara binária da região segmentada
    """
    # Carrega a imagem
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Erro ao carregar imagem: {image_path}")
    
    # Converte BGR para RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Converte para escala de cinza para análise de intensidade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    
    # Valida o seed_point
    x, y = seed_point
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(f"Seed point ({x}, {y}) está fora dos limites da imagem ({width}x{height})")
    
    # Inicializa a máscara
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Fila para pixels a serem processados (usando deque para eficiência)
    queue = deque([(x, y)])
    
    # Marca o seed point como visitado
    mask[y, x] = 255
    
    # Intensidade inicial do seed
    seed_intensity = float(gray[y, x])
    running_mean = seed_intensity
    count = 1
    
    # Direções para vizinhança 8-conectada
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    
    # Processa a fila
    while queue:
        cx, cy = queue.popleft()
        
        # Verifica vizinhos
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            
            # Verifica limites
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            
            # Verifica se já foi visitado
            if mask[ny, nx] == 255:
                continue
            
            # Obtém a intensidade do vizinho
            neighbor_intensity = float(gray[ny, nx])
            
            # Verifica se o pixel está dentro da tolerância da média atual
            # Usa a média em execução para adaptar a região
            if abs(neighbor_intensity - running_mean) <= tolerance:
                # Adiciona à região
                mask[ny, nx] = 255
                queue.append((nx, ny))
                
                # Atualiza a média em execução
                running_mean = (running_mean * count + neighbor_intensity) / (count + 1)
                count += 1
    
    # Cria a imagem segmentada aplicando a máscara
    imagem_segmentada = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    return image_rgb, imagem_segmentada, mask


def encontrar_seed_automatico(image_path):
    """
    Função auxiliar para encontrar automaticamente um seed point na região de pele.
    Usa thresholding simples para encontrar uma região de pele e retorna seu centro.
    
    Args:
        image_path (str): Caminho para a imagem
    
    Returns:
        tuple: (x, y) coordenadas do seed point
    """
    # Usa thresholding para encontrar região de pele
    _, _, mask = skin_color_thresholding(image_path)
    
    # Encontra contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Se não encontrou contornos, usa o centro da imagem
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        return (width // 2, height // 2)
    
    # Encontra o maior contorno (geralmente a face)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calcula o centroide do maior contorno
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    else:
        # Fallback: centro da imagem
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        return (width // 2, height // 2)

