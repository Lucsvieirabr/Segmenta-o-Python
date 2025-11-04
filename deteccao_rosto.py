import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque


def skin_color_thresholding(image_path):

    # Define os intervalos de cor de pele em HSV
    # Hue (Matiz): 0-50 (cores que vão de vermelho a amarelo)
    # Saturation (Saturação): 50-255 (evita cores muito esbranquiçadas)
    # Value (Brilho): 80-255 (evita sombras muito escuras)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Erro ao carregar imagem: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_skin = np.array([0, 50, 80], dtype=np.uint8)
    upper_skin = np.array([50, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    imagem_segmentada = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    return image_rgb, imagem_segmentada, mask


def seeded_segmentation(image_path, seed_point, tolerance=30):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Erro ao carregar imagem: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    
    x, y = seed_point
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(f"Seed point ({x}, {y}) está fora dos limites da imagem ({width}x{height})")
    
    mask = np.zeros((height, width), dtype=np.uint8)
    queue = deque([(x, y)])
    mask[y, x] = 255
    
    seed_intensity = float(gray[y, x])
    running_mean = seed_intensity
    count = 1
    
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    
    while queue:
        cx, cy = queue.popleft()
        
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            
            if mask[ny, nx] == 255:
                continue
            
            neighbor_intensity = float(gray[ny, nx])
            
            if abs(neighbor_intensity - running_mean) <= tolerance:
                mask[ny, nx] = 255
                queue.append((nx, ny))
                
                running_mean = (running_mean * count + neighbor_intensity) / (count + 1)
                count += 1
    
    imagem_segmentada = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    return image_rgb, imagem_segmentada, mask


def encontrar_seed_automatico(image_path):
    _, _, mask = skin_color_thresholding(image_path)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        return (width // 2, height // 2)
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    else:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        return (width // 2, height // 2)

