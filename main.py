import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from segmentacao_kmeans import segmentar_kmeans, plot_dispersao_3d
from deteccao_rosto import skin_color_thresholding, seeded_segmentation, encontrar_seed_automatico

IMAGENS_MACAS = [
    'images/2apples.jpg',
    'images/7apples.jpg',
    'images/variableObjects.jpeg',
    'images/2or4objects.jpg'
]

IMAGENS_FACES = [
    'images/face1.jpg',
    'images/face2.jpg',
    'images/face3.jpeg',
    'images/face4.jpg'
]

K_VALUES = {
    'images/2apples.jpg': 3,
    'images/7apples.jpg': 8,
    'images/variableObjects.jpeg': 5,
    'images/2or4objects.jpg': 5
}


def salvar_resultado_segmentacao(imagem_original, imagem_segmentada, nome_arquivo, K, tipo):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(imagem_original)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    axes[1].imshow(imagem_segmentada)
    axes[1].set_title(f'Segmentação K-means (K={K}, {tipo})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    nome_base = os.path.splitext(os.path.basename(nome_arquivo))[0]
    output_path = f"results/segmentacao_{nome_base}_K{K}_{tipo}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Resultado salvo em: {output_path}")
    
    plt.close()


def salvar_resultado_deteccao(imagem_original, imagem_segmentada, mascara, nome_arquivo, metodo):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(imagem_original)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    axes[1].imshow(mascara, cmap='gray')
    axes[1].set_title('Máscara Binária')
    axes[1].axis('off')
    
    axes[2].imshow(imagem_segmentada)
    axes[2].set_title(f'Detecção de Rosto - {metodo.capitalize()}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    nome_base = os.path.splitext(os.path.basename(nome_arquivo))[0]
    output_path = f"results/deteccao_{nome_base}_{metodo}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Resultado salvo em: {output_path}")
    
    plt.close()


def executar_projeto1_segmentacao_kmeans():
    print("\n" + "="*60)
    print("PROJETO 1: Segmentação de Maçãs com K-means")
    print("="*60)
    
    for imagem_path in IMAGENS_MACAS:
        if not os.path.exists(imagem_path):
            print(f"AVISO: Imagem não encontrada: {imagem_path}")
            continue
        
        print(f"\nProcessando: {imagem_path}")
        
        K = K_VALUES.get(imagem_path, 3)
        print(f"Usando K={K} clusters")
        
        image = cv2.imread(imagem_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"  - Segmentação RGB (3 dimensões)...")
        imagem_seg_rgb, labels_rgb, centers_rgb = segmentar_kmeans(imagem_path, K, dim_type='RGB')
        salvar_resultado_segmentacao(image_rgb, imagem_seg_rgb, imagem_path, K, 'RGB')
        
        print(f"  - Segmentação RGB+XY (5 dimensões)...")
        imagem_seg_xy, labels_xy, centers_xy = segmentar_kmeans(imagem_path, K, dim_type='RGB+XY')
        salvar_resultado_segmentacao(image_rgb, imagem_seg_xy, imagem_path, K, 'RGB+XY')
        
        if '2or4objects.jpg' in imagem_path:
            print(f"  - Gerando gráfico de dispersão 3D...")
            plot_dispersao_3d(imagem_path, K, labels_rgb.reshape(image_rgb.shape[:2]), centers_rgb)
    
    print("\nProjeto 1 concluído!")


def executar_projeto2_deteccao_rosto():
    print("\n" + "="*60)
    print("PROJETO 2: Detecção de Rosto")
    print("="*60)
    
    for imagem_path in IMAGENS_FACES:
        if not os.path.exists(imagem_path):
            print(f"AVISO: Imagem não encontrada: {imagem_path}")
            continue
        
        print(f"\nProcessando: {imagem_path}")
        
        print(f"  - Thresholding de cor de pele...")
        img_orig_thresh, img_seg_thresh, mask_thresh = skin_color_thresholding(imagem_path)
        salvar_resultado_deteccao(img_orig_thresh, img_seg_thresh, mask_thresh, imagem_path, 'thresholding')
        
        print(f"  - Segmentação semeadora...")
        seed_point = encontrar_seed_automatico(imagem_path)
        print(f"    Seed point encontrado: {seed_point}")
        
        img_orig_seeded, img_seg_seeded, mask_seeded = seeded_segmentation(
            imagem_path, seed_point, tolerance=30
        )
        salvar_resultado_deteccao(img_orig_seeded, img_seg_seeded, mask_seeded, imagem_path, 'seeded')
    
    print("\nProjeto 2 concluído!")


def discutir_resultados():
    print("\n" + "="*60)
    print("ANÁLISE E DISCUSSÃO")
    print("="*60)
    
    discussao = """
    
=== 1. COMPARAÇÃO K-means: RGB vs RGB+XY ===

Caso A (RGB apenas):
- Usa apenas as características de cor (R, G, B) de cada pixel
- Vantagens:
  * Simples e computacionalmente eficiente
  * Boa para segmentar objetos com cores distintas
  * Resultados mais próximos da segmentação por cor pura
  
- Desvantagens:
  * Pode não separar bem objetos da mesma cor em posições diferentes
  * Pode agrupar objetos diferentes que têm cores similares mas estão distantes
  * Não considera proximidade espacial

Caso B (RGB + XY):
- Usa cor (RGB) mais posição espacial (X, Y) de cada pixel
- Vantagens:
  * Melhor para separar objetos da mesma cor que estão em posições diferentes
  * Resultados mais espacialmente coerentes (regiões contínuas)
  * Reduz o "efeito sal e pimenta" na segmentação
  * Melhor para objetos que têm cores similares mas estão fisicamente separados
  
- Desvantagens:
  * Pode agrupar objetos diferentes que estão próximos espacialmente
  * Computacionalmente mais caro (5 dimensões vs 3)
  * Pode perder objetos pequenos que estão próximos de outros maiores
  * A normalização das coordenadas é crítica para balancear peso de cor vs posição

Conclusão:
- RGB é melhor quando: queremos segmentar baseado puramente em cor
- RGB+XY é melhor quando: queremos segmentação espacialmente coerente e objetos da mesma cor podem estar separados


=== 2. MÉTRICAS DE AVALIAÇÃO PARA SEGMENTAÇÃO (sem Deep Learning) ===

2.1. Intersection over Union (IoU):
- Mede a sobreposição entre região segmentada e ground truth
- IoU = |A ∩ B| / |A ∪ B|
- Varia de 0 (sem sobreposição) a 1 (perfeita sobreposição)
- Vantagem: Simples, amplamente usado, intuitivo
- Desvantagem: Requer ground truth manual

2.2. Dice Coefficient (F1 Score para segmentação):
- Similar ao IoU, mas dá mais peso à sobreposição
- Dice = 2|A ∩ B| / (|A| + |B|)
- Mais sensível à sobreposição que IoU
- Útil quando temos desbalanceamento de classes

2.3. Pixel Accuracy:
- Porcentagem de pixels classificados corretamente
- Acc = (TP + TN) / (TP + TN + FP + FN)
- Simples, mas pode ser enganoso se classes estão desbalanceadas

2.4. Boundary-based Metrics:
- Hausdorff Distance: Mede a máxima distância entre boundaries
- Boundary IoU: IoU calculado apenas nas bordas
- Útil para avaliar precisão de contornos

2.5. Region-based Metrics:
- Adjusted Rand Index (ARI): Mede concordância entre segmentações
- Variation of Information: Mede informação perdida/ganha entre segmentações
- Não requer ground truth absoluto

2.6. Métricas específicas para segmentação semântica:
- Mean IoU: Média de IoU sobre todas as classes
- Frequency Weighted IoU: Pondera por frequência de classes

Limitações:
- Todas essas métricas requerem ground truth para avaliação objetiva
- Para avaliação sem ground truth, podemos usar métricas de qualidade visual ou
  consistência interna (coerência espacial, suavidade de bordas)
- Métricas automáticas sem ground truth são difíceis e geralmente menos confiáveis

"""
    
    print(discussao)
    
    with open("results/DISCUSSAO.txt", "w", encoding="utf-8") as f:
        f.write(discussao)
    print("\nDiscussão salva em: results/DISCUSSAO.txt")


def main():
    print("="*60)
    print("PROCESSAMENTO DE IMAGENS - PROJETOS 1 E 2")
    print("="*60)
    
    os.makedirs("results", exist_ok=True)
    
    executar_projeto1_segmentacao_kmeans()
    
    executar_projeto2_deteccao_rosto()
    
    discutir_resultados()
    
    print("\n" + "="*60)
    print("TODOS OS PROJETOS CONCLUÍDOS!")
    print("Resultados salvos no diretório: results/")
    print("="*60)


if __name__ == "__main__":
    main()

