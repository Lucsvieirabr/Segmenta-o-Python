# Processamento de Imagens: Segmentação e Detecção

🔗 **Repositório GitHub:** [https://github.com/Lucsvieirabr/Segmenta-o-Python.git](https://github.com/Lucsvieirabr/Segmenta-o-Python.git)

## 📋 Sumário

1. [Visão Geral](#visão-geral)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Instalação e Configuração](#instalação-e-configuração)
4. [Como Executar](#como-executar)
5. [Detalhamento Técnico](#detalhamento-técnico)
6. [Escolha das Bibliotecas](#escolha-das-bibliotecas)
7. [Algoritmos Implementados](#algoritmos-implementados)
8. [Resultados Esperados](#resultados-esperados)
9. [Análise e Discussão](#análise-e-discussão)
10. [Conclusões](#conclusões)

---

## 🎯 Visão Geral

Este projeto implementa duas técnicas principais de processamento de imagens:

### **Projeto 1: Segmentação de Objetos usando K-means Clustering**
- Segmentação de maçãs e objetos em imagens usando K-means
- Comparação entre segmentação em espaço RGB (3D) e RGB+XY (5D)
- Visualização 3D dos clusters no espaço RGB

### **Projeto 2: Detecção de Rosto**
- Detecção de pele usando thresholding de cor em espaço HSV
- Segmentação semeadora (seeded segmentation) para extração de regiões de rosto

---

## 📁 Estrutura do Projeto

```
M2_PYTHON/
│
├── images/                          # Diretório com imagens de entrada
│   ├── 2apples.jpg                  # Imagem de 2 maçãs
│   ├── 7apples.jpg                  # Imagem de 7 maçãs
│   ├── variableObjects.jpeg         # Imagem com objetos variáveis
│   ├── 2or4objects.jpg              # Imagem com 2 ou 4 objetos
│   ├── face1.jpg                    # Face 1 para detecção
│   ├── face2.jpg                    # Face 2 para detecção
│   ├── face3.jpeg                   # Face 3 para detecção
│   └── face4.jpg                    # Face 4 para detecção
│
├── results/                         # Diretório para resultados (criado automaticamente)
│   ├── segmentacao_*.png            # Resultados da segmentação K-means
│   ├── dispersao_3d_*.png           # Gráficos 3D de dispersão RGB
│   ├── deteccao_*_thresholding.png # Resultados do thresholding
│   ├── deteccao_*_seeded.png        # Resultados da segmentação semeadora
│   └── DISCUSSAO.txt                # Análise e discussão dos resultados
│
├── main.py                          # Arquivo principal de execução
├── segmentacao_kmeans.py            # Módulo de segmentação K-means
├── deteccao_rosto.py                # Módulo de detecção de rosto
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Este arquivo
```

---

## 🔧 Instalação e Configuração

### Requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo a Passo

#### 1. Clone/Navegue até o diretório do projeto
```bash
cd "/home/lucas/Área de trabalho/M2_PYTHON"
```

#### 2. Crie um ambiente virtual (recomendado)
```bash
python3 -m venv venv
```

Isso cria um ambiente virtual isolado que evita conflitos com outras instalações Python.

**Por que usar ambiente virtual?**
- Isola as dependências do projeto
- Evita conflitos com outras bibliotecas instaladas globalmente
- Permite diferentes versões de bibliotecas para diferentes projetos
- Facilita o compartilhamento do projeto

#### 3. Ative o ambiente virtual

**No Linux/Mac:**
```bash
source venv/bin/activate
```

**No Windows:**
```bash
venv\Scripts\activate
```

Quando ativado, você verá `(venv)` no início do prompt do terminal.

#### 4. Instale as dependências
```bash
pip install -r requirements.txt
```

Isso instalará todas as bibliotecas necessárias:
- `numpy` - Operações numéricas e arrays multidimensionais
- `opencv-python` - Processamento de imagens
- `matplotlib` - Visualização e gráficos
- `scikit-learn` - Algoritmos de machine learning (K-means)

#### 5. Verifique a instalação
```bash
python -c "import cv2, numpy, matplotlib, sklearn; print('Todas as bibliotecas instaladas!')"
```

---

## 🚀 Como Executar

### Método 1: Usando ambiente virtual (RECOMENDADO)

#### Passo 1: Ativar o ambiente virtual

**No Linux/Mac:**
```bash
source venv/bin/activate
```

**No Windows:**
```bash
venv\Scripts\activate
```

Quando ativado, você verá `(venv)` no início do prompt do terminal.

#### Passo 2: Executar o projeto
```bash
python main.py
```

#### Passo 3: Desativar o ambiente virtual (opcional, ao terminar)
```bash
deactivate
```

---

### Método 2: Sem ambiente virtual (não recomendado)

Se preferir instalar globalmente (pode conflitar com outros projetos):
```bash
pip3 install --break-system-packages -r requirements.txt
python3 main.py
```

**⚠️ Nota:** Este método não é recomendado porque pode causar conflitos com outras instalações Python no sistema.

---

### O que acontece durante a execução?

1. **Projeto 1 - Segmentação K-means:**
   - Processa 4 imagens de maçãs/objetos
   - Aplica K-means em RGB (3D) e RGB+XY (5D)
   - Gera gráfico 3D para `2or4objects.jpg`
   - Salva resultados em `results/`

2. **Projeto 2 - Detecção de Rosto:**
   - Processa 4 imagens de faces
   - Aplica thresholding de cor de pele (HSV)
   - Aplica segmentação semeadora
   - Salva resultados em `results/`

3. **Análise e Discussão:**
   - Gera arquivo `results/DISCUSSAO.txt` com análises

---

### Estrutura de Resultados

Todos os resultados são salvos automaticamente no diretório `results/`:

#### Segmentação K-means:
- `segmentacao_[imagem]_K[K]_RGB.png` - Segmentação RGB (3 dimensões)
- `segmentacao_[imagem]_K[K]_RGB+XY.png` - Segmentação RGB+XY (5 dimensões)
- `dispersao_3d_K[K]clusters.png` - Gráfico 3D (apenas para `2or4objects.jpg`)

#### Detecção de Rosto:
- `deteccao_[imagem]_thresholding.png` - Detecção por thresholding de cor
- `deteccao_[imagem]_seeded.png` - Detecção por segmentação semeadora

#### Análise:
- `DISCUSSAO.txt` - Análise e discussão dos resultados (comparações RGB vs RGB+XY e métricas de segmentação)

**Exemplo de arquivos gerados:**
```
results/
├── segmentacao_2apples_K3_RGB.png
├── segmentacao_2apples_K3_RGB+XY.png
├── segmentacao_7apples_K8_RGB.png
├── segmentacao_7apples_K8_RGB+XY.png
├── segmentacao_variableObjects_K5_RGB.png
├── segmentacao_variableObjects_K5_RGB+XY.png
├── segmentacao_2or4objects_K5_RGB.png
├── segmentacao_2or4objects_K5_RGB+XY.png
├── dispersao_3d_5clusters.png
├── deteccao_face1_thresholding.png
├── deteccao_face1_seeded.png
├── deteccao_face2_thresholding.png
├── deteccao_face2_seeded.png
├── deteccao_face3_thresholding.png
├── deteccao_face3_seeded.png
├── deteccao_face4_thresholding.png
├── deteccao_face4_seeded.png
└── DISCUSSAO.txt
```

### Progresso e Feedback

Durante a execução, o programa exibe no terminal:
- Qual imagem está sendo processada
- Valor de K usado para cada imagem
- Método aplicado (RGB, RGB+XY, thresholding, seeded)
- Caminho onde cada resultado foi salvo
- Seed point encontrado para segmentação semeadora

**Exemplo de saída:**
```
============================================================
PROJETO 1: Segmentação de Maçãs com K-means
============================================================

Processando: images/2apples.jpg
Usando K=3 clusters
  - Segmentação RGB (3 dimensões)...
Resultado salvo em: results/segmentacao_2apples_K3_RGB.png
  - Segmentação RGB+XY (5 dimensões)...
Resultado salvo em: results/segmentacao_2apples_K3_RGB+XY.png
```

---

### Solução de Problemas Comuns

**Erro: "No module named 'cv2'" ou outros módulos não encontrados**
- Solução: Certifique-se de que o ambiente virtual está ativado e as dependências estão instaladas
  ```bash
  source venv/bin/activate
  pip install -r requirements.txt
  ```

**Erro: "Permission denied" ao salvar resultados**
- Solução: Verifique as permissões do diretório `results/` ou crie manualmente:
  ```bash
  mkdir -p results
  chmod 755 results
  ```

**Imagens não são encontradas**
- Solução: Verifique se todas as imagens estão no diretório `images/` com os nomes corretos conforme listados em `main.py`

**Resultados não são salvos**
- Solução: Verifique se o diretório `results/` existe e tem permissões de escrita

---

## 🔍 Detalhamento Técnico

### **Arquivo: `main.py`**

Arquivo principal que orquestra a execução dos dois projetos.

#### Funções Principais:

##### `salvar_resultado_segmentacao(imagem_original, imagem_segmentada, nome_arquivo, K, tipo)`

**O que faz:**
- Cria uma figura com duas imagens lado a lado
- Exibe a imagem original e a segmentada para comparação visual
- Salva o resultado em arquivo PNG no diretório `results/`

**Parâmetros:**
- `imagem_original`: Imagem RGB original (array NumPy)
- `imagem_segmentada`: Imagem RGB segmentada pelo K-means
- `nome_arquivo`: Caminho do arquivo original (para extrair nome)
- `K`: Número de clusters usado
- `tipo`: Tipo de segmentação ('RGB' ou 'RGB+XY')

**Por que criar essa função?**
- Padroniza a forma de salvar resultados
- Facilita comparação visual lado a lado
- Organiza os arquivos de saída de forma clara

##### `salvar_resultado_deteccao(imagem_original, imagem_segmentada, mascara, nome_arquivo, metodo)`

**O que faz:**
- Cria uma figura com 3 imagens em linha
- Exibe: original, máscara binária, resultado da detecção
- Salva o resultado em arquivo PNG

**Parâmetros:**
- `imagem_original`: Imagem RGB original
- `imagem_segmentada`: Imagem com apenas região detectada destacada
- `mascara`: Máscara binária (0 ou 255) da região detectada
- `nome_arquivo`: Caminho do arquivo original
- `metodo`: Método usado ('thresholding' ou 'seeded')

**Visualização em 3 painéis:**
1. **Original:** Imagem como está
2. **Máscara:** Visualização binária da região detectada (preto/branco)
3. **Resultado:** Imagem original com apenas a região detectada colorida

##### `executar_projeto1_segmentacao_kmeans()`

**O que faz:**
- Itera sobre todas as imagens de maçãs/objetos definidas em `IMAGENS_MACAS`
- Para cada imagem:
  - Obtém o valor de K apropriado do dicionário `K_VALUES`
  - Carrega a imagem original
  - Chama `segmentar_kmeans()` duas vezes (RGB e RGB+XY)
  - Salva resultados usando `salvar_resultado_segmentacao()`
  - Gera gráfico 3D apenas para `2or4objects.jpg`

**Lógica de valores de K:**
- `2apples.jpg` → K=3 (fundo + 2 maçãs)
- `7apples.jpg` → K=8 (fundo + 7 maçãs)
- `variableObjects.jpeg` → K=5 (fundo + alguns objetos)
- `2or4objects.jpg` → K=5 (fundo + objetos variáveis)

**Por que diferentes valores de K?**
- K deve corresponder ao número de regiões distintas na imagem
- Muito baixo: objetos diferentes podem ser agrupados
- Muito alto: mesmo objeto pode ser dividido em múltiplos clusters

##### `executar_projeto2_deteccao_rosto()`

**O que faz:**
- Itera sobre todas as imagens de faces em `IMAGENS_FACES`
- Para cada imagem:
  - Aplica `skin_color_thresholding()` para método de thresholding
  - Aplica `seeded_segmentation()` com seed point encontrado automaticamente
  - Salva resultados de ambos os métodos

**Fluxo:**
1. Carrega imagem de face
2. Aplica thresholding de cor → obtém máscara binária
3. Encontra seed point automaticamente na região de pele detectada
4. Aplica segmentação semeadora a partir do seed point
5. Salva ambos os resultados para comparação

##### `discutir_resultados()`

**O que faz:**
- Gera uma discussão textual sobre:
  - Comparação entre RGB vs RGB+XY no K-means
  - Métricas de avaliação para segmentação (IoU, Dice, etc.)
- Salva discussão em `results/DISCUSSAO.txt`
- Também imprime no terminal

---

### **Arquivo: `segmentacao_kmeans.py`**

Módulo responsável pela segmentação usando K-means clustering.

#### Funções Principais:

##### `segmentar_kmeans(image_path, K, dim_type='RGB')`

**O que faz:**
- Carrega e prepara a imagem
- Extrai features dependendo do `dim_type`
- Aplica algoritmo K-means usando scikit-learn
- Retorna imagem segmentada, labels e centros dos clusters

**Detalhamento do Processo:**

1. **Carregamento da Imagem:**
```python
image = cv2.imread(image_path)  # Carrega em BGR
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte para RGB
```
   - OpenCV carrega imagens em formato BGR (Blue-Green-Red)
   - Convertemos para RGB para compatibilidade com matplotlib e padrão usual

2. **Extração de Features - Caso A (RGB):**
```python
pixels = image_rgb.reshape((-1, 3))  # (height*width, 3)
features = pixels.astype(np.float32)
```
   - Redimensiona de (height, width, 3) para (N_pixels, 3)
   - Cada linha representa um pixel com valores [R, G, B]
   - 3 dimensões: R, G, B

3. **Extração de Features - Caso B (RGB+XY):**
```python
y_coords, x_coords = np.mgrid[0:height, 0:width]  # Matrizes de coordenadas
x_coords_norm = x_coords.flatten() / width  # Normaliza X para [0, 1]
y_coords_norm = y_coords.flatten() / height  # Normaliza Y para [0, 1]

# Combina RGB + coordenadas normalizadas escaladas para [0, 255]
features = np.hstack([
    pixels.astype(np.float32),
    x_coords_norm.reshape(-1, 1) * 255,
    y_coords_norm.reshape(-1, 1) * 255
])
```
   - 5 dimensões: R, G, B, X, Y
   - Coordenadas são normalizadas e escaladas para [0, 255] para balancear com RGB
   - Por que normalizar? Evita que coordenadas dominem o clustering

4. **Aplicação do K-means:**
```python
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(features)
```
   - `n_clusters=K`: Número de clusters desejado
   - `random_state=42`: Seed para reprodutibilidade
   - `n_init=10`: Executa K-means 10 vezes e escolhe melhor resultado

5. **Construção da Imagem Segmentada:**
```python
labels_reshaped = labels.reshape(height, width)  # Volta para forma 2D
imagem_segmentada = np.zeros_like(image_rgb)

for i in range(K):
    mask = labels_reshaped == i
    imagem_segmentada[mask] = centers_rgb[i]  # Substitui por cor do centro
```
   - Cada pixel recebe a cor do centro do seu cluster
   - Resultado: imagem com K cores distintas

**Retorno:**
- `imagem_segmentada`: Array NumPy (height, width, 3) com cores dos centros
- `labels`: Array NumPy 1D com rótulo de cluster de cada pixel
- `centers_rgb`: Array NumPy (K, 3) com cores RGB dos centros dos clusters

##### `plot_dispersao_3d(image_path, K, labels, centers)`

**O que faz:**
- Cria gráfico de dispersão 3D no espaço RGB
- Mostra como os pixels estão distribuídos e agrupados
- Visualiza os centros dos clusters

**Restrição:** Apenas para `images/2or4objects.jpg` (conforme especificação)

**Detalhamento:**

1. **Verificação de Imagem:**
```python
if '2or4objects.jpg' not in image_path:
    print("AVISO: plot_dispersao_3d deve ser usado apenas para images/2or4objects.jpg")
    return
```

2. **Preparação dos Dados:**
```python
pixels = image_rgb.reshape((-1, 3))  # Todos os pixels RGB

# Amostra para performance (visualizar 50k pixels é suficiente)
sample_size = min(50000, len(pixels))
indices = np.random.choice(len(pixels), sample_size, replace=False)
pixels_sample = pixels[indices]
labels_sample = labels_flat[indices]
```
   - Por que amostrar? Gráficos com milhões de pontos são lentos
   - 50.000 pixels são suficientes para visualizar a distribuição

3. **Criação do Gráfico 3D:**
```python
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plota pixels de cada cluster com cor diferente
for i in range(K):
    cluster_pixels = pixels_sample[labels_sample == i]
    ax.scatter(cluster_pixels[:, 0],  # R
              cluster_pixels[:, 1],   # G
              cluster_pixels[:, 2],   # B
              label=f'Cluster {i+1}',
              alpha=0.3, s=1)  # Transparência e tamanho pequeno

# Plota centros dos clusters em preto (X)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
          c='black', marker='x', s=200, label='Centros')
```
   - Eixos: R (vermelho), G (verde), B (azul)
   - Cada cluster tem cor diferente
   - Centros marcados com X preto grande

4. **Salvamento:**
```python
output_path = f"results/dispersao_3d_{K}clusters.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
```

**Interpretação do Gráfico:**
- Clusters bem separados: boa segmentação
- Clusters sobrepostos: cores similares, pode ser difícil separar
- Centros no meio dos clusters: algoritmo convergiu bem

---

### **Arquivo: `deteccao_rosto.py`**

Módulo responsável pela detecção de rostos usando técnicas de segmentação.

#### Funções Principais:

##### `skin_color_thresholding(image_path)`

**O que faz:**
- Detecta regiões de pele humana usando thresholding baseado em cor
- Usa espaço de cores HSV (não RGB!)
- Retorna imagem original, imagem segmentada e máscara binária

**Por que HSV é melhor que RGB?**

**HSV (Hue-Saturation-Value) separa:**
- **H (Hue/Matiz):** A cor pura (0-360° ou 0-180 em OpenCV)
- **S (Saturation):** Intensidade da cor (0-255)
- **V (Value):** Brilho/Luminosidade (0-255)

**Vantagens do HSV:**
1. **Robustez à iluminação:** Matiz (H) muda pouco com variação de luz
2. **Intuitividade:** Mais próximo da percepção humana de cor
3. **Thresholding eficiente:** Faixas de matiz para pele são estreitas
4. **Separação:** Cor e luminosidade são independentes

**Processo Detalhado:**

1. **Conversão para HSV:**
```python
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

2. **Definição de Intervalos de Cor de Pele:**
```python
lower_skin = np.array([0, 48, 89], dtype=np.uint8)   # HSV mínimo
upper_skin = np.array([50, 255, 255], dtype=np.uint8) # HSV máximo
```

**Interpretação dos intervalos:**
- **H (Hue): 0-50**
  - Cobre tons de vermelho a amarelo (tons de pele)
  - Humanos têm pele em faixa estreita de matiz
  
- **S (Saturation): 48-255**
  - Evita cores muito esbranquiçadas (baixa saturação)
  - Pele tem saturação moderada a alta
  
- **V (Value): 89-255**
  - Evita sombras muito escuras
  - Pele tem brilho moderado a alto

3. **Aplicação do Thresholding:**
```python
mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
```
   - `cv2.inRange()` cria máscara binária:
     - 255 (branco) para pixels dentro do intervalo
     - 0 (preto) para pixels fora do intervalo

4. **Operações Morfológicas:**
```python
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove ruídos
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Preenche buracos
```

**MORPH_OPEN (Abertura):**
- Remove pequenos ruídos (pixels isolados)
- Erosão seguida de dilatação

**MORPH_CLOSE (Fechamento):**
- Preenche pequenos buracos dentro da região
- Dilatação seguida de erosão

**Por que fazer isso?**
- Thresholding pode gerar ruído (pixels isolados classificados como pele)
- Operações morfológicas suavizam a máscara

5. **Aplicação da Máscara:**
```python
imagem_segmentada = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
```
   - `cv2.bitwise_and()` aplica máscara: pixels fora da máscara ficam pretos

**Retorno:**
- `image_rgb`: Imagem original em RGB
- `imagem_segmentada`: Imagem com apenas pixels de pele destacados
- `mask`: Máscara binária (0 ou 255)

##### `seeded_segmentation(image_path, seed_point, tolerance=30)`

**O que faz:**
- Segmentação semeadora (region growing) a partir de um ponto inicial
- Expandindo região baseado em similaridade de intensidade
- Usa média em execução para adaptar o crescimento

**Algoritmo de Region Growing:**

1. **Inicialização:**
```python
queue = deque([(x, y)])  # Fila de pixels a processar
mask[y, x] = 255  # Marca seed como visitado
seed_intensity = float(gray[y, x])  # Intensidade inicial
running_mean = seed_intensity  # Média em execução
count = 1  # Contador de pixels na região
```

2. **Processamento Iterativo:**
```python
while queue:
    cx, cy = queue.popleft()  # Pega próximo pixel da fila
    
    # Verifica vizinhos (8-conectados)
    for dx, dy in directions:
        nx, ny = cx + dx, cy + dy
        
        # Verifica se está dentro da tolerância
        if abs(neighbor_intensity - running_mean) <= tolerance:
            # Adiciona à região
            mask[ny, nx] = 255
            queue.append((nx, ny))
            
            # Atualiza média em execução
            running_mean = (running_mean * count + neighbor_intensity) / (count + 1)
            count += 1
```

**Detalhes Importantes:**

- **8-conectados:** Verifica 8 vizinhos (norte, sul, leste, oeste, diagonais)
- **Média em execução:** Adapta o limiar conforme a região cresce
  - Inicia com intensidade do seed
  - Atualiza conforme novos pixels são adicionados
  - Permite seguir gradientes suaves
  
- **Tolerância (30):**
  - Diferença máxima de intensidade permitida
  - Muito baixa: região não cresce
  - Muito alta: pode incluir áreas diferentes

**Por que usar média em execução?**
- Pele não tem intensidade uniforme (iluminação varia)
- Média em execução permite seguir variações graduais
- Melhor que usar intensidade fixa do seed

##### `encontrar_seed_automatico(image_path)`

**O que faz:**
- Encontra automaticamente um ponto na região de pele
- Usa thresholding para detectar pele primeiro
- Retorna centroide da maior região de pele

**Processo:**
1. Aplica thresholding de pele
2. Encontra contornos (regiões conectadas)
3. Seleciona maior contorno (geralmente a face)
4. Calcula centroide usando momentos
5. Retorna coordenadas (x, y) do centroide

**Por que automatizar?**
- Evita seleção manual de seed point
- Mais prático para processar múltiplas imagens
- Seed no centro da face geralmente funciona bem

---

## 📚 Escolha das Bibliotecas

### **NumPy (`numpy`)**

**O que faz:**
- Arrays multidimensionais eficientes
- Operações matemáticas vetorizadas
- Base para muitas bibliotecas de Python

**Por que escolher?**
- **Performance:** Operações vetorizadas são muito mais rápidas que loops
- **Eficiência de memória:** Arrays NumPy são compactos
- **Comunidade:** Padrão da indústria, muito documentado
- **Integração:** Todas as outras bibliotecas usam NumPy internamente

**Uso no projeto:**
- Armazenar imagens como arrays (height, width, channels)
- Operações matemáticas em pixels
- Manipulação de coordenadas e features

### **OpenCV (`opencv-python`)**

**O que faz:**
- Processamento de imagens e visão computacional
- Carregamento/salvamento de imagens
- Transformações de espaço de cores
- Operações morfológicas
- Detecção de contornos

**Por que escolher?**
- **Completo:** Tudo que precisamos para processamento de imagens
- **Otimizado:** Implementações em C++, muito rápido
- **Amplamente usado:** Padrão da indústria
- **Documentação:** Excelente documentação e exemplos

**Uso no projeto:**
- `cv2.imread()`: Carregar imagens
- `cv2.cvtColor()`: Converter espaços de cores (BGR→RGB, BGR→HSV)
- `cv2.inRange()`: Thresholding baseado em intervalo
- `cv2.morphologyEx()`: Operações morfológicas
- `cv2.findContours()`: Encontrar contornos para seed automático

### **Matplotlib (`matplotlib`)**

**O que faz:**
- Visualização de dados e gráficos
- Plotagem de imagens
- Gráficos 2D e 3D
- Exportação para PNG, PDF, etc.

**Por que escolher?**
- **Flexível:** Diferentes tipos de gráficos
- **Compatível:** Funciona bem com NumPy e imagens
- **Controlável:** Controle fino sobre aparência
- **Padrão:** Biblioteca padrão para visualização em Python

**Uso no projeto:**
- `plt.imshow()`: Exibir imagens
- `plt.scatter()`: Gráfico de dispersão 3D
- `plt.subplots()`: Múltiplas imagens em uma figura
- `plt.savefig()`: Salvar resultados

### **Scikit-learn (`scikit-learn`)**

**O que faz:**
- Algoritmos de machine learning
- Inclui K-means clustering otimizado
- Implementações eficientes e testadas

**Por que escolher?**
- **K-means otimizado:** Implementação eficiente com várias otimizações
- **Confiança:** Biblioteca amplamente usada e testada
- **Facilidade:** API simples e intuitiva
- **Performance:** Implementações otimizadas em C/Cython

**Uso no projeto:**
- `KMeans`: Classe para K-means clustering
- `fit_predict()`: Treinar e obter labels em uma chamada

**Alternativas consideradas:**
- Implementar K-means do zero: mais educativo, mas menos eficiente
- Outras bibliotecas: scikit-learn é padrão e suficiente

---

## 🧮 Algoritmos Implementados

### **K-means Clustering**

**Conceito:**
Algoritmo de clustering não-supervisionado que agrupa dados em K clusters.

**Funcionamento:**

1. **Inicialização:** Escolhe K centros aleatórios (ou usando K-means++)
2. **Atribuição:** Cada ponto é atribuído ao centro mais próximo
3. **Atualização:** Centros são recalculados como média dos pontos do cluster
4. **Iteração:** Repete 2 e 3 até convergência (centros não mudam)

**Matematicamente:**

- **Distância:** Usa distância euclidiana
  ```
  d(p, c) = √[(R_p - R_c)² + (G_p - G_c)² + (B_p - B_c)²]
  ```

- **Atualização de centros:**
  ```
  c_k = (1/|C_k|) * Σ p_i, para p_i ∈ C_k
  ```

- **RGB+XY:** Adiciona coordenadas à distância:
  ```
  d(p, c) = √[(R_p-R_c)² + (G_p-G_c)² + (B_p-B_c)² + (X_p-X_c)² + (Y_p-Y_c)²]
  ```

**Por que normalizar coordenadas?**
- RGB varia em [0, 255], coordenadas X/Y podem ser 0-1000+
- Sem normalização, coordenadas dominariam o clustering
- Normalizando e escalando para [0, 255], RGB e XY têm peso similar

**Complexidade:**
- Tempo: O(n * K * d * i) onde n=pixels, K=clusters, d=dimensões, i=iterações
- Espaço: O(n * d) para armazenar features

### **Region Growing (Seeded Segmentation)**

**Conceito:**
Algoritmo de segmentação que cresce uma região a partir de um ponto inicial (seed).

**Funcionamento:**

1. **Seed:** Ponto inicial na região desejada
2. **Crescimento:** Adiciona vizinhos se similaridade > threshold
3. **Parada:** Quando não há mais vizinhos similares

**Critério de Similaridade:**
```
|I(vizinho) - média_atual| ≤ tolerância
```

**Vizinhos 8-conectados:**
```
[-1, -1]  [-1,  0]  [-1,  1]
[ 0, -1]           [ 0,  1]
[ 1, -1]  [ 1,  0]  [ 1,  1]
```

**Média em Execução:**
- Inicia com intensidade do seed
- Cada novo pixel atualiza a média:
  ```
  média_nova = (média_antiga * n + I_novo) / (n + 1)
  ```
- Permite seguir gradientes graduais

**Vantagens:**
- Simples de implementar
- Adapta-se a variações graduais de intensidade
- Controle fino via tolerância

**Desvantagens:**
- Depende da escolha do seed point
- Pode "vazar" para regiões similares
- Sensível ao valor de tolerância

### **Thresholding Baseado em Cor**

**Conceito:**
Classificação binária: pixel pertence ou não à região (ex: pele).

**Espaço de Cores HSV:**

**Conversão RGB → HSV:**
- Baseada em transformação não-linear
- Separa matiz, saturação e valor

**Thresholding:**
```
Para cada pixel:
  Se (H_min ≤ H ≤ H_max) AND (S_min ≤ S ≤ S_max) AND (V_min ≤ V ≤ V_max):
    pixel = REGIÃO (255)
  Senão:
    pixel = FORA (0)
```

**Intervalos para Pele Humana:**
- Pesquisa mostra pele humana em faixa estreita de matiz
- H: 0-50 (tons vermelho a amarelo)
- S: 48-255 (saturação moderada a alta)
- V: 89-255 (brilho moderado a alto)

---

## 📊 Resultados Esperados

### **Estrutura de Arquivos de Saída:**

```
results/
├── segmentacao_2apples_K3_RGB.png           # Segmentação RGB (2 maçãs)
├── segmentacao_2apples_K3_RGB+XY.png       # Segmentação RGB+XY (2 maçãs)
├── segmentacao_7apples_K8_RGB.png          # Segmentação RGB (7 maçãs)
├── segmentacao_7apples_K8_RGB+XY.png       # Segmentação RGB+XY (7 maçãs)
├── segmentacao_variableObjects_K5_RGB.png
├── segmentacao_variableObjects_K5_RGB+XY.png
├── segmentacao_2or4objects_K5_RGB.png
├── segmentacao_2or4objects_K5_RGB+XY.png
├── dispersao_3d_5clusters.png               # Gráfico 3D (apenas 2or4objects.jpg)
├── deteccao_face1_thresholding.png          # Thresholding (face 1)
├── deteccao_face1_seeded.png                # Seeded (face 1)
├── deteccao_face2_thresholding.png
├── deteccao_face2_seeded.png
├── deteccao_face3_thresholding.png
├── deteccao_face3_seeded.png
├── deteccao_face4_thresholding.png
├── deteccao_face4_seeded.png
└── DISCUSSAO.txt                            # Análise e discussão
```

### **Interpretação dos Resultados:**

#### **Segmentação K-means:**

**RGB vs RGB+XY:**
- **RGB:** Segmenta baseado apenas em cor
  - Objetos da mesma cor são agrupados independente da posição
  - Pode ter "sal e pimenta" (pixels isolados com cores diferentes)
  
- **RGB+XY:** Segmenta baseado em cor + posição
  - Objetos próximos espacialmente tendem a estar no mesmo cluster
  - Resultados mais espacialmente coerentes (regiões contínuas)
  - Melhor para objetos da mesma cor em posições diferentes

**Gráfico 3D:**
- Mostra distribuição dos pixels no espaço RGB
- Clusters bem separados indicam boa segmentação
- Sobreposição indica dificuldade de separar cores

#### **Detecção de Rosto:**

**Thresholding:**
- Geralmente detecta mais áreas (inclui mãos, braços, etc.)
- Pode ter falsos positivos (objetos com cor similar à pele)
- Rápido e simples

**Seeded Segmentation:**
- Mais específico à região ao redor do seed
- Menos falsos positivos se seed está no rosto
- Depende da escolha do seed point

---

## 📝 Análise e Discussão

### **Comparação K-means: RGB vs RGB+XY**

#### **Caso A (RGB apenas):**

**Vantagens:**
- ✅ Simples e computacionalmente eficiente (3 dimensões)
- ✅ Boa para objetos com cores distintas
- ✅ Resultados próximos da segmentação por cor pura
- ✅ Interpretável (apenas cores)

**Desvantagens:**
- ❌ Não separa objetos da mesma cor em posições diferentes
- ❌ Pode agrupar objetos diferentes com cores similares
- ❌ Pode gerar segmentação "sal e pimenta" (pixels isolados)
- ❌ Não considera contexto espacial

**Quando usar?**
- Objetos com cores muito distintas
- Quando queremos segmentar apenas por cor
- Quando performance é crítica

#### **Caso B (RGB + XY):**

**Vantagens:**
- ✅ Melhor para separar objetos da mesma cor em posições diferentes
- ✅ Resultados espacialmente coerentes (regiões contínuas)
- ✅ Reduz efeito "sal e pimenta"
- ✅ Melhor para objetos com cores similares mas fisicamente separados

**Desvantagens:**
- ❌ Pode agrupar objetos diferentes próximos espacialmente
- ❌ Computacionalmente mais caro (5 dimensões vs 3)
- ❌ Pode perder objetos pequenos próximos de objetos maiores
- ❌ Normalização das coordenadas é crítica

**Quando usar?**
- Objetos com cores similares mas separados espacialmente
- Quando queremos segmentação espacialmente coerente
- Quando objetos têm variabilidade de cor dentro da região

**Conclusão:**
- **RGB:** Melhor quando queremos segmentar puramente por cor
- **RGB+XY:** Melhor quando queremos coerência espacial e separar objetos da mesma cor

### **Métricas de Avaliação para Segmentação**

Sem ground truth (verdade absoluta), é difícil avaliar quantitativamente. Discutimos métricas comuns:

#### **1. Intersection over Union (IoU)**

**Definição:**
```
IoU = |A ∩ B| / |A ∪ B|
```
- A = região segmentada
- B = ground truth

**Varia de 0 a 1:**
- 0: Sem sobreposição
- 1: Sobreposição perfeita

**Uso:**
- Padrão da indústria
- Fácil de calcular e interpretar

**Limitação:**
- Requer ground truth manual

#### **2. Dice Coefficient (F1 Score)**

**Definição:**
```
Dice = 2|A ∩ B| / (|A| + |B|)
```

**Similar ao IoU, mas:**
- Mais sensível à sobreposição
- Útil quando há desbalanceamento de classes

#### **3. Pixel Accuracy**

**Definição:**
```
Acc = (TP + TN) / (TP + TN + FP + FN)
```

**Problemas:**
- Pode ser enganoso se classes estão desbalanceadas
- Ex: Se 90% da imagem é fundo, 90% accuracy pode ser trivial

#### **4. Boundary-based Metrics**

**Hausdorff Distance:**
- Mede máxima distância entre boundaries
- Útil para avaliar precisão de contornos

**Boundary IoU:**
- IoU calculado apenas nas bordas
- Importante para aplicações onde bordas são críticas

#### **5. Region-based Metrics**

**Adjusted Rand Index (ARI):**
- Mede concordância entre segmentações
- Não requer ground truth absoluto (comparação entre métodos)

**Variation of Information:**
- Mede informação perdida/ganha entre segmentações
- Útil para comparação qualitativa

#### **6. Métricas Específicas**

**Mean IoU:**
- Média de IoU sobre todas as classes
- Útil para segmentação multi-classe

**Frequency Weighted IoU:**
- Pondera IoU pela frequência das classes
- Evita bias para classes raras

**Limitações Gerais:**
- Todas as métricas quantitativas requerem ground truth
- Para avaliação sem ground truth, precisamos:
  - Métricas de qualidade visual (coerência espacial, suavidade)
  - Métricas de consistência interna
  - Comparação qualitativa entre métodos

---

## 🎓 Conclusões

### **Aprendizados Principais:**

1. **Espaço de Cores Importa:**
   - HSV é superior a RGB para detecção baseada em cor
   - Escolha do espaço de cores depende da aplicação

2. **Features Multidimensionais:**
   - Adicionar contexto espacial (XY) melhora coerência
   - Normalização é crítica para balancear dimensões

3. **Algoritmos Complementares:**
   - Thresholding: Rápido, geral
   - Region Growing: Específico, controlável
   - K-means: Não-supervisionado, flexível

4. **Avaliação Qualitativa:**
   - Sem ground truth, avaliação é principalmente visual
   - Comparação entre métodos fornece insights valiosos

### **Aplicações Práticas:**

- **Segmentação K-means:** Análise de imagens médicas, detecção de objetos, compressão
- **Detecção de Pele:** Sistemas de segurança, análise de gestos, realidade aumentada
- **Region Growing:** Segmentação médica, análise de texturas, isolamento de objetos

### **Melhorias Futuras:**

1. **K-means:**
   - Seleção automática de K (elbow method, silhouette score)
   - Inicialização inteligente (K-means++)
   - Aplicação a vídeo (segmentação temporal)

2. **Detecção de Rosto:**
   - Deep Learning para melhor precisão
   - Multi-escala para faces de tamanhos diferentes
   - Rastreamento em vídeo

3. **Avaliação:**
   - Implementar métricas quantitativas
   - Interface para marcação de ground truth
   - Análise estatística de resultados

---

## 📞 Suporte e Referências

### **Documentação das Bibliotecas:**

- **NumPy:** https://numpy.org/doc/
- **OpenCV:** https://docs.opencv.org/
- **Matplotlib:** https://matplotlib.org/stable/contents.html
- **Scikit-learn:** https://scikit-learn.org/stable/

### **Referências Teóricas:**

- K-means Clustering: "Pattern Recognition and Machine Learning" - Bishop
- Region Growing: "Digital Image Processing" - Gonzalez & Woods
- Espaços de Cores: "Computer Vision: Algorithms and Applications" - Szeliski

### **Problemas Comuns:**

**Erro: "No module named 'cv2'"**
- Solução: Instale opencv-python: `pip install opencv-python`

**Erro: "Permission denied" ao salvar resultados**
- Solução: Verifique permissões do diretório `results/`

**Resultados ruins na segmentação**
- Solução: Ajuste valores de K ou tolerância nos algoritmos

**Imagens não carregam**
- Solução: Verifique caminhos das imagens em `main.py`

---

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais e acadêmicos.

---

**Desenvolvido para:** Processamento de Imagens - M2_PYTHON  
**Data:** 2024  
**Autor:** [Seu Nome]

