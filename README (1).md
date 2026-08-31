# MVP de SSD — Risco, Custo e Tempo para Implantação de Nova Unidade de Controle de Dados

**Disciplina:** Sistemas de Suporte à Decisão
**Universidade de Brasília (UnB)**
**Aluna:** Ana Luisa Breide Pessôa Guerra — 231013289

MVP de **apoio à decisão para a implantação de uma nova unidade de controle de dados**, construído sobre uma base de **10.000 registros de telemetria** de data centers (*Green AI Data Center Telemetry*).

## O Problema de Decisão

Antes de investir na abertura de uma nova unidade, uma empresa precisa responder a três perguntas:

| Eixo | Pergunta de decisão | Variável / Indicador |
|---|---|---|
| **RISCO** | Qual o nível de risco operacional esperado para a configuração escolhida? | `risk_score` — índice composto derivado de PUE, carbono, temperatura e eficiência de resfriamento |
| **CUSTO** | Quanto custa operar cada configuração? | `compute_cost_usd`, `power_consumption_kw`, `pue` |
| **TEMPO** | Em quanto tempo a unidade pode ser implantada? | Não existe na base — estimado por modelo heurístico (ver abaixo) |

A empresa controla três decisões no momento do projeto: **tipo de servidor**, **tipo de resfriamento** e **região**.

## Hipótese

As três decisões de projeto, tomadas **antes de qualquer servidor ser ligado**, junto com um índice de risco construído sobre a telemetria basal e um modelo de custo treinado sobre os dados reais, contêm informação suficiente para orientar a escolha da configuração da nova unidade — mesmo a base não contendo uma variável direta de tempo de implantação.

Para testar isso, o notebook:
1. constrói um **índice de risco (0–100)** a partir de fatores de telemetria com peso explícito e interpretável (não é uma classificação treinada, e sim um score de engenharia de risco, adequado quando não há rótulo histórico de falha na base);
2. treina um **modelo de custo** (Random Forest) sobre as variáveis de telemetria disponíveis;
3. estima o **tempo de implantação** com um modelo heurístico parametrizado, combinando prazos-base por tipo de servidor/resfriamento, fator logístico regional e ajustes proporcionais ao risco e ao custo da configuração;
4. combina os três eixos em uma **matriz de decisão multicritério (MCDA)**, testando a robustez do resultado por análise de sensibilidade e simulação de Monte Carlo.

## Dois Cuidados nos Dados

### 1. A leitura ingênua do CSV apaga a América do Norte

A coluna `datacenter_region` usa as siglas `EU`, `APAC`, `ME` e **`NA`** (North America). Como `"NA"` faz parte da lista padrão de valores nulos do pandas, uma leitura convencional converte **3.990 registros — 39,9% da base — em `NaN`**.

```python
pd.read_csv(arquivo)                                          # 3.990 nulos, 3 regiões
pd.read_csv(arquivo, keep_default_na=False, na_values=[''])   # 0 nulos, 4 regiões
```

O efeito prático seria grave: quase 40% da base seria descartada como "dado faltante", e a América do Norte simplesmente desapareceria da análise sem que ninguém percebesse.

### 2. Verificação de vazamento entre colunas

A coluna `energy_efficiency_class` (Alta/Média/Baixa) é claramente derivada da telemetria — por isso foi checada quanto à correlação com o `pue` antes de qualquer modelagem. A correlação é forte, mas **não é uma função determinística** (faixas de PUE se sobrepõem entre as classes), diferente de um vazamento perfeito. Ainda assim, a coluna foi **excluída do cálculo do índice de risco** por precaução metodológica — usar um rótulo de eficiência para prever risco (que também depende de eficiência) seria parcialmente circular.

## Tratamento dos Dados

- Correção da leitura do CSV (`keep_default_na=False`) para preservar a região `NA`;
- Verificação de tipos, duplicatas e valores fisicamente implausíveis (`pue < 1,0`, percentuais fora de [0, 100], etc.) — nenhuma inconsistência encontrada após a correção de leitura;
- Resultado: **10.000 → 10.000 linhas**, 0 valores ausentes, 0 duplicatas.

## Variáveis Utilizadas

**Índice de risco (5 componentes, normalizados 0–1, pesos explícitos)**
`pue` (25%), `carbon_emission_kg` (20%), desvio de `inlet_temperature_c` em relação à faixa recomendada ASHRAE 18–27 °C (20%), `cooling_efficiency` invertida (20%), contagem de outliers via IQR em potência/custo/carbono (15%).

**Modelo de custo (features)**
`workload_intensity`, `cpu_utilization`, `memory_utilization`, `network_throughput_gbps`, `inlet_temperature_c`, `cooling_efficiency`, `power_consumption_kw`, `pue`, `server_type`, `cooling_type`, `datacenter_region`, `time_of_day` → alvo: `compute_cost_usd`.

**Exclusão deliberada**
- `energy_efficiency_class` — não entra no índice de risco nem no modelo de custo, por potencial circularidade com o próprio PUE (ver acima).

## Modelos

**Risco** — índice composto (engenharia de atributos + soma ponderada), não um classificador treinado, já que a base não traz um rótulo histórico de falha. A classificação em Baixo/Médio/Alto usa os tercis (percentis 33/66) da própria distribuição do score.

**Custo (regressão)** — Random Forest Regressor (300 árvores, profundidade máxima 10) sobre as variáveis de telemetria e as três decisões de projeto, avaliado em divisão treino/teste 80/20.

## Resultados

Conjunto de teste: 2.000 registros (20%, `random_state=42`).

### Modelo de custo

| Métrica | Valor |
|---|---|
| R² | 0,9893 |
| MAE | US$ 0,41 |
| MAE do baseline (prever a média) | US$ 3,84 |
| Redução do erro | 89,4% |

### Importância das variáveis no custo

| Variável | Importância |
|---|---|
| `datacenter_region` | 0,530 |
| `power_consumption_kw` | 0,465 |
| demais 10 variáveis (somadas) | 0,005 |

Duas variáveis — região e potência consumida — respondem por praticamente toda a capacidade preditiva do modelo. As variáveis de utilização (CPU, memória, rede) têm efeito residual sobre o custo *depois* de controlar por potência e região.

### Índice de risco

| Estatística | Valor |
|---|---|
| Média | 18,2 / 100 |
| Desvio-padrão | 7,0 |
| Mínimo | 0,9 |
| Máximo | 54,2 |

| Classe (tercis) | Registros |
|---|---|
| Baixo | 3.300 |
| Médio | 3.300 |
| Alto | 3.400 |

## Análise dos Três Eixos

### Risco por decisão de projeto (% de risco Alto)

| Região | % Alto | | Resfriamento | % Alto |
|---|---|---|---|---|
| APAC | 48,4% | | Air | 43,1% |
| ME | 39,6% | | Hybrid | 38,6% |
| NA | 32,7% | | Liquid | 16,1% |
| EU | 24,6% | | | |

### Custo médio por região

| Região | Custo médio | PUE médio | Carbono médio (kg) |
|---|---|---|---|
| ME | US$ 4,13 | 1,489 | 41,2 |
| APAC | US$ 6,54 | 1,481 | 49,1 |
| NA | US$ 9,89 | 1,478 | 33,0 |
| EU | US$ 14,70 | 1,481 | 20,4 |

A distribuição de tipos de servidor é praticamente uniforme entre as regiões (~40–43% `Compute` em todas), então a diferença de custo não é efeito de mix de equipamentos: isolando apenas os servidores `Compute`, a razão de custo entre EU e ME permanece quase idêntica (US$ 14,55 vs. US$ 4,18) — **o efeito da região é real**.

Nota-se um padrão de troca (*trade-off*) entre risco e custo/carbono: as regiões mais baratas (ME, APAC) apresentam mais risco Alto; a região mais cara (EU) tem o menor risco e o menor carbono — mas não o menor PUE, que é bastante estável entre regiões (~1,48). O driver de carbono e custo por região não parece vir da eficiência energética em si, e sim de outros fatores de contexto capturados implicitamente pela variável `datacenter_region`.

### Risco por servidor × resfriamento (score médio 0–100)

| Servidor | Air | Hybrid | Liquid |
|---|---|---|---|
| Compute | 20,0 | 19,3 | 14,7 |
| Edge | 18,6 | 17,9 | 13,4 |
| GPU | 22,4 | 21,4 | 16,6 |
| Storage | 18,7 | 18,4 | 13,9 |

O resfriamento líquido reduz o risco em todas as categorias de servidor, de forma consistente — é a alavanca mais robusta contra risco no conjunto de decisões controláveis.

### Tempo — modelo heurístico

Como a base não contém prazo de implantação, o tempo é estimado por uma fórmula com parâmetros explícitos:

```
T (semanas) = (T_base_servidor + T_base_resfriamento) × F_região × (1 + 0,5 × risco_norm) × (1 + 0,3 × custo_norm)
```

| Parâmetro | Valores assumidos |
|---|---|
| T_base_servidor (semanas) | Edge 3 · Compute 5 · Storage 6 · GPU 9 |
| T_base_resfriamento (semanas) | Air 1 · Hybrid 3 · Liquid 5 |
| F_região (fator logístico) | NA 1,00 · EU 1,15 · APAC 1,30 · ME 1,40 |

Estes são **parâmetros assumidos, não estimados a partir dos dados** — precisam ser calibrados com histórico real de projetos da organização antes de uso em decisões definitivas.

## Entregáveis de Decisão

### 1. Ranking de configurações candidatas

As **48 combinações** possíveis (4 regiões × 4 tipos de servidor × 3 tipos de resfriamento) são avaliadas simultaneamente nos três eixos — todas as combinações estão presentes na base, com no mínimo 15 amostras por cenário.

Com pesos de **40% risco / 35% custo / 25% tempo**:

```text
CONFIGURAÇÃO RECOMENDADA
  Tipo de servidor  : Edge
  Resfriamento      : Liquid
  Região            : ME (Oriente Médio)
  Score de decisão  : 82,5 / 100
  Risco médio       : 14,8 / 100
  Custo médio       : US$ 3,42
  Tempo estimado    : 12,5 semanas (~2,9 meses)
```

**Top 5 configurações**

| # | Servidor | Resfriamento | Região | Score | Risco | Custo (US$) | Tempo (sem.) |
|---|---|---|---|---|---|---|---|
| 1 | Edge | Liquid | ME | 82,5 | 14,8 | 3,42 | 12,5 |
| 2 | Edge | Liquid | NA | 81,8 | 13,3 | 7,63 | 9,1 |
| 3 | Storage | Liquid | ME | 80,5 | 13,8 | 3,36 | 16,6 |
| 4 | Edge | Liquid | APAC | 76,4 | 15,6 | 5,38 | 12,3 |
| 5 | Edge | Liquid | EU | 76,3 | 11,5 | 11,76 | 10,6 |

Uma análise de sensibilidade varia os pesos em torno do ponto default (simulação de Dirichlet, 300 sorteios) e mostra que a **configuração #1 permanece no Top 5 em 95% das simulações de pesos** — a decisão é robusta a variações razoáveis do apetite de risco do decisor.

### 2. Robustez sob incerteza amostral (Monte Carlo)

Além da incerteza de pesos, existe incerteza estatística nos valores médios de cada cenário (amostras finitas de 15 a 769 registros por combinação). Uma simulação de *bootstrap* (300 repetições, reamostrando 30 registros por cenário a cada rodada) estima a probabilidade de cada configuração permanecer entre as 5 melhores:

| Configuração | Estabilidade a pesos | Robustez (Monte Carlo) |
|---|---|---|
| Edge / Liquid / ME | 95,0% | 98,7% |
| Edge / Liquid / NA | 89,0% | 95,3% |
| Storage / Liquid / ME | 85,3% | 86,7% |
| Edge / Liquid / APAC | 56,3% | 51,7% |
| Edge / Liquid / EU | 53,7% | 45,0% |

A configuração recomendada tem alta robustez em ambas as fontes de incerteza — pesos do decisor e variabilidade amostral.

## Principais Conclusões

1. **As decisões de projeto, combinadas a um índice de risco simples, bastam para orientar a escolha da configuração antes de construir.** Não há necessidade de um classificador treinado com rótulo histórico de falha — um índice interpretável, com pesos explícitos, já permite discriminar risco entre 0,9 e 54,2 pontos e separar claramente as configurações por região e resfriamento.

2. **Custo é dominado por apenas duas variáveis: região e potência consumida.** Juntas respondem por 99,5% da importância do modelo (R² = 0,989). Isso sugere que otimizar `power_consumption_kw` (por meio de melhor eficiência energética) e escolher bem a região são as duas alavancas de maior impacto no custo operacional.

3. **Risco e custo se opõem geograficamente.** ME e APAC são as regiões mais baratas e também as de maior risco; EU é a mais cara e a de menor risco. Essa oposição persiste mesmo controlando o tipo de servidor, ou seja, não é efeito de mix de equipamentos.

4. **A refrigeração líquida é a alavanca de risco mais consistente entre todas as categorias de servidor**, reduzindo o score de risco médio em cerca de 25–30% em relação ao ar, sem custo adicional evidente na base (o custo é dominado por região/potência, não pelo tipo de resfriamento).

5. **O tempo de implantação precisa ser tratado como suposição explícita, não como resultado estatístico.** Como a base não contém essa variável, o modelo heurístico usado aqui deve ser visto como um ponto de partida para discussão com a área de operações da empresa, não como uma previsão calibrada.

6. **A recomendação final é robusta**, tanto à escolha de pesos do decisor (95% de estabilidade) quanto à incerteza amostral (98,7% via Monte Carlo) — reforçando que `Edge / Liquid / ME` é uma escolha defensável dentro das premissas do modelo.

## Estrutura do Repositório

```text
mvp-unidade-controle-dados/
├── MVP_Risco_Custo_Tempo_Unidade_Dados.ipynb
├── green_ai_datacenter.csv
├── resultado_mcda_unidade_dados.csv
├── relatorio_mvp_unidade_dados.txt
└── README.md
```

## Como Executar

O notebook foi desenvolvido para o **Google Colab**.

### Google Colab (recomendado)

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Vá em **File → Open notebook → GitHub**, cole a URL deste repositório e selecione o arquivo `.ipynb`
   — ou troque, na barra de endereço, `github.com` por `colab.research.google.com/github` na URL do notebook neste repositório.
3. Execute as células em ordem (**Runtime → Run all**).
4. Quando solicitado, faça upload do arquivo `green_ai_datacenter.csv` (disponível neste repositório).
5. Ao final, o notebook exporta `resultado_mcda_unidade_dados.csv` e `relatorio_mvp_unidade_dados.txt`.

### Localmente

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn nbformat
jupyter notebook MVP_Risco_Custo_Tempo_Unidade_Dados.ipynb
```

> As simulações de sensibilidade e Monte Carlo (300 repetições cada) são as células mais demoradas — poucos segundos a poucos minutos, dependendo da máquina.

## Limitações

- A base é **sintética**, portanto os valores absolutos (US$, kg CO₂) não devem ser extrapolados diretamente para um projeto real sem validação com dados operacionais da organização.
- O índice de risco é um **score de engenharia de atributos com pesos definidos por julgamento**, não um modelo treinado sobre rótulos históricos de falha — a base não contém esse rótulo.
- `compute_cost_usd` é um custo computacional por registro de telemetria, **não o custo total de implantação** (não há CAPEX — terreno, obra, licenças — na base).
- A base **não contém prazo de implantação**; o eixo "tempo" é uma estimativa heurística com parâmetros assumidos, que precisa ser calibrada com histórico real de projetos.
- As regiões são agregados continentais (`NA`, `EU`, `APAC`, `ME`); escolher um país ou cidade específica exigiria dados locais de energia, conectividade e regulação.
- O MCDA usa pesos lineares e assume independência entre os três critérios.

## Próximos Passos

- incorporar **CAPEX e cronograma de obra** para completar o eixo de tempo com um prazo de implantação real, não apenas heurístico;
- validar/calibrar o índice de risco com dados de falha reais, se disponíveis, migrando de um score heurístico para um modelo preditivo supervisionado;
- testar Gradient Boosting / XGBoost e comparar com o Random Forest usado no modelo de custo;
- aplicar SHAP para explicar a recomendação configuração a configuração;
- estender a simulação para uma **carteira de múltiplas unidades**, otimizando a distribuição geográfica;
- desagregar a variável `datacenter_region` (hoje dominante no custo) em fatores mais interpretáveis, como preço local de energia e custo de mão de obra.

## Fonte dos Dados

**Green AI Data Center Telemetry** — dataset sintético do Kaggle para experimentos de eficiência energética em data centers.
10.000 registros, 15 colunas, geradas por simulação (workload, utilização de CPU/memória, throughput de rede, temperatura de entrada, eficiência de resfriamento, potência, emissão de carbono, custo computacional, PUE e classe de eficiência energética).

> Adicione aqui o link exato da página do Kaggle de onde a base foi baixada.

---

**Aluna:** Ana Luisa Breide Pessôa Guerra
**Matrícula:** 231013289
**Disciplina:** Sistemas de Suporte à Decisão
**Universidade de Brasília (UnB)**
