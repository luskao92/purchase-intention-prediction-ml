# 🛒 Purchase Intention Prediction

### Predição de Conversão em Sessões de E-commerce com Machine Learning

> **MVP — Machine Learning & Analytics · PUC-Rio**
> Entrega: 05 de Julho de 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1InlAxUBgMNb5s6aDYLU3x2fHp4N4Gx0n?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat&logo=pandas&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Pipelines-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Conclu%C3%ADdo-4CAF50?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 🎯 Problema de Negócio

Em e-commerce, a maior parte do investimento de **growth** vai para aquisição de tráfego — mas só ~15% das sessões convertem. Os outros 85% do tráfego pago ou orgânico não geram receita, e a operação só descobre isso depois que a sessão acabou.

Este projeto analisa **12.330 sessões de navegação** de um e-commerce para responder:

> *Dado o comportamento de navegação de um usuário durante a sessão, é possível prever se ele vai converter — e quais sinais comportamentais mais importam?*

**Tipo de problema:** classificação binária supervisionada · variável-alvo `Revenue` (desbalanceada: 85% / 15%)

**Aplicação direta:** score de propensão para intervenção em tempo real (oferta, chat proativo), priorização de audiências de remarketing e redução de CAC.

---

## 💡 Principais Resultados

| # | Resultado | Impacto de Negócio |
| --- | --- | --- |
| 1 | Gradient Boosting otimizado: **F1 0,675 · AUC 0,938** no teste | Contra F1 zero do baseline e ~0,49 da regressão logística |
| 2 | Precisão de 71,9% na classe compradora | A cada 10 sessões priorizadas, 7 são compradores reais |
| 3 | `PageValues` domina com ~71% da importância | Engajamento com páginas de valor é o sinal nº 1 de conversão |
| 4 | Novos visitantes convertem 1,8× mais que recorrentes (24,9% vs 14,1%) | Achado contraintuitivo — recorrente pesquisa, novo compra |
| 5 | Novembro: maior taxa (25,5%) e 2º maior volume de sessões | Assinatura de pico promocional (Black Friday) confirmada no modelo |

---

## 🔬 Hipóteses Testadas

| # | Hipótese | Verificação | Veredicto |
| --- | --- | --- | --- |
| H1 | Engajamento com páginas de produto prevê conversão | EDA (boxplots) + importância de features | ✅ Confirmada — `PageValues` domina o modelo |
| H2 | Tipo de visitante altera a taxa de conversão | Taxa por segmento | ✅ Confirmada — direção contraintuitiva (novos > recorrentes) |
| H3 | Sazonalidade impacta a propensão de compra | Taxa por mês + `Month_Nov` no top 3 de features | ✅ Confirmada — efeito Black Friday |

---

## 📦 Dataset

**Online Shoppers Purchasing Intention** · UCI Machine Learning Repository · [Sakar & Kastro, 2018](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

| Dimensão | Valor |
| --- | --- |
| Instâncias | 12.330 sessões (12.205 após remoção de duplicatas) |
| Atributos | 17 features + variável-alvo |
| Natureza das features | Métricas de web analytics (bounce rate, exit rate, page value) |
| Variável-alvo | `Revenue` — desbalanceada (84,5% / 15,5%) |

```python
URL = (
    "https://raw.githubusercontent.com/luskao92/purchase-intention-prediction-ml/refs/heads/main/data/online_shoppers_intention.csv"
)
df = pd.read_csv(URL)
```

---

## ⚙️ Abordagem de Modelagem

| Etapa | Decisão | Justificativa |
| --- | --- | --- |
| Pré-processamento | `Pipeline` + `ColumnTransformer` | Blindagem contra data leakage — scaler e encoder aprendem só no treino |
| Divisão | 80/20 estratificada · seed 42 | Preserva a proporção da classe minoritária |
| Baseline | `DummyClassifier` (majoritária) | Prova que acurácia engana: 84% "acertando" sem achar um comprador |
| Candidatos | Regressão Logística · Random Forest · Gradient Boosting | Complexidade crescente, comparados por CV estratificada (5 folds) |
| Otimização | Random Search (20 iterações) no Gradient Boosting | 432 combinações no espaço — busca aleatória cobre com fração do custo |
| Métricas | F1 e AUC (decisão) · precisão e recall (leitura de negócio) | Adequadas ao desbalanceamento |

---

## 🗂️ Estrutura do Notebook (10 Seções)

```
Seção 1  — Definição do Problema
           1.1 Contexto · 1.2 Tipo de tarefa · 1.3 Por que ML
           1.4 Premissas e hipóteses · 1.5 Restrições

Seção 2  — Apresentação dos Dados
           Fonte · Estrutura · Limitações · Critérios de escolha

Seção 3  — Configuração do Ambiente e Carga dos Dados

Seção 4  — Análise Exploratória          ← orientada às hipóteses H1–H3
           4.1 Qualidade (duplicatas, tipos) · 4.2 Estatísticas descritivas
           4.3 Distribuição do alvo · 4.4–4.5 Testes visuais das hipóteses

Seção 5  — Preparação dos Dados          ← Pipeline + ColumnTransformer
Seção 6  — Modelagem e Treinamento       ← baseline + 3 candidatos, CV 5 folds
Seção 7  — Otimização de Hiperparâmetros ← Random Search
Seção 8  — Avaliação dos Resultados      ← teste intocado · matriz de confusão
                                            curva ROC · overfitting · importâncias
Seção 9  — Conclusão
Seção 10 — Checklist do MVP
```

---

## 🧹 Decisões de Qualidade dos Dados

| Problema identificado | Campo | Decisão |
| --- | --- | --- |
| 125 linhas duplicadas | — | Remoção — evita leakage de cópias entre treino e teste |
| Categóricas tipadas como inteiro | `OperatingSystems`, `Browser`, `Region`, `TrafficType` | Conversão + one-hot — códigos não têm ordem |
| Escalas muito distintas | Durações vs. taxas | `StandardScaler` nas numéricas |
| Assimetria e outliers extremos | Variáveis de duração | Preferência por modelos de árvore, robustos a outliers |

---

## 📁 Estrutura do Repositório

```
purchase-intention-prediction-ml/
│
├── README.md
├── data/
│   └── online_shoppers_intention.csv     ← dataset · CSV · 12.330 linhas
└── mvp_purchase_intention.ipynb      ← notebook completo (Colab-ready)
```

---

## 🛠️ Stack

`Python 3.10+` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Google Colab`

---

## 📋 Checklist MVP — PUC-Rio

- [x] Definição do problema com contexto de negócio e hipóteses
- [x] Dataset real, público, carregado via URL raw do GitHub
- [x] EDA orientada a hipóteses, com análise textual após cada visualização
- [x] Pré-processamento 100% dentro de Pipelines (anti-leakage)
- [x] Divisão estratificada + validação cruzada estratificada (5 folds)
- [x] Baseline + 3 modelos candidatos comparados de forma justa
- [x] Otimização de hiperparâmetros com Random Search justificado
- [x] Avaliação em teste intocado: F1, AUC, matriz de confusão, curva ROC
- [x] Discussão de overfitting/underfitting e limitações
- [x] Seeds fixadas — notebook executável do início ao fim
- [x] Checklist do edital respondido na Seção 10

---

## 👤 Autor

**Lucas Alves Medeiros** · [@luskao92](https://github.com/luskao92)
PUC-Rio

---

## 📄 Licença

[MIT](LICENSE)
