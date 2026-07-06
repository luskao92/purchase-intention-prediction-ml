# Predição de Intenção de Compra em Sessões de E-commerce

MVP de Machine Learning & Analytics — Pós-graduação PUC-Rio

## O problema

Em e-commerce, a maior parte do investimento de growth vai para aquisição de tráfego — mas só uma 
fração pequena das sessões converte. Neste dataset, ~15%. A pergunta deste projeto é direta: **dado o 
comportamento de navegação de um usuário durante a sessão, dá para prever se ele vai converter?**

Um modelo assim tem aplicação imediata em CRO e mídia: intervenção em tempo real (oferta, chat 
proativo) só para sessões de alta propensão, priorização de audiências de remarketing e redução de CAC.

## Os dados

[Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) 
(UCI Machine Learning Repository) — 12.330 sessões, 17 atributos comportamentais e a variável-alvo 
`Revenue`. As features espelham métricas reais de web analytics (bounce rate, exit rate, page value), 
o que aproxima o exercício de um caso real de growth.

## Abordagem

- EDA orientada a hipóteses de negócio (engajamento, tipo de visitante, sazonalidade)
- Pré-processamento 100% dentro de `Pipeline` + `ColumnTransformer` (blindagem contra data leakage)
- Baseline ingênuo + Regressão Logística + Random Forest + Gradient Boosting, comparados por 
validação cruzada estratificada (5 folds)
- Otimização de hiperparâmetros via Random Search no melhor candidato
- Avaliação final em conjunto de teste intocado, com F1, AUC, matriz de confusão e discussão de 
overfitting

## Principais resultados

[PREENCHER AO FINAL — ex.: O Gradient Boosting otimizado atingiu F1 de X,XX e AUC de X,XX no conjunto 
de teste, contra F1 zero do baseline. PageValues dominou a importância de features, confirmando a 
hipótese central da EDA.]

## Como executar

O notebook foi desenhado para rodar de ponta a ponta sem configuração: o dataset é carregado 
diretamente pela raw URL deste repositório.

**Notebook no Colab:** [PREENCHER — link "Abrir no Colab" após o upload final]

## Estrutura do repositório

```
├── README.md
├── data/
│   └── online_shoppers_intention.csv
└── notebook/
    └── mvp_purchase_intention.ipynb
```
