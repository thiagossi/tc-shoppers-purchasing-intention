# Purchase Intent — Online Shoppers Purchasing Intention

Tech Challenge Fase 2 (FIAP) — sistema preditivo de propensão de compra
a partir do comportamento de navegação de usuários em e-commerce.

Dataset: [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI/Kaggle).

## Status

Em desenvolvimento — Etapa 1 (Clean Code e Estrutura).

## Estrutura do projeto

```
configs/                 # arquivos de configuração (ex: params.yaml)
data/
  raw/                    # dados brutos, não versionados no Git (via DVC futuramente)
  processed/              # dados tratados
models/                   # modelos treinados
src/purchase_intent/
  data/                   # ingestão de dados
  features/               # engenharia de features
  models/                 # treino/predição
  evaluation/             # avaliação de modelos
  utils/                  # utilitários gerais
tests/                    # testes automatizados
```

## Instalação

Requer [Poetry](https://python-poetry.org/) e Python >= 3.11.

```bash
poetry install
```

## Rodando os testes

```bash
poetry run pytest
```
