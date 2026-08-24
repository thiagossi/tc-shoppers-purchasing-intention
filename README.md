# Purchase Intent — Online Shoppers Purchasing Intention

Prevê se um usuário de e-commerce vai fechar uma compra, a partir do
comportamento de navegação dele — páginas visitadas, tempo em cada uma,
se é visitante novo ou recorrente, entre outros sinais.

Dataset: [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
(UCI/Kaggle), já incluído em `data/raw/` — não precisa baixar nada à parte.

## Rodando do zero

Precisa de Python 3.11+ e [Poetry](https://python-poetry.org/docs/#installation)
instalados. Docker é opcional, só entra se quiser testar a versão
containerizada (precisa do Docker Desktop aberto).

Instalando o Poetry, se ainda não tiver:

```powershell
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```
```bash
# Linux / macOS / Git Bash
curl -sSL https://install.python-poetry.org | python3 -
```

Se `poetry --version` não for reconhecido depois, falta adicionar a pasta
dele ao PATH (o instalador mostra o caminho certinho no final).

Daí:

```bash
git clone https://github.com/thiagossi/tc-shoppers-purchasing-intention.git
cd tc-shoppers-purchasing-intention
poetry install
```

Copia o `.env.example` pra `.env` (`cp .env.example .env` no Bash,
`Copy-Item .env.example .env` no PowerShell — nada de segredo real pra
configurar ainda, é só a variável do MLflow).

Pra conferir que a instalação foi limpa:

```bash
poetry run pytest --cov=purchase_intent
```

E pra rodar o pipeline de verdade:

```bash
poetry run dvc repro
```

Isso processa os dados brutos (`data/processed/features.csv`,
`models/preprocessor.joblib`) e depois treina o modelo (Random Forest,
`models/model.joblib`, `metrics.json`, registrado no MLflow).

## Estrutura

```
configs/                 # params.yaml — hiperparâmetros de treino
data/
  raw/                    # dados brutos (Git + DVC)
  processed/              # gerado pelo pipeline, não versionado
models/                   # artefatos treinados, gerados pelo pipeline
src/purchase_intent/
  data/                   # ingestão
  features/               # pré-processamento
  models/                 # treino/predição
  evaluation/             # avaliação
  pipeline/               # orquestra os estágios (loader → preprocess → train)
  utils/
tests/
dvc.yaml
Dockerfile
metrics.json              # métricas do último treino, versionado
mlflow.db                 # tracking store do MLflow, gerado localmente
```

## DVC

`dvc.yaml` tem dois estágios — `preprocess` e `train` — e cada um só roda
de novo se o dado, o código ou os parâmetros relevantes mudarem (hash do
DVC, nada além disso).

`preprocess` lê o CSV bruto, escala as colunas numéricas e faz one-hot nas
categóricas. `train` pega esse resultado, treina um `RandomForestClassifier`
(hiperparâmetros em `configs/params.yaml`), avalia contra um conjunto de
teste separado e salva o modelo e as métricas.

Uma peculiaridade do dataset: não tem registro de Janeiro nem Abril. Se
aparecer alguém navegando nesses meses no futuro, o `OneHotEncoder`
(`handle_unknown="ignore"`) zera as colunas de mês em vez de quebrar —
perde um pouco de sinal ali, mas o pipeline não cai.

## Treino e MLflow

Random Forest com `class_weight="balanced"`, porque só ~15% das sessões
terminam em compra — sem isso o modelo aprenderia a chutar "não compra"
sempre e ainda pareceria bom olhando só a acurácia. Por isso a avaliação
também usa precision, recall, f1 e roc_auc.

Cada treino vira uma run no MLflow. Pra abrir a interface:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`http://localhost:5000`.

O modelo é registrado como `shoppers-purchase-intent` a cada execução, e só
ganha o alias `champion` se o f1 dessa versão empatar ou superar o da
versão campeã atual — a promoção segue essa regra fixa, sem depender de
alguém validar manualmente.

## Docker

```bash
docker build -t purchase-intent:0.2 .
docker run purchase-intent:0.2
```

Roda o pipeline inteiro (preprocess + train) dentro do container — o
dataset já vem embutido na imagem, não precisa montar volume nem baixar
nada. Terminar sem erro (`exit 0`) confirma que funcionou, incluindo o
registro do modelo no MLflow.

Um detalhe: o `mlflow.db` que nasce dentro do container some junto com ele
quando termina. O histórico "de verdade" é o local, gerado pelo próprio
`dvc repro`.
