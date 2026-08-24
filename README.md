# Purchase Intent — Online Shoppers Purchasing Intention

Tech Challenge Fase 2 (FIAP) — sistema preditivo de propensão de compra
a partir do comportamento de navegação de usuários em e-commerce.

Dataset: [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI/Kaggle). Já incluído no repositório em `data/raw/`, não precisa baixar nada separado.

## Status

Etapas 1-4 concluídas: estrutura Clean Code, Poetry, DVC, Docker, treino do
modelo com MLflow Tracking e Model Registry.

## Pré-requisitos

Antes de começar, tenha instalado:

- **[Python 3.11+](https://www.python.org/downloads/)** — verifique com `python --version`
- **[Poetry](https://python-poetry.org/docs/#installation)** — gerenciador de dependências do projeto. Instale com:

  Windows (PowerShell):
  ```powershell
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
  ```

  Linux / macOS / Git Bash:
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```

  Verifique com `poetry --version`. Se o comando não for reconhecido, adicione a pasta do Poetry ao PATH do sistema (o instalador informa o caminho exato ao final).
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** — necessário **apenas** se for testar a seção "Executando com Docker" abaixo. Precisa estar aberto e rodando (ícone estável na bandeja do sistema) antes de usar `docker build`/`docker run`.

## Passo a passo — rodando o projeto do zero

**1. Clone o repositório e entre na pasta:**
```bash
git clone https://github.com/thiagossi/tc-shoppers-purchasing-intention.git
cd tc-shoppers-purchasing-intention
```

**2. Instale as dependências do projeto** (cria um ambiente virtual isolado em `.venv/`, sem afetar seu Python global):
```bash
poetry install
```

**3. Configure as variáveis de ambiente:**

Linux / macOS / Git Bash:
```bash
cp .env.example .env
```

Windows (PowerShell):
```powershell
Copy-Item .env.example .env
```

(Não precisa editar nada por enquanto — não há segredos reais no projeto ainda.)

**4. Rode os testes automatizados** (confirma que tudo foi instalado corretamente):
```bash
poetry run pytest --cov=purchase_intent
```

**5. Rode o pipeline completo (pré-processamento + treino):**
```bash
poetry run dvc repro
```
Isso executa dois estágios: `preprocess` (gera `data/processed/features.csv` e
`models/preprocessor.joblib`) e `train` (treina um Random Forest, gera
`models/model.joblib` e `metrics.json`, e registra o experimento no MLflow).

## Estrutura do projeto

```
configs/                 # arquivos de configuração (ex: params.yaml)
data/
  raw/                    # dados brutos (versionados via Git + DVC)
  processed/              # dados tratados, gerados pelo pipeline (não versionado)
models/                   # modelos/artefatos treinados, gerados pelo pipeline
src/purchase_intent/
  data/                   # ingestão de dados
  features/               # engenharia de features
  models/                 # treino/predição
  evaluation/             # avaliação de modelos
  pipeline/               # scripts que orquestram os estágios do pipeline
  utils/                  # utilitários gerais
tests/                    # testes automatizados
dvc.yaml                  # pipeline reprodutível (DVC)
Dockerfile                # imagem containerizada do pipeline
metrics.json              # métricas da última execução de treino (versionado)
mlflow.db                 # tracking store do MLflow (gerado, não versionado)
```

## Sobre o pipeline de dados (DVC)

O `dvc.yaml` declara dois estágios:

- **`preprocess`** — lê `data/raw/online_shoppers_intention.csv`, aplica escala
  nas colunas numéricas e one-hot encoding nas categóricas, salva
  `data/processed/features.csv` e `models/preprocessor.joblib`.
- **`train`** — lê o dataset processado, treina um `RandomForestClassifier`
  (hiperparâmetros em `configs/params.yaml`), avalia no conjunto de teste e
  salva `models/model.joblib` e `metrics.json`.

Rodar `dvc repro` novamente só reexecuta um estágio se seus dados, código ou
parâmetros tiverem mudado (comportamento padrão do DVC, baseado em hash).

**Limitação conhecida:** o dataset não possui registros de Janeiro/Abril. Para
entradas futuras nesses meses, o `OneHotEncoder` (configurado com
`handle_unknown="ignore"`) zera as colunas de mês em vez de falhar — degradação
graciosa, não um erro do pipeline.

## Treinamento, MLflow Tracking e Model Registry

O modelo é um `RandomForestClassifier` (scikit-learn) com `class_weight="balanced"`,
já que o dataset é desbalanceado (~85% não compra / ~15% compra). Por isso a
avaliação usa `accuracy`, `precision`, `recall`, `f1` e `roc_auc` — não só
acurácia, que seria enganosa nesse cenário.

Cada execução de treino é registrada como uma *run* do MLflow (parâmetros,
métricas e o modelo em si). Para abrir a interface visual e comparar execuções:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Depois acesse `http://localhost:5000` no navegador.

**Model Registry:** a cada treino, o modelo é registrado sob o nome
`shoppers-purchase-intent`, ganhando uma versão nova automaticamente. Se a
métrica `f1` dessa versão for igual ou melhor que a versão atualmente marcada
com o alias `champion`, ela é promovida a `champion` — o critério objetivo de
promoção pedido pelo desafio, sem aprovação manual.

## Executando com Docker (opcional)

Requer Docker Desktop instalado e aberto (veja Pré-requisitos). A partir da raiz
do projeto:

```bash
docker build -t purchase-intent:0.2 .
docker run purchase-intent:0.2
```

Isso constrói uma imagem que já contém o dataset e o código, e executa **os
dois estágios** (pré-processamento e treino) de ponta a ponta dentro do
container, sem depender de nada externo. Uma saída sem erros (código de
saída `0`) confirma que o pipeline rodou com sucesso, incluindo o registro
do modelo no MLflow Model Registry.

*Nota: o `mlflow.db` gerado dentro do container é efêmero — some quando o
container termina, a menos que você monte um volume. O histórico "de
verdade" de experimentos é o `mlflow.db` gerado localmente via `dvc repro`
(passo 5 acima).*
