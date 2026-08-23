# Purchase Intent — Online Shoppers Purchasing Intention

Tech Challenge Fase 2 (FIAP) — sistema preditivo de propensão de compra
a partir do comportamento de navegação de usuários em e-commerce.

Dataset: [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI/Kaggle). Já incluído no repositório em `data/raw/`, não precisa baixar nada separado.

## Status

Etapa 3 concluída (Containerização e Versionamento). Próxima etapa: modelagem e MLflow.

## Pré-requisitos

Antes de começar, tenha instalado:

- **[Python 3.11+](https://www.python.org/downloads/)** — verifique com `python --version`
- **[Poetry](https://python-poetry.org/docs/#installation)** — gerenciador de dependências do projeto. Instale com:
  ```bash
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
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
```bash
cp .env.example .env
```
(Não precisa editar nada por enquanto — não há segredos reais no projeto ainda.)

**4. Rode os testes automatizados** (confirma que tudo foi instalado corretamente):
```bash
poetry run pytest --cov=purchase_intent
```

**5. Rode o pipeline de dados:**
```bash
poetry run dvc repro
```
Isso gera `data/processed/features.csv` (dados pré-processados) e `models/preprocessor.joblib` (transformador treinado).

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
```

## Sobre o pipeline de dados (DVC)

O `dvc.yaml` declara o estágio `preprocess`, que lê `data/raw/online_shoppers_intention.csv`,
aplica escala nas colunas numéricas e one-hot encoding nas categóricas, e salva os
artefatos em `data/processed/` e `models/`. Rodar `dvc repro` novamente só reexecuta
o estágio se o dado ou o código de pré-processamento tiverem mudado (comportamento
padrão do DVC, baseado em hash).

**Limitação conhecida:** o dataset não possui registros de Janeiro/Abril. Para
entradas futuras nesses meses, o `OneHotEncoder` (configurado com
`handle_unknown="ignore"`) zera as colunas de mês em vez de falhar — degradação
graciosa, não um erro do pipeline.

## Executando com Docker (opcional)

Requer Docker Desktop instalado e aberto (veja Pré-requisitos). A partir da raiz
do projeto:

```bash
docker build -t purchase-intent:0.1 .
docker run purchase-intent:0.1
```

Isso constrói uma imagem que já contém o dataset e o código, e executa o
estágio de pré-processamento de ponta a ponta dentro do container, sem
depender de nada externo. Uma saída sem erros (código de saída `0`) confirma
que o pipeline rodou com sucesso.
