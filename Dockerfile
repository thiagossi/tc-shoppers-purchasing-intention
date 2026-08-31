FROM python:3.11-slim AS build

WORKDIR /build

RUN pip install --no-cache-dir poetry poetry-plugin-export

COPY pyproject.toml poetry.lock ./
RUN poetry export --without-hashes -f requirements.txt -o requirements.txt

RUN pip install --no-cache-dir --ignore-installed --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN addgroup --system mlgroup && adduser --system --ingroup mlgroup mluser

COPY --from=build /install /usr/local

COPY src/ ./src
COPY configs/ ./configs
COPY data/raw/online_shoppers_intention.csv ./data/raw/online_shoppers_intention.csv

ENV PYTHONPATH=/app/src

RUN chown -R mluser:mlgroup /app
USER mluser

ENV GIT_PYTHON_REFRESH=quiet

CMD ["sh", "-c", "python -m purchase_intent.pipeline.preprocess && python -m purchase_intent.pipeline.train"]