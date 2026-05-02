# Visual Geo-localization

## Pre-requisites

```shell
# Copy env file, update the variables.
cp .env.example .env
```

```shell
uvx marimo edit --sandbox notebooks/fig-visloc-val-flight-map.py
```

Best model inference:
```shell
# Run as a script
uv run notebooks/ft-dinov3-vitb-ssl4eo_ch-visloc.py

# Run as a notebook
uvx marimo edit --sandbox notebooks/ft-dinov3-vitb-ssl4eo_ch-visloc.py
```

Inspect the embeddings:
```shell
uvx marimo edit --sandbox notebooks/explore-embeddings.py
```

Run any zeroshot model:
```shell
# Run as a script
uv run notebooks/zeroshot-
```