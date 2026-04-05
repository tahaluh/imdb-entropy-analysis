# Projeto Final - PDI

Objetivo: investigar se existe relacao entre compressao/entropia de legendas e a nota no IMDb.

## 1) Ambiente (env)

No Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 2) Estrutura atual

- `01_collect_imdb.py`: coleta os datasets publicos do IMDb, faz limpeza e salva uma base inicial de filmes.
- `.env.example`: variaveis de configuracao do pipeline.
- `requirements.txt`: dependencias Python.
- `source/README.md`: etapas do pipeline por arquivo.

## 3) Explicacao da etapa 01

Arquivo: `01_collect_imdb.py`

Etapas executadas:

1. Le variaveis de ambiente (`DATA_DIR`, `MIN_VOTES`, `MIN_YEAR`, URLs do IMDb).
2. Cria pastas de trabalho:
   - `data/raw/imdb`
   - `data/processed`
3. Baixa (ou reaproveita cache em pickle) os arquivos:
   - `title.basics.tsv.gz`
   - `title.ratings.tsv.gz`
4. Substitui `\\N` por valor ausente (`NA`).
5. Faz merge por `tconst`.
6. Filtra para filmes (`titleType == movie`), nao adultos e com campos essenciais preenchidos.
7. Converte colunas numericas (`year`, `runtime`, `votes`, `rating`).
8. Aplica corte minimo por votos/ano (`MIN_VOTES`, `MIN_YEAR`).
9. Seleciona e renomeia colunas para um formato mais limpo.
10. Ordena por votos e nota, e salva em `data/processed/movies.csv`.

Como rodar:

```bash
python 01_collect_imdb.py
```

Saida esperada:

- Arquivo `data/processed/movies.csv`
- Impressao no terminal com caminho salvo e total de filmes.

## 4) Dados coletados (status atual)

- Total de filmes IMDb (apos filtros): **10060**
- IMDb IDs com legenda no dataset: **4666**
- Filmes combinados (IMDb + legendas): **1368**

Dataset de legendas:
- Fonte: [Kaggle - Movie Subtitle Dataset](https://www.kaggle.com/datasets/adiamaan/movie-subtitle-dataset)
- Arquivo: `movies_subtitles.csv` (start_time, end_time, text, imdb_id)
- Armazenamento: `data/raw/subtitles/movies_subtitles.csv`

## 5) Proximo passo sugerido

Extrair metricas de compressa/entropia/complexidade das legendas (juntando com o dataset combinado em `movies_with_subtitle_stats.csv`) e correlacionar com ratings IMDb para validar a hipotese do trabalho.
# imdb-entropy-analysis
