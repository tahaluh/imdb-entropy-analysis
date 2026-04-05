# Projeto Final - PDI

Objetivo: investigar se existe relacao entre compressao/entropia de legendas e a nota no IMDb.

## 1) Ambiente (env)

No Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Pipeline de análise

O projeto segue 8 etapas principais:

1. **01_collect_imdb.py** – Coleta dados do IMDb (filmes, ratings, votos)
2. **02_plot_movies.py** – Visualizações exploratórias dos filmes
3. **03_collect_subtitles.py** – Placeholder para coleta de legendas (Kaggle dataset usado)
4. **04_merge_subtitles_with_movies.py** – Merge IMDb + legendas brutas
5. **05_plot_movies_with_subtitles.py** – Visualizações dos dados combinados
6. **06_normalize_subtitles.py** – Limpeza de texto e normalização
7. **07_filter_subtitles.py** – Filtro por tamanho de legenda (500-6000 linhas)
8. **08_extract_information_features.py** – **Extração de entropia e compressão** ← núcleo do trabalho

## 3) Estrutura atual

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

| Etapa | Output | Registros | Descrição |
|-------|--------|-----------|----------|
| 01 | `movies.csv` | 10.060 | Filmes IMDb brutos (≥10k votos, ano≥1990) |
| 04 | `movies_with_subtitle_stats.csv` | 1.368 | Merge IMDb + legendas (merge inner) |
| 06 | `movies_with_subtitles.csv` | 1.368 | Legendas normalizadas + features de texto |
| 07 | `movies_filtered.csv` | **1.304** | **Dataset final com 500-6000 linhas de legenda** |
| 08 | `movies_information_features.csv` | **1.304** | **Features de entropia + compressão** |

### Dataset de legendas
- Fonte: [Kaggle - Movie Subtitle Dataset](https://www.kaggle.com/datasets/adiamaan/movie-subtitle-dataset)
- Arquivo: `movies_subtitles.csv` (4.8GB)
- Colunas: `start_time`, `end_time`, `text`, `imdb_id`

## 5) Features calculadas no arquivo 08

O script `08_extract_information_features.py` calcula as métricas principais do trabalho:

**Entropia (bits/símbolo):**
- `char_entropy` – Entropia de Shannon por caractere individual
- `bigram_entropy` – Entropia de pares de caracteres
- `trigram_entropy` – Entropia de trios de caracteres
- `word_entropy` – Entropia baseada em palavras/tokens

**Compressão (razões e economia):**
- `gzip_ratio`, `bz2_ratio`, `lzma_ratio` – Tamanho comprimido / original
- `gzip_saving`, `bz2_saving`, `lzma_saving` – Economia percentual (1 - ratio)
- `gzip_bits_per_byte`, `bz2_bits_per_byte`, `lzma_bits_per_byte` – Bits médios por byte após compressão

## 6) Próximos passos sugeridos

Análise de correlação entre as features de entropia/compressão e os ratings IMDb:
- Pearson correlation
- Spearman correlation
- Regressão linear/não-linear
- Visualizações (scatter plots, heat maps)
- Relatório final
# imdb-entropy-analysis
