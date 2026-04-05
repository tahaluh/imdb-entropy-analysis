# Pipeline proposto por arquivos

Este diretorio organiza as etapas da pesquisa sobre a relacao entre caracteristicas de legendas e nota no IMDb.

## Arquivos e etapas

1. `01_collect_imdb.py` (ja implementado na raiz)
   - Coleta e prepara base de filmes + notas do IMDb.

2. `02_collect_subtitles.py` (proximo)
   - Recebe lista de filmes (`movies.csv`).
   - Busca e baixa legendas (SRT) de uma fonte definida no projeto.
   - Salva em `data/raw/subtitles/`.

3. `03_extract_text_features.py`
   - Le os SRTs.
   - Limpa timestamps e tags.
   - Calcula metadados de texto e complexidade.

4. `04_entropy_compression_features.py`
   - Calcula entropia de Shannon do texto.
   - Mede taxa de compressao (ex.: gzip/bz2/zstd).
   - Gera tabela de features por filme.

5. `05_correlation_analysis.py`
   - Junta features com `rating`/`votes`.
   - Executa correlacoes (Pearson/Spearman) e regressao simples.
   - Exporta tabelas e graficos finais.

## Status atual da coleta de dados

| Metrica | Valor |
|---------|-------|
| Total de filmes IMDb | 10060 |
| IMDb IDs em movies_subtitles.csv | 4666 |
| Filmes com legenda coletados | 1368 |

## Fontes de dados

1. **IMDb Datasets**
   - URL: https://datasets.imdbws.com/
   - Arquivos: title.basics.tsv.gz, title.ratings.tsv.gz

2. **Movie Subtitle Dataset (Kaggle)**
   - URL: https://www.kaggle.com/datasets/adiamaan/movie-subtitle-dataset
   - Arquivo: movies_subtitles.csv (4.8GB)
   - Colunas: start_time, end_time, text, imdb_id

## Observacao

Os scripts 01-05 ja foram implementados conforme o andamento do projeto; os arquivos 06+ podem ser criados conforme a necessidade de analise de entropia/compressao e correlacoes.
