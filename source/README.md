# Pipeline proposto por arquivos

Este diretorio organiza as etapas da pesquisa sobre a relacao entre caracteristicas de legendas e nota no IMDb.

## Arquivos e etapas

1. `01_collect_imdb.py` (implementado)
   - Coleta e prepara base de filmes + notas do IMDb.

2. `02_plot_movies.py` (implementado)
   - Visualizacoes do dataset de filmes (distribuicao de notas, por ano, etc).

3. `03_collect_subtitles.py` (implementado)
   - Placeholder para coleta de legendas via API OpenSubtitles.

4. `04_merge_subtitles_with_movies.py` (implementado)
   - Junta movies.csv com metricas basicas de legendas (total de segmentos, duracao, etc).

5. `05_plot_movies_with_subtitles.py` (implementado)
   - Visualizacoes dos dados combinados (filmes + legendas).

6. `06_normalize_subtitles.py` (implementado)
   - Normaliza e limpa texto das legendas (lowercasing, tags, pontuacao, etc).
   - Calcula features de texto (comprimento, palavras unicas, riqueza vocabular).
   - Gera `movies_with_subtitles.csv` com todos os dados combinados.

7. `07_filter_subtitles.py` (implementado)
   - Filtra legendas por quantidade de linhas (500 <= segment_count <= 6000).
   - Remove outliers e legendas muito curtas/longas.
   - Saida final: `movies_filtered.csv` com 1304 filmes prontos para analise.

8. `08_extract_information_features.py` (implementado) **← NÚCLEO DO TRABALHO**
   - Calcula **entropia de Shannon** (caractere, bigrama, trigrama, palavra).
   - Calcula **compressao** (gzip, bz2, lzma): ratios, economia, bits/byte.
   - Saida: `movies_information_features.csv` - dados prontos para correlacao.

## Status atual da coleta de dados

| Metrica | Valor |
|---------|-------|
| Total de filmes IMDb | 10060 |
| IMDb IDs em movies_subtitles.csv | 4666 |
| Filmes com legenda coletados | 1368 |
| Filmes apos filtro (500-6000 linhas) | **1304** |

## Fontes de dados

1. **IMDb Datasets**
   - URL: https://datasets.imdbws.com/
   - Arquivos: title.basics.tsv.gz, title.ratings.tsv.gz

2. **Movie Subtitle Dataset (Kaggle)**
   - URL: https://www.kaggle.com/datasets/adiamaan/movie-subtitle-dataset
   - Arquivo: movies_subtitles.csv (4.8GB)
   - Colunas: start_time, end_time, text, imdb_id

## Observacao

Os scripts 01-08 ja foram implementados:
- **01-07**: pipeline de coleta, normalizacao e filtro
- **08**: extracao das features informacionais (entropia + compressao) ← analise principal

Os proximos scripts (09+) devem focar em:
- Analise de correlacao com ratings
- Visualizacoes e heat maps
- Regressoes e modelos preditivos
- Relatorio final e insights
