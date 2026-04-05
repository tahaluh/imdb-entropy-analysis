# Projeto Final - PDI

Objetivo: investigar a relacao entre caracteristicas informacionais de legendas (entropia, compressao e metadados de texto) e nota no IMDb.

## 1) Ambiente

No Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Fonte dos dados

1. IMDb Datasets
- https://datasets.imdbws.com/
- Arquivos usados: `title.basics.tsv.gz`, `title.ratings.tsv.gz`

2. Kaggle Movie Subtitle Dataset
- https://www.kaggle.com/datasets/adiamaan/movie-subtitle-dataset
- Arquivo usado: `movies_subtitles.csv`
- Colunas base: `start_time`, `end_time`, `text`, `imdb_id`

## 3) Pipeline implementado (01 a 12)

1. `01_collect_imdb.py`
- Coleta e filtra filmes do IMDb.

2. `02_plot_movies.py`
- Graficos exploratorios do dataset de filmes.

3. `03_collect_subtitles.py`
- Estrutura para coleta de legendas por API (atual: dataset Kaggle ja integrado no fluxo).

4. `04_merge_subtitles_with_movies.py`
- Agrega legenda por `imdb_id` e junta com `movies.csv`.

5. `05_plot_movies_with_subtitles.py`
- Graficos para o dataset combinado com legendas.

6. `06_normalize_subtitles.py`
- Normaliza texto de legendas e gera features textuais basicas.

7. `07_filter_subtitles.py`
- Filtro de qualidade/tamanho: `500 <= segment_count <= 6000`.

8. `08_extract_information_features.py`
- Extrai entropia e metricas de compressao (core do trabalho).

9. `09_analysis.py`
- Correlacoes, graficos e resumos exploratorios.

10. `10_linear_regression.py`
- Regressao linear simples para prever `rating`.

11. `11_model_comparison_cv.py`
- Comparacao de modelos com validacao cruzada (inclui XGBoost quando instalado).

12. `12_feature_expansion.py`
- Expansao de features e benchmark de modelos em dois cenarios:
  - com metadados adicionais (ex.: genero/votos)
  - somente informacionais/textuais

## 4) Tamanho dos datasets por etapa

| Etapa | Output | Registros |
|---|---|---:|
| 01 | `data/processed/movies.csv` | 10060 |
| 04 | `data/processed/movies_with_subtitle_stats.csv` | 1368 |
| 06 | `data/processed/movies_with_subtitles.csv` | 1368 |
| 07 | `data/processed/movies_filtered.csv` | 1304 |
| 08 | `data/processed/movies_information_features.csv` | 1304 |

## 5) Features informacionais (etapa 08)

Entropia:
- `char_entropy`
- `bigram_entropy`
- `trigram_entropy`
- `word_entropy`

Compressao:
- `gzip_ratio`, `bz2_ratio`, `lzma_ratio`
- `gzip_saving`, `bz2_saving`, `lzma_saving`
- `gzip_bits_per_byte`, `bz2_bits_per_byte`, `lzma_bits_per_byte`

Metadados textuais de apoio:
- `segment_count`, `clean_text_words`, `unique_words`, `clean_text_length`, `avg_word_length`, `avg_words_per_segment`

## 6) Principais resultados

1. Sinal informacional isolado (entropia/compressao/palavras)
- Correlacoes com `rating` fracas (ordem de ~0.02 a ~0.12 em valor absoluto na maioria dos casos).
- Modelagem com apenas features informacionais/textuais: desempenho baixo (`R2` proximo de 0).

2. Cenario com features ampliadas (12 com metadados extras)
- Melhor modelo: XGBoost
- `R2` aproximado: 0.40
- Isso indica ganho preditivo relevante quando contexto adicional e incluído.

3. Interpretacao objetiva
- As features informacionais de legenda possuem sinal, mas fraco quando analisadas sozinhas.
- A nota IMDb depende de fatores adicionais alem da estrutura informacional do texto da legenda.

## 7) Conclusao para o relatorio

Conclusao principal: neste recorte de dados, entropia/compressao de legendas nao explicam sozinhas a variacao de nota IMDb com alta forca, mas contribuem como parte de um conjunto maior de variaveis.

Conclusao metodologica: a hipotese de relacao existe em baixa magnitude no cenario univariado/simples e melhora em modelos multivariados com contexto adicional.

## 8) Como reproduzir rapidamente

```bash
python3 01_collect_imdb.py
python3 04_merge_subtitles_with_movies.py
python3 06_normalize_subtitles.py
python3 07_filter_subtitles.py
python3 08_extract_information_features.py
python3 09_analysis.py
python3 10_linear_regression.py
python3 11_model_comparison_cv.py
python3 12_feature_expansion.py
```
