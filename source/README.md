# Pipeline por Arquivo

Este diretorio documenta as etapas implementadas da pesquisa sobre entropia/compressao de legendas e nota IMDb.

## Etapas implementadas (01-12)

1. `01_collect_imdb.py`
- Coleta e filtra filmes do IMDb.

2. `02_plot_movies.py`
- EDA de filmes sem legenda.

3. `03_collect_subtitles.py`
- Estrutura para coleta por API (dataset Kaggle usado na pratica).

4. `04_merge_subtitles_with_movies.py`
- Agrega e junta legendas com filmes IMDb.

5. `05_plot_movies_with_subtitles.py`
- EDA do dataset com legenda.

6. `06_normalize_subtitles.py`
- Normalizacao textual e features basicas.

7. `07_filter_subtitles.py`
- Filtro: `500 <= segment_count <= 6000`.

8. `08_extract_information_features.py`
- Entropia + compressao (core informacional).

9. `09_analysis.py`
- Correlacoes, scatter plots e sumarios por grupo.

10. `10_linear_regression.py`
- Baseline de regressao linear.

11. `11_model_comparison_cv.py`
- Benchmark com validacao cruzada.

12. `12_feature_expansion.py`
- Expansao de features e comparacao de cenarios.

## Estado atual dos dados

| Dataset | Linhas |
|---|---:|
| `movies.csv` | 10060 |
| `movies_with_subtitle_stats.csv` | 1368 |
| `movies_with_subtitles.csv` | 1368 |
| `movies_filtered.csv` | 1304 |
| `movies_information_features.csv` | 1304 |

## Leitura para o relatorio

1. Cenario so com variaveis informacionais/textuais:
- correlacao fraca com rating
- capacidade preditiva baixa (R2 proximo de 0)

2. Cenario com features ampliadas:
- melhora relevante de desempenho
- melhor resultado observado proximo de R2 = 0.40

3. Interpretacao:
- entropia/compressao tem sinal, mas insuficiente de forma isolada
- contexto adicional melhora a explicacao de nota
