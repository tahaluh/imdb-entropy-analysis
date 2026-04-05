# Relacao Entre Entropia de Legendas e Nota IMDb

## Resumo

Este projeto investiga se metricas informacionais extraidas de legendas de filmes
(entropia, compressao e estatisticas de texto) possuem relacao com as notas do IMDb.

Pipeline implementado do script 01 ao 12, com coleta, limpeza, filtragem,
extracao de features, analise estatistica e modelagem preditiva.

## Objetivo

Objetivo principal:

- testar se a estrutura informacional das legendas explica variacao da nota IMDb.

Pergunta de pesquisa:

- existe sinal estatistico entre entropia/compressao de legenda e rating?

## Fontes de Dados

1. IMDb Datasets
- URL: https://datasets.imdbws.com/
- Arquivos: title.basics.tsv.gz, title.ratings.tsv.gz

2. Kaggle Movie Subtitle Dataset
- URL: https://www.kaggle.com/datasets/adiamaan/movie-subtitle-dataset
- Arquivo: movies_subtitles.csv
- Colunas base: start_time, end_time, text, imdb_id

## Pipeline Executado

1. 01_collect_imdb.py
- coleta e filtra filmes do IMDb.

2. 02_plot_movies.py
- analise exploratoria do dataset de filmes.

3. 03_collect_subtitles.py
- estrutura para coleta por API (na pratica, fluxo seguiu com dataset Kaggle).

4. 04_merge_subtitles_with_movies.py
- agrega legendas por imdb_id e junta com metadados de filmes.

5. 05_plot_movies_with_subtitles.py
- analise exploratoria para conjunto com legendas.

6. 06_normalize_subtitles.py
- limpeza e normalizacao de texto das legendas.

7. 07_filter_subtitles.py
- filtro de qualidade por tamanho da legenda:
- 500 <= segment_count <= 6000.

8. 08_extract_information_features.py
- extracao de features informacionais (entropia e compressao).

9. 09_analysis.py
- correlacoes, scatter plots e resumos.

10. 10_linear_regression.py
- baseline com regressao linear simples.

11. 11_model_comparison_cv.py
- benchmark de modelos com validacao cruzada.

12. 12_feature_expansion.py
- expansao de features e comparacao de cenarios.

## Tamanho dos Conjuntos

| Etapa | Arquivo | Linhas |
|---|---|---:|
| 01 | data/processed/movies.csv | 10060 |
| 04 | data/processed/movies_with_subtitle_stats.csv | 1368 |
| 06 | data/processed/movies_with_subtitles.csv | 1368 |
| 07 | data/processed/movies_filtered.csv | 1304 |
| 08 | data/processed/movies_information_features.csv | 1304 |

## Features Utilizadas

Entropia:

- char_entropy
- bigram_entropy
- trigram_entropy
- word_entropy

Compressao:

- gzip_ratio, bz2_ratio, lzma_ratio
- gzip_saving, bz2_saving, lzma_saving
- gzip_bits_per_byte, bz2_bits_per_byte, lzma_bits_per_byte

Texto e estrutura de legenda:

- segment_count
- clean_text_words
- unique_words
- clean_text_length
- avg_word_length
- avg_words_per_segment

## Resultados Principais

1. Cenario somente informacional/textual
- correlacoes com rating geralmente fracas (ordem baixa, prox. de zero).
- modelos com essas variaveis isoladas ficaram com R2 muito baixo.

2. Cenario com features ampliadas (metadados adicionais)
- melhor desempenho observado com XGBoost.
- R2 em torno de 0.40 no melhor cenario combinado.

3. Interpretacao
- entropia e compressao possuem sinal, mas fraco de forma isolada.
- nota IMDb parece depender tambem de fatores nao textuais.

## Conclusao

Conclusao tecnica:

- as metricas informacionais de legenda nao explicam sozinhas, com alta forca,
  a variacao de rating IMDb neste recorte.
- em conjunto com variaveis adicionais, o desempenho melhora substancialmente.

Conclusao de pesquisa:

- a hipotese de relacao existe em baixa magnitude no cenario puro de texto,
  e se fortalece em modelos multivariados com mais contexto.

## Reproducao

Executar em ordem:

1. python3 01_collect_imdb.py
2. python3 04_merge_subtitles_with_movies.py
3. python3 06_normalize_subtitles.py
4. python3 07_filter_subtitles.py
5. python3 08_extract_information_features.py
6. python3 09_analysis.py
7. python3 10_linear_regression.py
8. python3 11_model_comparison_cv.py
9. python3 12_feature_expansion.py

## Proxima Etapa: Relatorio

Sugestao de secoes do relatorio final:

1. Introducao e motivacao
2. Base teorica (entropia, compressao, correlacao)
3. Metodologia e pipeline
4. Resultados experimentais
5. Discussao e limitacoes
6. Conclusao e trabalhos futuros
