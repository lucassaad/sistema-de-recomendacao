# 🎵 Sistema de Recomendação de Músicas Nacionais com TF-IDF

Sistema de recomendação de músicas construído do zero em Python, sem o uso de bibliotecas de machine learning. Utiliza a técnica **TF-IDF** (Term Frequency - Inverse Document Frequency) combinada com **similaridade de cosseno** para gerar recomendações personalizadas com base no perfil do usuário.

> Projeto desenvolvido para a disciplina de Introdução à Inteligência Artificial (IIA).

---

## 📌 Sumário

- [Visão Geral](#visão-geral)
- [Como Funciona](#como-funciona)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Autores](#autores)

---

## Visão Geral

O sistema lê um dataset de músicas nacionais em formato CSV, constrói uma representação vetorial de cada música via TF-IDF e, a partir das avaliações fornecidas pelo usuário, monta um perfil de preferências. As recomendações são geradas calculando a similaridade de cosseno entre o perfil do usuário e cada música ainda não avaliada.

---

## Como Funciona

O pipeline segue estas etapas:

1. **Carregamento do dataset** — leitura do arquivo `dataset.csv`
2. **Construção do vocabulário** — conjunto de todos os termos únicos dos atributos das músicas
3. **Construção do corpus** — cada música é representada como um `Document` com seu `bag_of_attributes`
4. **Cálculo do TF-IDF** — vetorização de cada música com base em:
   - **TF** (Term Frequency): frequência relativa do termo no documento
   - **IDF** (Inverse Document Frequency): penalidade para termos muito comuns
5. **Perfil do usuário** — vetor construído a partir das avaliações ponderadas pelo rating do usuário
6. **Recomendação** — ranking por similaridade de cosseno entre o perfil e músicas não avaliadas

### Fórmulas

$$TF(t, d) = \frac{\text{contagem de } t \text{ em } d}{\text{total de termos em } d}$$

$$IDF(t) = \log\left(\frac{N}{DF(t)}\right)$$

$$TFIDF(t, d) = TF(t, d) \times IDF(t)$$

$$\cos(\theta) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

---

## Estrutura do Projeto

```
├── sistema_recomendacao_tfidf.ipynb  # Notebook principal
├── dataset.csv                       # Dataset de músicas nacionais
└── README.md
```

### Principais componentes

| Componente | Descrição |
|---|---|
| `Document` | Dataclass que representa uma música (identificador, bag of attributes, TF, TF-IDF) |
| `build_vocabulary` | Constrói o vocabulário global a partir do dataset |
| `build_corpus` | Transforma o dataset em uma lista de objetos `Document` |
| `compute_document_frequencies` | Conta em quantos documentos cada termo aparece |
| `compute_TF` / `compute_IDF` | Calcula as métricas individuais de TF e IDF |
| `tfidf` | Aplica o cálculo TF-IDF em todo o corpus |
| `build_user_profile` | Constrói o vetor de preferências do usuário |
| `cosine_similarity` | Mede a similaridade entre dois vetores |
| `recommend_songs` | Retorna as top-N músicas recomendadas |

---


### Dependências

```bash
pip install pandas
```

> As demais dependências (`csv`, `math`, `random`, `dataclasses`) fazem parte da biblioteca padrão do Python.

---


## Autores

- **Arthur Menezes Botelho** — Matrícula: 231003362
- **Lucas Saad Rodrigues** — Matrícula: 231035393
