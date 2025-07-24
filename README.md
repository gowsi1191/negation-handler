Negation plays a pivotal role in clinical information retrieval, yet it remains poorly handled by most neural retrieval systems. Vector-based models like BGE (BAAI General Embedding) rely on dense similarity, which often fails to capture the semantic contradiction expressed in queries containing negation (e.g., “excluding warfarin”). This frequently leads to false positives—documents that mention excluded terms instead of filtering them out.
We propose a lightweight hybrid reranking pipeline for negation-aware medical search that achieves high accuracy without requiring large-scale supervised training. Our system first uses BGE to retrieve the top x candidate documents from a 1M+ corpus using a tiered threshold sampling strategy. If the query contains explicit negation, we rerank these using a combination of DeBERTa-v3-MNLI, which produces entailment, neutral, and contradiction scores, and a small-data-trained XGBoost classifier that learns to prioritize truly relevant content. XGBoost outputs a predicted relevance score for each document, and we generate the final ranking by sorting the candidates in descending order of these scores. Remarkably, our reranker is trained on just 35 labeled negation queries.
On a held-out test set of 86 queries, our model achieves a Precision@1 of 91.76%, significantly outperforming the BGE baseline (True PRECISION@1: 47.06%). Additional metrics further validate the system’s robustness: MRR@2 = 0.9529, nDCG@2 = 0.9622, and PRECISION@2 = 0.8000. A margin mean of 0.3817 confirms strong separation between relevant and irrelevant candidates. These results highlight that contradiction-aware reranking, even with minimal supervision, can dramatically improve performance for negation-sensitive clinical queries. 
<img width="468" height="317" alt="image" src="https://github.com/user-attachments/assets/1b0f5194-cc21-4020-b08d-e0b12ea2e49e" />

# negation-handler
benchmark.py
compares the performance of BGE embedding similarity and NLI-based contradiction scoring across clinical queries with negation. It evaluates each query–document pair using DeBERTa-MNLI and BGE models, logging entailment probabilities, cosine similarity scores, and inference time per document.

bge_p@1.py
Evaluates top-2 ranking effectiveness for BGE by computing standard IR metrics like True P@1, Adjusted P@1, P@2, MRR@2, and nDCG@2 across all queries in a given JSON file. It helps quantify how well the top-ranked documents align with negation-aware relevance labels.

gen_p@1.py
Calculates Precision@1 for multiple models by evaluating the top-ranked document's relevance across saved JSON files in a folder. It prints out each model’s P@1 score along with the count of correct predictions.

xg_best_train.py
Iterates through all .json files in the directory, extracts NLI-based features (e, n, c), trains an XGBoost model per file with 5-fold cross-validation, reports F1 scores, and saves each trained model with a unique filename.

xg_best_test.py
Evaluates an XGBoost model for negation-aware document reranking. It computes relevance-based scores for each query–document pair using NLI features, calculates P@1, P@2, MRR@2, nDCG@2, and margin stats, and visualizes score distributions via a scatter plot.

2explicit.json
has 34 dataset to train


2explicit_test.json
has 85 dataset to test 
