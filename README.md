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