# Technical Report: Architectural Optimization of the "La Ghigliottina" Solver

**Author:** Stefano Colella (Master in Computer Science and Artificial Intelligence)  
**Context:** Development of an Information Retrieval and Reranking system for the "La Ghigliottina" NLP task (inspired by the EVALITA evaluation frameworks).

---

## Getting Started

### How to Run
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and execute the main Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

### Project Structure
- `main.ipynb`: The primary executable notebook containing the project pipeline and experiments.
- `requirements.txt`: List of Python dependencies required to run the project.
- `scripts/`: Auxiliary Python scripts for data processing and model training.
- `corpora/`: Large unstructured text corpora used for the distributional architecture. *(Excluded from repo due to size)*
- `dataset_ghigliottina/`: Historical dataset of "La Ghigliottina" games.
- `polirematiche_proverbi/`: Local knowledge bases, dictionaries, and idioms used for the symbolic architecture.
- `dashboard/`: Web dashboard application files.
- `grafi/`: Serialized graph structures (Lexical Knowledge Graph). *(Large matrices excluded from repo)*
- `output/`: Generated outputs, models, and evaluation results.
- `papers/` & `images/`: Project documentation and graphical assets.

### Data Availability & Reproducibility
> **Note on executing the code**: Due to GitHub's strict 100MB file limit, the massive text corpora (Paisà, Wikipedia, OpenSubtitles - ~10GB total) and the compiled distributional graph matrices (`.npz` cache) are excluded from this repository. 
> 
> The repository serves as a complete **reference codebase**. You can read through `main.ipynb` and the Python scripts to understand the architecture, logic, and results. To execute the distributional pipeline locally, you must independently download the source corpora and rebuild the sparse matrix using `scripts/build_fast_sparse_knowledge.py`.

---

## 1. Project Objective

The project aims to develop an Artificial Intelligence agent capable of solving the "La Ghigliottina" language game. The system receives 5 clue words (hints) and must infer the target word. To achieve State-of-the-Art (SOTA) performance, the research explored two opposing AI paradigms: logical extraction from structured dictionaries (Symbolic AI) and statistical extraction from massive unstructured corpora (Distributional Machine Learning), ultimately converging towards a hybrid LTR (Learning-to-Rank) architecture.

---

## 2. Phase 1: Symbolic Architecture (Lexical Knowledge Graph)

The first phase of development focused on creating a deterministic system based on highly qualified and local domain knowledge bases: `demauro.poli`, `polirematiche`, and `proverbi` (totaling approximately 36,000 exact intersection rows), integrated with 80% of the game's historical dataset.

### 2.1. Topological Extraction Methodology
A Lexical Knowledge Graph (LKG) based on an Inverted Index in volatile memory ($\mathcal{O}(1)$ read time) was engineered. The algorithm performs a Path-Based Reasoning search starting from the 5 hint-nodes. To overcome the basic heuristic, the system was enhanced with a **Learning-to-Rank (XGBoost)** model operating in a vector space based exclusively on *topological features*, including:
* Source decoupling (differentiated weights for Proverbs vs. Multi-word expressions).
* Node Degree Centrality (Statistical penalization of linguistic Hubs to filter false positives).

To prevent overfitting (Data Leakage), the model was trained applying a strict Hold-out Split (80% for graph population, 20% blind for decision tree training).

### 2.2. Empirical Results (Symbolic Approach)
Isolating purely logical and idiomatic dynamics yielded significant metrics on the 100-instance Test Set:

| Evaluation Metric | Result (Symbolic LKG Architecture) |
| :--- | :--- |
| **Accuracy @1 (Exact Match)** | **21.00%** |
| **Accuracy @5 (Top-5 Window)** | **42.00%** |
| **Mean Reciprocal Rank (MRR)** | **0.2957** |
| **Average Inference Time** | **< 0.10s per game** (Zero Timeout) |

The high Accuracy @5 (42.00%) demonstrates the Symbolic Graph's effectiveness in precision *Retrieval* on strong associations (e.g., exact idioms). The observed limitation is the so-called **Knowledge Bottleneck**: modern or nuanced linguistic associations not encoded in the text files cannot be extracted.

---

## 3. Phase 2: Distributional Architecture on Massive Corpora

To break the physiological ceiling imposed by local dictionaries, the pipeline was rewritten to ingest terabytes of unstructured data (Wikipedia, Paisà, OpenSubtitles). This implementation resolves memory bottlenecks by adopting SOTA paradigms for IR and matrix computation.

### 3.1. SparseCooccurrenceEngine (Unsupervised Knowledge)
The ingestion module processes texts by extracting syntactic co-occurrences. To prevent RAM collapse, the tensors generated by the documents are periodically dumped in a compressed format (Chunking) and algebraically summed to a master `scipy.sparse.csr_matrix`. The FW-NPMI (Frequency-Weighted Normalized Pointwise Mutual Information) metric calculation occurs via direct indexing in constant time $\mathcal{O}(1)$.

### 3.2. Topological Consolidation and Results (Matrix Pruning)
In the asymmetrical configuration ($\sim$ 2.1M Paisà docs, $\sim$ 1.58M Wikipedia docs, $\sim$ 37.7M Subs docs), the system generates a Graph with hundreds of millions of edges. The activation of the **Matrix Pruning** module ($f_{xy} \ge 2$) reduced topological dimensionality by eliminating entropic co-occurrences.

The reranking delegated to XGBoost, operating solely in the distributional space, yielded:

| Evaluation Metric | Result (Pure NPMI) |
| :--- | :--- |
| **Accuracy @1 (Exact Match)** | **19.00 - 21.00%** |
| **Accuracy @5 (Top-5 Window)** | **27.00%** |

The analytically theorized limit is confirmed: the encyclopedia's inclusion causes an explosion of *entropic noise (Long Tail)*. Relying solely on statistics dilutes the signal, rewarding formal associations at the expense of lateral thinking.

---

## 4. Phase 3: SOTA Hybrid Architecture (Retrieve-and-Rerank)

In accordance with academic literature produced by EVALITA frameworks (where purely symbolic or distributional systems show structural gaps, but their combined *Theoretical Upper Bound* exceeds 70%), the final step involved developing a **Hybrid Ensemble System**.

### 4.1. Engineering the Expanded Vector Space
The infrastructure merges the two approaches:
1.  **Deterministic Retrieval (LKG as a Filter):** Local dictionaries act as a primary filter, ensuring execution times in tenths of a second and extracting a candidate pool to minimize noise.
2.  **Distributional Feature Injection:** The vector space intended for Machine Learning was expanded to $\mathbb{R}^8$. The determining feature ($f_8$) contains the Historical NPMI Score extracted from the massive matrix built on the Corpora.
3.  **XGBoost Regularization:** The Reranker dynamically balances the *strong logical features* ($f_1, f_3$) extracted from the rigid Symbolic Graph and the distributional metric ($f_8$), penalizing the latter when in conflict with the exact topological intersection.

### 4.2. Data Leakage Prevention
A critical architectural element is the strict Hold-out Split applied to `train.json` to ensure proper generalization:
* **Topological Injection (80% of Train Set):** This portion populates the Graph with explicit `GAME_HISTORY` labels, endowing the algorithm with long-term memory for recurring television patterns.
* **XGBoost Vectorization (20% of Train Set):** This portion undergoes Retrieval. Extracted candidates are assigned label `1` (Real Solution) or `0` (Negative Samples).

If XGBoost trained on nodes already known to the Graph (the 80%), the historical feature ($f_6$) would obtain artificial *Information Gain* (Data Leakage), blinding the model to pure statistics.

### 4.3. Empirical Results (Hybrid Model)

| Evaluation Metric | Result (SOTA LTR Hybrid Ensemble) | Variance vs Symbolic Baseline |
| :--- | :--- | :--- |
| **Accuracy @1 (Exact Match)** | **33.00%** | **+12.00%** (Absolute) |
| **Accuracy @5 (Top-5 Window)** | **48.00%** | **+6.00%** (Absolute) |
| **Mean Reciprocal Rank (MRR)** | **0.3950** | **+0.0993** |
| **Average Inference Time** | **~0.12s per game** | (Real-time Systems) |

The Information Gain extracted from the decision tree provides mathematical proof of the balancing: the web-related statistical feature ($f_8$) absorbs **16.21%** of the decision weight, harmoniously joining the topological *core features* ($f_1$: 30.48%, $f_3$: 24.32%).

---

## 5. Generative Evaluation (Explainable AI via LLM)

To complete the MLOps infrastructure, a cloud **Explainable AI (XAI)** module was integrated.
Leveraging the asynchronous Groq LPU APIs and the *LLaMA-3.3-70B* model, the system performs zero-shot generation of the natural language semantic explanation linking the 5 hints to the target word identified by the hybrid architecture.

The Ground Truth generated by the Large Language Model underwent automated academic NLG (Natural Language Generation) evaluation:
* **BLEU Score (Lexical Precision):** **~0.1069**
* **ROUGE-L F1 (Lexical Recall):** **~0.3265**
* **BERTScore F1 (Vector Semantics):** **~0.7636**

The BERTScore attests to an excellent quality level (0.76) in the semantic alignment between the machine-generated explanations and the human logic required by the game's domain.

The observed divergence between the high BERTScore and the lower BLEU and ROUGE-L metrics is an expected outcome of the prompt engineering strategy and highlights the structural limitations of $n$-gram-based evaluation. The Large Language Model was instructed via few-shot prompting to generate comprehensive, discursive explanations for each semantic link. Consequently, the generated text introduces explanatory tokens and connective phrasing that are completely absent from the highly concise, telegraphic human-written ground truth. This verbosity heavily penalizes exact string-matching metrics like BLEU (Precision) and ROUGE (Recall). Conversely, the Multilingual BERTScore bypasses string-matching constraints by computing pairwise cosine similarity on contextual token embeddings. It correctly recognizes that the underlying semantic intent of the LLM perfectly encapsulates the human ground truth. This validates the excellent 0.76 score, confirming that the machine successfully captured the associative logic without being mathematically penalized for its discursive syntactic formulation.

---

## 6. Conclusions

The shift from 21.00% to 33.00% in *Accuracy @1* unequivocally demonstrates the research's structural thesis: pure syntactic knowledge requires distributional semantics to disambiguate the complex polysemy of lateral thinking. The integration of Vector Caching, Asynchronous Throttling, and Regularized Reranking techniques produced a robust solver, empirically documented and compliant with the highest architectural standards for Italian computational linguistics tasks.