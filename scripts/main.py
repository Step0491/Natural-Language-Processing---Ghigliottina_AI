import sys, os, json, time, pickle, warnings, math
from collections import defaultdict

import numpy as np
import scipy.sparse
import xgboost as xgb

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from scripts.xai_generator import ExplainableAIGenerator

warnings.filterwarnings('ignore')

# =========================================================
# GESTIONE DINAMICA STOPWORDS ITALIANE (NLTK Corpus)
# =========================================================
# Download automatico e silenzioso del corpus se mancante
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# 1. Caricamento base standard
BASE_STOPWORDS = set(stopwords.words('italian'))

# 2. Estensione con lemmi/particelle specifiche del parsing
CUSTOM_STOPWORDS = {"c'è", "c'era", "ecco", "chiunque", "faccio", "fai", "fa", "facciamo", "fate", "fanno"}
STOPWORDS_EXTENDED = BASE_STOPWORDS.union(CUSTOM_STOPWORDS)

# 3. Preservazione semantica (Parole vitali per la logica del gioco)
PAROLE_VITALI = {"niente", "tutto", "fare", "fatto", "essere", "stato", "avere", "avuto", "contro", "senza"}

# 4. Insieme finale e POS Tags
STOPWORDS_FINAL = STOPWORDS_EXTENDED - PAROLE_VITALI
POS_TAGS = {"sos", "avv", "agg", "ver", "pre", "cong", "pron"}

# =========================================================
# INIZIALIZZAZIONE STEMMER
# =========================================================
STEMMER = SnowballStemmer('italian')

def stem(word):
    return STEMMER.stem(word.lower())

# =========================================================
# SYMBOLIC GRAPH AND FEATURE EXTRACTOR
# =========================================================
class SymbolicKnowledgeGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.source_weights = {
            'PROVERBI': 2.0,
            'POLIREMATICHE': 2.0,
            'DEMAURO_POLI': 1.5,
            'STORICO_GIOCO': 1.0
        }
        
        # =========================================================
        # LOADING WEB STATISTICAL MEMORY (SOTA MATRICES)
        # =========================================================
        self.vocab = {}
        self.marginal_counts = {}
        self.total_pairs = 1
        self.npmi_matrix = None
        self.has_npmi = False
        
        meta_path = "grafi/cache_grafo_P800K_W300K_S1M_meta.pkl" 
        matrix_path = "grafi/cache_grafo_P800K_W300K_S1M_matrix.npz"
        
        if os.path.exists(meta_path) and os.path.exists(matrix_path):
            print(f"[*] Initializing NPMI Tensors from local disk...")
            try:
                # 1. Metadata and O(1) Mapping
                with open(meta_path, 'rb') as f:
                    state = pickle.load(f)
                    
                if 'word2id' in state:
                    self.vocab = state['word2id']
                    self.marginal_counts = state['marginal_counts']
                    self.total_pairs = state['total_pairs']
                    
                    # 2. Sparse Structure
                    self.npmi_matrix = scipy.sparse.load_npz(matrix_path).tocsr()
                    self.has_npmi = True
                    
                    print(f"    [+] NPMI Engine Hooked.")
                    print(f"    [+] Vocabulary: {len(self.vocab):,} lemmas.")
                    print(f"    [+] Sparse Edges: {self.npmi_matrix.nnz:,} active connections.")
                else:
                    print("    [!] Incompatible Pickle File: Key 'word2id' missing.")
            except Exception as e:
                print(f"    [!] Error loading tensors: {e}")

        # SOTA Regularized XGBoost Initializer (Anti-Masking)
        # BUGFIX: Removed deprecated use_label_encoder parameter
        self.ranker = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.7,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        self.is_trained = False
        
    def _compute_npmi_sparse(self, w1, w2):
        """Correct O(1) vector computation for integer IDs and fallback to -1.0"""
        if not self.has_npmi: return -1.0
        
        id1 = self.vocab.get(w1)
        id2 = self.vocab.get(w2)
        if id1 is None or id2 is None: return -1.0
        
        f_xy = max(self.npmi_matrix[id1, id2], self.npmi_matrix[id2, id1])
        if f_xy <= 0: return -1.0
            
        f_x = self.marginal_counts.get(id1, 0)
        f_y = self.marginal_counts.get(id2, 0)
        
        if f_x <= 0 or f_y <= 0: return -1.0
        
        p_xy = f_xy / self.total_pairs
        p_x = f_x / self.total_pairs
        p_y = f_y / self.total_pairs
        
        denominator = -math.log2(p_xy)
        if denominator <= 0: return -1.0
        
        dividend_ratio = p_xy / (p_x * p_y)
        if dividend_ratio <= 0: return -1.0
        
        return math.log2(dividend_ratio) / denominator

    def _extract_features(self, candidate_dict, hints_clean, hints_stems):
        """Expanded vector space to R^8 (7 Topological + 1 Web Statistical)"""
        f1 = candidate_dict["unique_hints"]
        f2 = candidate_dict["evidences_count"]
        f3 = candidate_dict["symbolic_score"]
        f4 = candidate_dict["count_proverbi"]
        f5 = candidate_dict["count_poli"]
        f6 = candidate_dict["count_storico"]
        f7 = candidate_dict["node_degree"]
        
        # CALCULATING f8 FEATURE: Average NPMI
        f8 = -1.0  # Starting baseline: No co-occurrence found
        if self.has_npmi:
            cand_word = candidate_dict["word"]
            cand_stem = stem(cand_word)
            
            npmi_scores = []
            for idx, h_clean in enumerate(hints_clean):
                h_stem = hints_stems[idx]
                score = self._compute_npmi_sparse(cand_word, h_clean)
                
                if score == -1.0:
                    score = self._compute_npmi_sparse(cand_stem, h_stem)
                    
                npmi_scores.append(score)
            
            if npmi_scores:
                f8 = sum(npmi_scores) / len(npmi_scores)

        return [f1, f2, f3, f4, f5, f6, f7, f8]
    
    def _clean_and_tokenize(self, phrase):
        clean = phrase.lower().replace("'", " ").replace(",", "").replace(".", "").replace(";", "").replace(":", "")
        tokens = clean.split()
        if tokens and (tokens[0].isdigit() or tokens[0] in POS_TAGS):
            tokens = tokens[1:]
        final = [stem(w) for w in tokens if len(w) > 1 and w not in STOPWORDS_FINAL]
        return final, phrase

    def ingest_knowledge(self, filepath, source_label):
        if not os.path.exists(filepath): return
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                phrase = line.strip()
                if not phrase: continue
                tokens, _ = self._clean_and_tokenize(phrase)
                for token in tokens:
                    self.graph[token].append((source_label, phrase))

    def get_raw_candidates(self, hints):
        """Pure Topological Extraction (Retrieval)"""
        hints_clean = [h.lower().strip() for h in hints]
        hints_stems = [stem(h) for h in hints_clean]

        evidence_board = defaultdict(list)
        for idx, hint in enumerate(hints_clean):
            s_hint = hints_stems[idx]
            if s_hint in self.graph:
                for source, phrase in self.graph[s_hint]:
                    tokens_in_phrase, _ = self._clean_and_tokenize(phrase)
                    for cand_token in tokens_in_phrase:
                        if cand_token != s_hint and cand_token not in hints_stems:
                            weight = self.source_weights.get(source, 1.0)
                            evidence_board[cand_token].append((hint, source, phrase, weight))

        raw_candidates = []
        for cand_stem, evidences in evidence_board.items():
            hint_to_max_weight = defaultdict(float)
            
            count_proverbi = count_poli = count_storico = 0
            
            for (h, src, phr, w) in evidences:
                if w > hint_to_max_weight[h]:
                    hint_to_max_weight[h] = w
                
                if src == 'PROVERBI': count_proverbi += 1
                elif src in ['POLIREMATICHE', 'DEMAURO_POLI']: count_poli += 1
                elif src == 'STORICO_GIOCO': count_storico += 1
            
            symbolic_score = sum(hint_to_max_weight.values())

            if symbolic_score >= 1.0:
                original = cand_stem
                for ev in evidences:
                    toks = ev[2].lower().replace("'", " ").split()
                    for t in toks:
                        if stem(t) == cand_stem and t not in hints_clean:
                            original = t
                            break
                    if original != cand_stem: break

                cand_dict = {
                    "word": original,
                    "unique_hints": len(hint_to_max_weight),
                    "evidences_count": len(evidences),
                    "symbolic_score": symbolic_score,
                    "count_proverbi": count_proverbi,
                    "count_poli": count_poli,
                    "count_storico": count_storico,
                    "node_degree": len(self.graph.get(cand_stem, [])),
                    "evidences": evidences
                }
                cand_dict["features"] = self._extract_features(cand_dict, hints_clean, hints_stems)
                raw_candidates.append(cand_dict)

        raw_candidates.sort(key=lambda x: (x["symbolic_score"], x["evidences_count"]), reverse=True)
        return raw_candidates[:50]

    def train_reranker(self, train_data):
        print("[*] Generating Vector Dataset for XGBoost...")
        X, y = [], []
        
        for item in train_data:
            gold = item['sol'].lower().strip()
            hints = [item[f'hint{j}'] for j in range(1, 6)]
            
            candidates = self.get_raw_candidates(hints)
            if not candidates: continue
            
            # BUGFIX: Ensure the Gold Solution is injected into the training set 
            # if the topological retrieval failed to fetch it in the Top-50.
            gold_in_cands = False
            
            for cand in candidates:
                X.append(cand["features"])
                if cand["word"] == gold:
                    label = 1
                    gold_in_cands = True
                else:
                    label = 0
                y.append(label)
                
            if not gold_in_cands:
                # Synthesize a baseline vector for the missing gold target
                # This forces XGBoost to learn to rely on f8 (NPMI) when topology fails
                dummy_gold_dict = {
                    "word": gold, "unique_hints": 0, "evidences_count": 0,
                    "symbolic_score": 0.0, "count_proverbi": 0, "count_poli": 0,
                    "count_storico": 0, "node_degree": 0
                }
                hints_clean = [h.lower().strip() for h in hints]
                hints_stems = [stem(h) for h in hints_clean]
                dummy_gold_feat = self._extract_features(dummy_gold_dict, hints_clean, hints_stems)
                
                X.append(dummy_gold_feat)
                y.append(1)
                
        print(f"[*] Training LTR on {len(X)} vectors...")
        # BUGFIX: Strictly type to float32 to prevent XGBoost memory warnings
        self.ranker.fit(np.array(X, dtype=np.float32), np.array(y, dtype=int))
        
        print("[*] Feature Importance (Information Gain):")
        feature_names = ['f1_hints', 'f2_edges', 'f3_sym_score', 'f4_prov', 'f5_poli', 'f6_storico', 'f7_degree', 'f8_NPMI']
        importances = self.ranker.feature_importances_
        for name, imp in zip(feature_names, importances):
            print(f"    - {name}: {imp:.4f}")
        self.is_trained = True
        print("[*] XGBoost Semantic Reranker Operational.")

    def reason_and_solve(self, hints):
        candidates = self.get_raw_candidates(hints)
        if not candidates: return []
        
        if self.is_trained:
            # Batch XGBoost Inference
            X_test = np.array([c["features"] for c in candidates], dtype=np.float32)
            probs = self.ranker.predict_proba(X_test)[:, 1] # Class 1 probability
            
            for i, c in enumerate(candidates):
                c["xgb_score"] = float(probs[i])
                
            candidates.sort(key=lambda x: x["xgb_score"], reverse=True)
            
        return candidates[:5]

    def save_state(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.graph), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, filepath):
        with open(filepath, 'rb') as f:
            self.graph = defaultdict(list, pickle.load(f))

# =========================================================
# MAIN EXECUTION (NO LEAKAGE, XGBOOST RERANKING + XAI)
# =========================================================
if __name__ == '__main__':
    print("=" * 60)
    print(" GHIGLIOTTINA – LKG + XGBOOST (L-T-R) ARCHITECTURE")
    print("=" * 60)

    GENERA_SPIEGAZIONI_LLM = True  

    kg = SymbolicKnowledgeGraph()
    CACHE_GRAPH = "grafi/kg_symbolic_xgb.pkl"

    # 1. DETERMINISTIC SPLIT
    try:
        with open('dataset_ghigliottina/train.json', 'r', encoding='utf-8') as f:
            full_train_data = [json.loads(line) for line in f]
    except Exception as e:
        print(f"[ERROR] Unable to read train.json: {e}")
        sys.exit(1)

    split_idx = int(len(full_train_data) * 0.8)
    train_graph_data = full_train_data[:split_idx]
    train_xgb_data = full_train_data[split_idx:]

    # 2. GRAPH CONSTRUCTION
    if os.path.exists(CACHE_GRAPH):
        print(f"[*] Loading LKG from {CACHE_GRAPH}...")
        kg.load_state(CACHE_GRAPH)
    else:
        print("[*] Building LKG...")
        kg.ingest_knowledge('polirematiche_proverbi/demauro.poli', 'DEMAURO_POLI')
        kg.ingest_knowledge('polirematiche_proverbi/polirematiche', 'POLIREMATICHE')
        kg.ingest_knowledge('polirematiche_proverbi/proverbi', 'PROVERBI')
        
        for item in train_graph_data:
            try:
                gold = item['sol'].lower().strip()
                hints = [item[f'hint{j}'].lower().strip() for j in range(1, 6)]
                for hint in hints:
                    kg.graph[stem(hint)].append(('STORICO_GIOCO', f"{hint} {gold}"))
                    kg.graph[stem(gold)].append(('STORICO_GIOCO', f"{hint} {gold}"))
            except: continue
        kg.save_state(CACHE_GRAPH)

    # 3. XGBOOST MODEL TRAINING
    kg.train_reranker(train_xgb_data)

    # 4. FINAL INFERENCE ON TEST SET
    try:
        with open('dataset_ghigliottina/test.json', encoding='utf-8') as f:
            test_set = [json.loads(line) for line in f]
    except Exception as e:
        print(f"[ERROR] Unable to read test.json: {e}")
        sys.exit(1)

    print(f"\n[EVALUATION] Starting test on {len(test_set)} games...")
    top1, top5, mrr_sum = 0, 0, 0.0
    total = len(test_set)
    start_total = time.perf_counter()

    for instance in test_set:
        gold = instance['sol'].lower().strip()
        hints = [instance[f'hint{j}'] for j in range(1, 6)]

        preds = kg.reason_and_solve(hints)
        pred_words = [p["word"] for p in preds]

        if pred_words:
            if pred_words[0] == gold: top1 += 1
            if gold in pred_words:
                top5 += 1
                mrr_sum += 1.0 / (pred_words.index(gold) + 1)

    print("\n" + "═" * 60)
    print(" FINAL METRICS LKG + XGBOOST (Sparsity Aware)")
    print("═" * 60)
    print(f" Accuracy@1     : {(top1 / total) * 100:.2f}%")
    print(f" Accuracy@5     : {(top5 / total) * 100:.2f}%")
    print(f" MRR            : {mrr_sum / total:.4f}")
    print(f" Total time     : {time.perf_counter() - start_total:.2f}s")
    print("═" * 60)

    # 5. POST-HOC EXPLAINABILITY (INVOKING EXTERNAL CLOUD MODULE)
    if GENERA_SPIEGAZIONI_LLM:
        xai = ExplainableAIGenerator() 
        xai.enrich_test_set(test_set, output_path="output/test_enriched.json")