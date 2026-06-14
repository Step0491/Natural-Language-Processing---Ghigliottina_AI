import os
import sys
import time
import pickle
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from tqdm.auto import tqdm
import spacy
import multiprocessing

class FastSparseIngestor:
    def __init__(self, vocab_path="grafi/cache_grafo_P800K_W300K_S1M_meta.pkl", matrix_path="grafi/cache_grafo_P800K_W300K_S1M_matrix.npz"):
        self.vocab_path = vocab_path
        self.matrix_path = matrix_path
        
        self.word2id = {}
        self.id2word = {}
        self.marginal_counts = defaultdict(int)
        self.total_pairs = 0
        
        self.MAX_BUFFER = 10_000_000  
        self.row_buffer = []
        self.col_buffer = []
        self.data_buffer = []
        
        self.master_matrix = None

    def _get_id(self, word):
        if word not in self.word2id:
            new_id = len(self.word2id)
            self.word2id[word] = new_id
            self.id2word[new_id] = word
        return self.word2id[word]

    def _flush_buffer_to_matrix(self):
        if not self.data_buffer: return
        
        current_size = len(self.word2id)
        shape = (current_size, current_size)
        
        chunk_csr = sp.coo_matrix(
            (self.data_buffer, (self.row_buffer, self.col_buffer)), 
            shape=shape, 
            dtype=np.float32
        ).tocsr()
        
        if self.master_matrix is None:
            self.master_matrix = chunk_csr
        else:
            if self.master_matrix.shape != shape:
                self.master_matrix.resize(shape)
            self.master_matrix = self.master_matrix + chunk_csr
            
        self.row_buffer.clear()
        self.col_buffer.clear()
        self.data_buffer.clear()

    def ingest_corpus(self, filepath, nlp, max_lines):
        if not os.path.exists(filepath):
            print(f"[!] Source file not found: {filepath}. Skipping pipeline.")
            return
            
        print(f"\n[*] Starting asynchronous streaming processing on: {filepath}")
        print(f"[*] Reading target set to: {max_lines:,} rows.")
        
        def stream_lines():
            lines_read = 0
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if lines_read >= max_lines: break
                    cleaned = line.strip()
                    if cleaned.startswith('<') and cleaned.endswith('>'): continue
                    yield cleaned
                    lines_read += 1

        docs = nlp.pipe(stream_lines(), batch_size=512, n_process=6)
        
        for doc in tqdm(docs, total=max_lines, desc="Pipeline Throughput", unit=" doc"):
            words = [
                t.lemma_.lower().strip() for t in doc 
                if t.pos_ in {"NOUN", "ADJ", "VERB"} 
                and t.is_alpha and not t.is_stop and len(t.lemma_) > 2
            ]
            
            len_words = len(words)
            if len_words < 2: continue
            
            word_ids = [self._get_id(w) for w in words]
            
            for idx in range(len_words):
                w_id = word_ids[idx]
                self.marginal_counts[w_id] += 1
                
                start_idx = max(0, idx - 2)
                end_idx = min(len_words, idx + 3)
                
                for j in range(start_idx, end_idx):
                    if j != idx:
                        neighbor_id = word_ids[j]
                        self.row_buffer.append(w_id)
                        self.col_buffer.append(neighbor_id)
                        self.data_buffer.append(1.0)
                        self.total_pairs += 1
                        
            if len(self.data_buffer) >= self.MAX_BUFFER:
                self._flush_buffer_to_matrix()

        self._flush_buffer_to_matrix()

    def finalize_and_save(self):
        if self.master_matrix is None:
            print("[!] No data ingested. Saving process aborted.")
            return
            
        print("\n[*] Starting sparse memory consolidation and optimization...")
        
        self.master_matrix.data[self.master_matrix.data < 2.0] = 0.0
        self.master_matrix.eliminate_zeros()
        
        marginal_words = {self.id2word[i]: count for i, count in self.marginal_counts.items()}
        
        # CRITICAL FIX: The key must be 'word2id' for compatibility with main.py
        meta_data = {
            'word2id': self.word2id,
            'marginal_counts': marginal_words,
            'total_pairs': self.total_pairs
        }
        
        print(f"    [+] Final graph density (Valid edges): {self.master_matrix.nnz:,}")
        print(f"    [+] Unified Vocabulary Size: {len(self.word2id):,} lemmas.")
        
        # Safe creation of the grafi/ directory
        os.makedirs(os.path.dirname(self.vocab_path), exist_ok=True)
        
        print(f"[*] Writing metadata to: {self.vocab_path}")
        with open(self.vocab_path, 'wb') as f:
            pickle.dump(meta_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"[*] Writing NPZ compressed tensors to: {self.matrix_path}")
        sp.save_npz(self.matrix_path, self.master_matrix)
        print("[SUCCESS] Computation and writing pipeline completed successfully.")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # Naming consistent with main
    FILE_META = "grafi/cache_grafo_P800K_W300K_S1M_meta.pkl"
    FILE_MATRICE = "grafi/cache_grafo_P800K_W300K_S1M_matrix.npz"
    
    print("============================================================")
    print("        FAST SPARSE INGESTOR (OOM-FREE ARCHITECTURE)        ")
    print("============================================================")
    
    print("[*] Optimized loading of spaCy it_core_news_lg...")
    nlp = spacy.load("it_core_news_lg", disable=["parser", "ner", "senter"])
    print("[*] NLP Pipeline allocated. Starting processing...\n")
    
    ingestor = FastSparseIngestor(vocab_path=FILE_META, matrix_path=FILE_MATRICE)
    
    ingestor.ingest_corpus('corpora/corpus_paisa.txt', nlp, max_lines=800_000)
    ingestor.ingest_corpus('corpora/wikipedia_puro.jsonl', nlp, max_lines=300_000)
    ingestor.ingest_corpus('corpora/OpenSubtitles.txt', nlp, max_lines=1_000_000)
    
    ingestor.finalize_and_save()