import json
import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
import os

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def update_dashboard_json(bleu_val, rouge_val, bert_val):
    """Updates the last run in the JSON file using foolproof absolute paths."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "dashboard", "dashboard.json")
    
    print(f"\n[*] Attempting to save metrics to: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"[!] I/O ERROR: Cannot update. The file does not exist in the specified path.")
        return False
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if len(data) > 0:
            # THIS IS THE UPDATED ENGLISH KEY:
            data[-1]["xai_evaluation"] = {
                "bleu": float(round(bleu_val, 4)),
                "rouge_l": float(round(rouge_val, 4)),
                "bertscore_f1": float(round(bert_val, 4))
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"[✔] Dashboard updated successfully! Values are no longer 0.0.")
            return True
        else:
            print("[!] ERROR: The dashboard.json file is empty.")
            return False
    except Exception as e:
        print(f"[!] CRITICAL ERROR during JSON writing: {e}")
        return False
    
def calculate_nlp_metrics(pred_file="output/test_enriched.json", gold_file="dataset_ghigliottina/test.json"):
    print("=" * 60)
    print(" STARTING NLG METRICS EVALUATION (BLEU, ROUGE, BERTScore)")
    print("=" * 60)
    
    # Building absolute paths for input files as well
    base_dir = os.path.dirname(os.path.abspath(__file__))
    abs_pred_file = os.path.join(base_dir, os.path.normpath(pred_file))
    abs_gold_file = os.path.join(base_dir, os.path.normpath(gold_file))
    
    predictions = []
    references = []

    try:
        with open(abs_pred_file, 'r', encoding='utf-8') as f_pred:
            preds_data = [json.loads(line) for line in f_pred]
            
        with open(abs_gold_file, 'r', encoding='utf-8') as f_gold:
            golds_data = {json.loads(line)['id']: json.loads(line) for line in f_gold}
            
    except Exception as e:
        print(f"[!] Error loading JSON files: {e}")
        return False

    for p in preds_data:
        g_id = p['id']
        if g_id in golds_data and 'desc' in golds_data[g_id]:
            # Ignore API errors
            if "Errore API" not in p['desc'] and "Errore di connessione" not in p['desc']:
                predictions.append(p['desc'])
                references.append(golds_data[g_id]['desc'])
            
    if not predictions:
        print("[!] No valid description found for comparison in test_enriched.json.")
        return False

    print(f"[*] Evaluation in progress on {len(predictions)} aligned instances...")

    smoother = SmoothingFunction().method1
    rouge_calc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    bleu_scores = []
    rougeL_scores = []

    for ref, pred in zip(references, predictions):
        ref_tokens = nltk.word_tokenize(ref.lower())
        pred_tokens = nltk.word_tokenize(pred.lower())
        
        b_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
        bleu_scores.append(b_score)
        
        r_score = rouge_calc.score(ref, pred)
        rougeL_scores.append(r_score['rougeL'].fmeasure)

    print("[*] Calculating BERTScore (Model: bert-base-multilingual-cased)...")
    # Suppress transformers library warnings
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    P, R, F1 = score(
        predictions, 
        references, 
        lang="it", 
        model_type="bert-base-multilingual-cased", 
        verbose=True
    )

    final_bleu = np.mean(bleu_scores)
    final_rouge = np.mean(rougeL_scores)
    final_bert = F1.mean().item()

    print("\n" + "═" * 60)
    print(" GENERATION EVALUATION REPORT (EXPLAINABILITY)")
    print("═" * 60)
    print(f" Average BLEU Score   : {final_bleu:.4f}")
    print(f" Average ROUGE-L (F1) : {final_rouge:.4f}")
    print(f" BERTScore (F1)       : {final_bert:.4f}")
    print("═" * 60)
    
    # Formal save execution
    update_dashboard_json(final_bleu, final_rouge, final_bert)
    return True

if __name__ == '__main__':
    calculate_nlp_metrics()