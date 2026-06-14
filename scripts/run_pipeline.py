import subprocess
import sys
import time
import os

def run_step(step_name, command):
    print("\n" + "═" * 70)
    print(f" 🚀 PIPELINE STEP: {step_name}")
    print("═" * 70)
    
    start_time = time.perf_counter()
    
    try:
        result = subprocess.run(command, check=True, text=True)
        elapsed = time.perf_counter() - start_time
        print(f"\n[✔] {step_name} completed successfully in {elapsed:.2f} seconds.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[✘] CRITICAL ERROR in {step_name}. Execution aborted.")
        print(f"Error details: The process exit code was {e.returncode}")
        return False

if __name__ == "__main__":
    print("""
    ██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗
    ██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝
    ██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗  
    ██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝  
    ██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗
    ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝
    """)
    print("Run Pipeline - La Ghigliottina\n")
    
    python_exe = sys.executable  

    # =========================================================
    # PHASE 0: CORPORA MATRICES CHECK AND INGESTION
    # =========================================================
    matrix_path = "grafi/cache_grafo_P800K_W300K_S1M_matrix.npz"
    meta_path = "grafi/cache_grafo_P800K_W300K_S1M_meta.pkl"
    
    if not (os.path.exists(matrix_path) and os.path.exists(meta_path)):
        print("[!] Warning: Distributional NPMI Matrices not found in the system.")
        print("[*] Automatically starting the Massive Ingestion module (May take 10-15 minutes)...")
        step0_ok = run_step(
            "PHASE 0: FastSparse Matrix Construction from Corpora", 
            [python_exe, "build_fast_sparse_knowledge.py"]
        )
        if not step0_ok: sys.exit(1)
    else:
        print("[✔] PHASE 0 Skipped: Distributional NPMI Matrices already present and indexed in 'grafi/'.\n")

    # =========================================================
    # PHASE 1: INFERENCE, XGBOOST AND XAI (LLaMA)
    # =========================================================
    step1_ok = run_step(
        "PHASE 1: Hybrid LTR Inference and Explainable AI Generation", 
        [python_exe, "main.py"]
    )
    if not step1_ok: sys.exit(1)

    # =========================================================
    # PHASE 2: METRICS EVALUATION
    # =========================================================
    step2_ok = run_step(
        "PHASE 2: Academic NLG Evaluation and Dashboard Update", 
        [python_exe, "evaluate_metrics.py"]
    )
    if not step2_ok: sys.exit(1)

    # =========================================================
    # PHASE 3: STREAMLIT DASHBOARD DEPLOYMENT
    # =========================================================
    print("\n" + "═" * 70)
    print(" 📊 PIPELINE STEP: Starting Streamlit Dashboard")
    print("═" * 70)
    print("[*] The dashboard will open automatically in the browser.")
    print("[*] To terminate the server and close everything, press CTRL+C in the terminal.\n")
    
    try:
        subprocess.run([python_exe, "-m", "streamlit", "run", "dashboard/dashboard.py"])
    except KeyboardInterrupt:
        print("\n[*] Streamlit server terminated by the user. Goodbye.")