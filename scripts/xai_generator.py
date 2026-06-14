import os
import json
import time
import asyncio
from dotenv import load_dotenv

# Load hidden variables from .env file
load_dotenv()

class ExplainableAIGenerator:
    def __init__(self, api_key=None):
        self.is_available = False
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        # System prompt translated to English, but strictly commanding Italian output
        self.system_prompt = (
            "You are a linguistic assistant specialized in the Italian game 'La Ghigliottina'. "
            "Your only task is to generate the description of the semantic connections between the 5 hint words and the solution. "
            "You must STRICTLY follow this exact schema, without any introduction or conclusion, writing in Italian:\n"
            "[Capitalized Solution]: [hint1] [connection with solution]; [hint2] [connection with solution]; "
            "[hint3] [connection with solution]; [hint4] [connection with solution]; [hint5] [connection with solution]."
        )
        
        # Few-shot examples MUST remain in Italian so the LLM understands the exact formatting required
        self.few_shot_example = (
            "Indizi: calcio, stato, vivere, tariffa, voto\n"
            "Soluzione: estero\n"
            "Risposta: Estero: calcio estero, ovvero campionati e squadre di calcio estere; Stato estero, ovvero un altro Paese del mondo; vivere all'estero, ovvero trasferirsi e vivere in un altro Paese; tariffa estero, ovvero i costi per effettuare chiamate e mandare messaggi in Paesi esteri; voto all'estero, ovvero il voto degli abitanti di un Paese non residenti in esso."
        )
        
        if self.api_key:
            try:
                from groq import AsyncGroq
                self.client = AsyncGroq(api_key=self.api_key)
                self.model_name = "llama-3.3-70b-versatile"
                self.is_available = True
                print(f"\n[*] Cloud XAI Module Activated. End-point: Async Groq API ({self.model_name})")
            except ImportError:
                print("\n[!] 'groq' library not installed in venv. Run: pip install groq")
        else:
            print("\n[!] GROQ_API_KEY not found. XAI Module disabled.")

    async def _fetch_description(self, item, delay_seconds=1.5):
        """
        Asynchronous task with Throttle:
        1. Executes a mandatory pause before starting.
        2. Sends the request.
        """
        await asyncio.sleep(delay_seconds) # Micro-pause to mitigate Rate Limit
        
        hints = [item[f'hint{i}'] for i in range(1, 6)]
        solution = item['sol']
        hints_str = ", ".join(hints)
        user_content = f"Indizi: {hints_str}\nSoluzione: {solution}\nRisposta:"
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{self.few_shot_example}\n{user_content}"}
                ],
                temperature=0.1,  
                max_tokens=250
            )
            description = completion.choices[0].message.content.strip()
        except Exception as e:
            description = f"{solution.capitalize()}: API Error ({str(e)})"
            
        item['desc'] = description
        item['ttg'] = False
        return item

    def enrich_test_set(self, test_data, output_path="output/test_enriched.json"):
        if not self.is_available: 
            print("[!] Cloud generation aborted: API Client or keys not configured.")
            return
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # =====================================================================
        # SMART CACHING LOGIC (CHECKPOINTING)
        # =====================================================================
        existing_data = []
        processed_ids = set()
        
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            if 'desc' in item and "API Error" not in item['desc'] and "Errore" not in item['desc']:
                                existing_data.append(item)
                                processed_ids.add(item['id'])
            except Exception as e:
                print(f"[!] Error reading cache: {e}. Files will be overwritten.")
                existing_data = []
                processed_ids = set()

        items_to_process = [item for item in test_data if item['id'] not in processed_ids]
        
        if not items_to_process:
            print(f"\n[✔] XAI CACHE HIT: All {len(test_data)} instances are already present and valid in '{output_path}'.")
            print("[*] Skipping API generation to save time and resources.")
            return

        if processed_ids:
            print(f"\n[XAI] Resuming from checkpoint: {len(processed_ids)} instances already in cache.")
            print(f"[XAI] Starting Throttled semantic enrichment on the remaining {len(items_to_process)} instances...")
        else:
            print(f"\n[XAI] Starting Throttled semantic enrichment on {len(items_to_process)} instances...")
            
        start_time = time.perf_counter()
        
        # --- SEQUENTIAL ASYNCHRONOUS THROTTLING LOGIC ---
        async def run_pipeline():
            enriched_dataset = existing_data.copy()
            
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(items_to_process), desc="Throttled LPU Queries", unit=" instance")
            except ImportError:
                pbar = None

            for idx, item in enumerate(items_to_process):
                result = await self._fetch_description(item, delay_seconds=1.5)
                enriched_dataset.append(result)
                
                if pbar: pbar.update(1)
                
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    enriched_dataset.sort(key=lambda x: x['id'])
                    for line in enriched_dataset:
                        f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
                        
            if pbar: pbar.close()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_pipeline())
        loop.close()
        
        elapsed = (time.perf_counter() - start_time) / 60
        print(f"[XAI] Asynchronous processing completed in {elapsed:.2f} minutes.")
        print(f"[XAI] Enriched dataset saved to: {output_path}")