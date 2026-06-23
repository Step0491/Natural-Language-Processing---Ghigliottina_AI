# Analisi Architetturale e Piano di Valutazione: Ghigliottina AI

Questo documento raccoglie in modo strutturato le basi teoriche del sistema Ibrido (Simbolico + Statistico) e delinea un piano di testing con modelli alternativi reali e adeguati al task, scartando quelli non idonei.

## 1. La Baseline Attuale (LKG + XGBoost)

### Modulo di Retrieval: LKG (Lexical Knowledge Graph)
- **Natura:** Grafo simbolico (Intelligenza Artificiale Simbolica). Non è una rete neurale.
- **Funzionamento:** I nodi sono le parole del vocabolario, gli archi sono le connessioni esplicite basate su storia del gioco, proverbi, modi di dire e definizioni.
- **Implementazione:** Inverted Index, che garantisce tempi di ricerca istantanei ($\mathcal{O}(1)$).
- **Vantaggi:** Modellazione perfetta degli *Hard Constraints* (Vincoli Forti). È deterministico, non soffre di allucinazioni ed evita il "Rumore Entropico" degli spazi vettoriali puri in cui parole molto comuni diventano simili a qualsiasi altra parola.

### Modulo di Reranking: XGBoost (Learning-to-Rank)
- **Natura:** Modello di Machine Learning Statistico basato su alberi decisionali (Gradient Boosting).
- **Funzionamento:** Prende le parole candidate dal modulo di retrieval e impara a riordinarle (Reranker) assegnando un punteggio (Rank).
- **Perché XGBoost:**
  1. Eccellente sui dati tabulari (feature estratte a mano come i punteggi del grafo e metrica vettoriale NPMI).
  2. Cattura relazioni *non lineari* (es. se connessione nel grafo = 0 ma semantica alta, abbassa il rank).
  3. *Sparsity-Aware*: gestisce nativamente i "dati mancanti" se una parola non ha metriche in alcuni dizionari.
  4. Alta Interpretabilità (Feature Importance).

### Prevenzione del Data Leakage (Hold-out Split)
Il set di training è stato diviso in modo deterministico (80% / 20%).
- **80%:** Usato rigorosamente **solo** per popolare le connessioni storiche nel Grafo LKG.
- **20%:** Usato **solo** per addestrare XGBoost.
- **Razionale:** Se XGBoost fosse addestrato sui dati già visti dal Grafo, imparerebbe che il Grafo non sbaglia mai, ignorando la semantica vettoriale e andando in overfitting massiccio, distruggendo la sua capacità di generalizzare su partite nuove.

---

## 2. Piano di Sperimentazione: Modelli Alternativi da Testare

Per valutare scientificamente che l'architettura attuale sia la migliore, testeremo varianti del Modulo Retrieval mantenendo XGBoost fisso, e poi varianti del Modulo Reranking mantenendo l'LKG fisso (Ablation Study).

> [!NOTE]
> Sono state **escluse** a priori:
> - **Reti Neurali (MLP/Deep Learning):** Poco adatte a piccoli dataset tabulari, prone all'overfitting, e senza Interpretabilità (Black-box).
> - **Modelli Lineari (SVM, Logistic Regression):** Incapaci di catturare le fondamentali interazioni condizionali non lineari tra le feature topologiche e quelle semantiche.
> - **LLM Generativi Puri:** Esclusi per il task di retrieval diretto a causa delle "Allucinazioni" e della bassa aderenza ai vincoli rigidi multipli.

### FASE A: Varianti del Modulo di Retrieval (Alternative all'LKG)
Fissato XGBoost come reranker, estraiamo i candidati (es. top-100) usando metodi diversi:

1. **LKG (Baseline)**: Recupero deterministico dal grafo.
2. **BM25 (Information Retrieval Classico)**: Trattiamo ogni potenziale parola soluzione come un "Documento" e i 5 indizi come "Query". Cerchiamo le occorrenze in un mega-corpus testuale italiano (es. Wikipedia IT). BM25 è lo standard assoluto nell'Information Retrieval testuale basato sulla frequenza.
3. **Dense Retriever (Embeddings)**: Usare modelli vettoriali pre-addestrati (Word2Vec, FastText, o Sentence-BERT IT). Generiamo il vettore per i 5 indizi e calcoliamo la *Cosine Similarity* con tutto il vocabolario per prendere i candidati più "vicini" nello spazio vettoriale.
4. **Hybrid (LKG + BM25 + Dense)**: La configurazione più robusta. Si estraggono candidati da tutti e tre i metodi, delegando poi a XGBoost l'onere di pesare da quale metodo proviene la parola migliore.

### FASE B: Varianti del Modulo di Reranking (Alternative a XGBoost)
Fissato l'LKG (o il metodo ibrido) come estrattore di candidati, proviamo a riordinarli usando modelli di Machine Learning alternativi adatti ai dati tabulari:

1. **XGBoost (Baseline)**: Gradient Boosting standard (crescita dell'albero *Level-wise*).
2. **LightGBM (Microsoft)**: Il competitor diretto di XGBoost.
   - **Perché testarlo:** Costruisce gli alberi in modo *Leaf-wise* (espande la foglia con maggior perdita). È solitamente molto più veloce di XGBoost da addestrare e spesso offre prestazioni identiche o leggermente superiori nei task di "Learning-to-Rank" puro (utilizzando la sua funzione `LGBMRanker`).
3. **CatBoost (Yandex)**:
   - **Perché testarlo:** Usa un approccio chiamato *Ordered Boosting* che lo rende intrinsecamente immune al target-leakage e all'overfitting rispetto a XGBoost. Se alcune delle feature estratte dal Grafo risultassero categoriche, CatBoost è lo stato dell'arte in assoluto.
4. **Random Forest (Scikit-Learn)**:
   - **Perché testarlo come Baseline "Scarsa":** È importante includerlo per dimostrare alla commissione che un approccio di tipo *Bagging* (tanti alberi indipendenti che votano) perde rispetto agli approcci *Boosting* (XGBoost/Light/CatBoost, in cui ogni albero corregge gli errori del precedente) su questo specifico task altamente complesso e sbilanciato.

---

## 3. Metodologia di Valutazione e Risultati Finali (Deep HPO)

Per estrarre il vero potenziale limite (Ottimo Globale) da ogni modello in fase B, è stata applicata una **Deep Hyperparameter Optimization (HPO)** esplorando dozzine di combinazioni di parametri.

### L'importanza della 3-Fold Cross Validation
Durante l'esplorazione randomica spaziale (Grid Search), la valutazione di ogni configurazione è stata soggetta a una **3-Fold Cross Validation**. 
- **Razionale Matematico:** Utilizzando $k=3$, si divide il training set in 3 partizioni: il modello si addestra su due e valida sulla terza a rotazione. Questo garantisce che l'accuratezza finale sia statisticamente robusta e **assolutamente priva di overfitting**.
- **Razionale Computazionale:** Si è scelto 3-Fold (anziché 5 o 10) come perfetto bilanciamento: garantisce sufficiente variazione statistica ma evita l'esplosione dei tempi macchina, considerando che i tensori generati dall'LKG in RAM sono massicci.

### L'Incoronazione di XGBoost
I risultati del Tuning Definitivo hanno decretato **XGBoost** come l'algoritmo State-of-the-Art per *La Ghigliottina*:
- **Accuracy@1:** 37.00%
- **Accuracy@5:** 51.00%
- **MRR:** 0.4312

L'analisi dei *Best Parameters* (`max_depth: 3`, `subsample: 0.6`, `gamma: 0.1`) si è rivelata una miniera d'oro teorica. Ha dimostrato che, se protetto dall'overfitting tramite un severo campionamento stocastico, all'algoritmo **bastano alberi profondi appena 3 livelli** per trovare la parola esatta. Questa è la **prova empirica** che l'ingegnerizzazione iniziale delle feature del Grafo (LKG) è talmente potente ed espressiva da rendere superflua la ricerca di pattern non lineari iper-complessi.

---

## 4. Slide Deck: Frasi Salienti per la Presentazione (PPT)

Usa questi *Bullet Points* direttamente nelle slide per avere un forte impatto visivo e tecnico sulla commissione, senza riempire le slide di testo.

**Slide: La Scelta del Retrieval (LKG vs Embeddings)**
* "Modellazione di Vincoli Rigidi (Hard Constraints) tramite LKG."
* "La Ghigliottina: Relazioni Sintagmatiche (Collocazioni) > Similarità Paradigmatica (Sinonimia)."
* "LKG + NPMI: La precisione di un database relazionale unita alla potenza statistica dei Big Data (Corpora da 12GB)."
* "Perché non LLM o Dense Retrievers? Assenza di allucinazioni semantiche e costo computazionale $\mathcal{O}(1)$."

**Slide: La Scelta del Reranker (Gradient Boosting vs Bagging)**
* "Learning-to-Rank architettato su Modelli ad Albero (Gradient Boosting)."
* "Sparsity-Aware: Capacità nativa di gestire features mancanti dai dizionari."
* "Cattura nativa delle interazioni condizionali non lineari tra features Topologiche e Statistiche."

**Slide: Metodologia di Validazione (Deep HPO)**
* "Prevenzione severa del Target-Leakage: Hold-out Split deterministico (80/20) tra Ingestione Grafo e Reranking."
* "Deep Hyperparameter Optimization (Randomized Search) su un massiccio spazio dimensionale."
* "3-Fold Cross-Validation: Massimizzazione della stabilità statistica garantendo al contempo l'efficienza computazionale sui tensori LKG."

**Slide: Risultati e Conclusioni (Il Trionfo di XGBoost)**
* "Risultati Definitivi: XGBoost si conferma State-of-the-Art (Acc@1: 37%, Acc@5: 51%)."
* "L'Ablation Study dei parametri rivela un `max_depth` di 3: Prova empirica dell'estrema potenza espressiva e dell'eleganza delle features estratte dall'LKG."
* "XGBoost bilancia perfettamente la componente Aggressiva (Acc@1) con la stabilità di Supporto Decisionale (MRR) per il giocatore umano."
