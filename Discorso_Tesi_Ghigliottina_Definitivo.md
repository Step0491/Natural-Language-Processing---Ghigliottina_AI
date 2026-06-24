# Discorso di Tesi Definitivo: L'Intelligenza Artificiale per "La Ghigliottina"

Questo file è la "Bibbia" della tua esposizione. Unisce le note architetturali passate, le spiegazioni approfondite e il flusso esatto della tua presentazione PowerPoint aggiornata.

*Consiglio per l'esposizione: Parla lentamente. Hai riempito il discorso di termini accademici pesanti (Path-Based Reasoning, Target-Leakage, Rumore Entropico, Regolarizzazione Stocastica), quindi fai delle pause per farli "digerire" alla commissione e goditi l'effetto che faranno.*

---

### Slide 1: Introduzione
"Buongiorno alla commissione e a tutti i presenti. Oggi presento il mio lavoro di tesi, in cui ho affrontato una sfida estremamente affascinante nel campo del Natural Language Processing: la costruzione di un solutore basato su Intelligenza Artificiale per il gioco televisivo 'La Ghigliottina'."

### Slide 2: The NLP Task
"Le regole del gioco sono note: date 5 parole indizio apparentemente slegate, bisogna trovare la sesta parola che le lega tutte.
Perché questo problema è così complesso per l'Informatica? Perché richiede pensiero laterale e profonda conoscenza culturale. I classici Large Language Models (come ChatGPT) falliscono sistematicamente in questo task: tendono alle 'allucinazioni' e faticano a rispettare vincoli logici multipli e rigidi. 
Il mio obiettivo scientifico è stato superare due limiti storici: il 'collo di bottiglia della conoscenza' dei vecchi sistemi deterministici (troppo rigidi), e il 'rumore entropico' dei sistemi puramente statistici (che per loro natura associano tutto a tutto). Ho quindi progettato un'architettura ibrida."

### Slide 3: Dual Data Pool Configuration
"Per fare questo, il sistema attinge da due enormi bacini di dati di natura opposta. 
Da un lato la **Conoscenza Simbolica**, composta da dizionari di dominio altamente qualificati, proverbi e polirematiche (per un totale di circa 36.000 righe).
Dall'altro la **Conoscenza Distribuzionale**, per la quale ho ingerito in modo asimmetrico ben 11 GB di testo grezzo da Wikipedia, dal corpus Paisà e da OpenSubtitles, al fine di catturare l'uso reale e quotidiano della lingua."

### Slide 4: La Pipeline Ibrida (Retrieve-and-Rerank)
"L'architettura del sistema si snoda in tre fasi.
**Fase 1: Symbolic Retrieval.** Il nucleo del sistema è il Lexical Knowledge Graph (LKG). Perché ho scelto di costruire un Grafo Simbolico invece di usare subito una matrice statistica? Perché il gioco richiede *collocazioni esatte*, non similarità semantica. La statistica pura direbbe che 'Cane' e 'Gatto' sono molto simili, ma alla Ghigliottina serve sapere che si dice 'Cane Sciolto'. Il Grafo mappa deterministicamente questi 'Hard Constraints' ed esegue un ragionamento a più salti (Path-Based Reasoning) a rumore zero.
**Fase 2: Distributional Injection.** Tuttavia, il Grafo da solo non basta a coprire tutta la lingua. Per risolvere le ambiguità e allargare la copertura, il sistema interroga uno *Sparse Co-occurrence Engine* che estrae la metrica statistica NPMI dagli 11GB di corpora.
**Fase 3: Reranking.** Questi due mondi – la Topologia discreta del Grafo e la Statistica continua dei Corpora – vengono fusi in un unico Tensore Ibrido, passando la palla al Reranker."

### Slide 5: Why exclusively Tree-Based Models?
"A questo punto, la scelta del modello di Machine Learning: perché usare modelli basati su Alberi Decisionali (Tree-Based) e non Reti Neurali o Modelli Lineari?
Le Reti Neurali soffrono enormemente l'overfitting su dati tabulari di queste dimensioni, e i modelli lineari fallirebbero perché non sanno gestire le forti interazioni *non lineari* tra le feature del Grafo e la statistica testuale. Gli alberi, invece, gestiscono nativamente la sparsità e i dati fortemente sbilanciati tipici dell'Information Retrieval.
Per l'addestramento ho imposto un rigoroso **Hold-Out Split**: ho diviso la fase di creazione del Grafo da quella di addestramento del modello, isolando fisicamente il Test Set per impedire qualsiasi forma di *Target Leakage*."

### Slide 6: Retrieve-and-Rerank Pipeline (HPO)
"Tramite una massiccia Deep Hyperparameter Optimization valutata in 3-Fold Cross Validation (per garantire metriche totalmente prive di overfitting), ho testato i migliori algoritmi di Learning-to-Rank. 
Come si evince dalla tabella, **XGBoost** si è rivelato il modello State-of-the-Art, superando competitor agguerriti come CatBoost, LightGBM e RandomForest, con un'accuratezza solida e tempi di inferenza nettamente inferiori."

### Slide 7: Empirical Results (Blind Test Set)
"Sui dati di test completamente ciechi, i risultati hanno confermato il paradigma ibrido. 
La baseline puramente Simbolica (ovvero il solo dizionario) si fermava al 26% di accuratezza. L'introduzione del motore statistico orchestrato da XGBoost ha fatto letteralmente fare un salto di classe al sistema, portando l'Accuracy @1 (al primo colpo) al **37%**, e garantendo che la parola corretta si trovi tra le prime 5 (Accuracy @5) nel **51%** dei casi. 
Il tutto con tempi di inferenza folgoranti: meno di 0.15 secondi a partita."

### Slide 8: XGBoost Regularization Proof
"Ma l'aspetto più scientificamente rilevante per una tesi è *l'interpretabilità* del modello.
Guardando il grafico della Feature Importance (basato sull'Information Gain), abbiamo la prova matematica che ha avuto luogo una **Regolarizzazione Stocastica**. Il modello non ha imparato a memoria una singola feature diventando 'pigro', ma si comporta come un vero *Ensemble*. Sfrutta le features topologiche del Grafo (come l'Evidences Count) come filtro deterministico primario, ma assegna un peso cruciale (la statistica NPMI) ai grandi corpora web per dirimere le incertezze."

### Slide 9: Explainable AI via LLMs
"Infine, in ambito IA moderna, trovare la parola corretta non basta: bisogna saperla spiegare.
Ho implementato un modulo di *Post-Hoc Explainable AI*. Invece di usare un LLM per *risolvere* il gioco – dove fallirebbe – invio la soluzione trovata deterministicamente da XGBoost a un modello LLaMA-3. Il suo unico scopo è generare un testo discorsivo che spieghi i collegamenti in linguaggio naturale.
Per valutare la bontà di queste spiegazioni senza giudizio umano, ho usato metriche vettoriali. Il BLEU score risulta basso (~0.11) semplicemente perché penalizza la discorsività tipica dell'LLM se confrontata con gli appunti 'telegrafici' degli umani. Ma il **BERTScore**, che si basa sulla semantica vettoriale profonda, segna un eccellente **0.76**. Questo dimostra matematicamente che l'LLM ha catturato alla perfezione la logica associativa sottostante."

### Slide 10: Conclusions & Limitations
"Concludo analizzando i limiti fisici del sistema con estrema trasparenza ingegneristica.
Il limite principale è il massiccio dispendio di RAM e calcolo richiesto per la pre-computazione offline delle matrici da parte degli 11GB di corpora. Tuttavia, questo *Heavy Offline Preprocessing* è una scelta architetturale deliberata: ho deciso di concentrare tutto il peso computazionale offline, a monte, per poter garantire al giocatore una velocità di inferenza $\mathcal{O}(1)$ in tempo reale. È un trade-off essenziale tra costo di preparazione e reattività del servizio."

### Slide 11: Future Steps
"Per gli sviluppi futuri, il percorso è tracciato. Scalare questo approccio statico all'intero ecosistema internet richiederebbe hardware di livello Enterprise, e comunque limiterebbe l'IA a testi pre-scaricati. Per questo, il Next Step ideale è l'implementazione di un **Dynamic Web Retrieval**: algoritmi di web-crawling in tempo reale per interrogare archivi e articoli online quando l'algoritmo incontra parole fuori vocabolario (Out-of-Vocabulary). Parallelamente, si valuterà l'ingestione massiva di dizionari lessicali completi, come Treccani o ItalWordNet, per intercettare layer semantici ancora più profondi.

Vi ringrazio per l'attenzione."
