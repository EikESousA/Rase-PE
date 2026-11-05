<p align="center">
  <img src="docs/images/ufs.png" alt="UFS logo" width="10%"/>
  &nbsp;&nbsp;&nbsp;
  <img src="docs/images/procc.png" alt="PROCC logo" width="17%"/>
</p>

<p align="center">
  <img src="docs/tags/release.svg" alt="Release"/>
  <img src="docs/tags/license.svg" alt="License"/>
  <img src="docs/tags/contributors.svg" alt="Contributors"/>
</p>

<p align="center">
  <img src="docs/tags/python.svg" alt="Python"/>
  <img src="docs/tags/plataform.svg" alt="Platform"/>
  <img src="docs/tags/build.svg" alt="Build"/>
</p>

# A Comparative Analysis of Open-Source LLMs for Converting Architecture, Engineering and Construction Standards to the RASE Format

> **Eike Natan Sousa Brito**¹ · **André Carvalho**¹ · **Marco Antônio Brasiel Sampaio**¹
>
> ¹ Universidade Federal de Sergipe (UFS) — Graduate Program in Computer Science (PROCC) — São Cristóvão, Sergipe, Brazil
>
> Corresponding author: `eike.sousa@gmail.com`
>
> Source code: <https://github.com/EikESousA/Rase-PE>

---

## Abstract

The *Building Information Modeling* (BIM) methodology organizes the planning and execution of construction projects around a detailed three-dimensional model, reducing rework and anticipating interferences before construction begins. This model, however, does not by itself solve the verification of design compliance against regulatory technical standards, whose interpretive nature makes automation an open problem. The **RASE** methodology (*Requirement, Applicability, Selection, Exception*) addresses this gap by rewriting each normative excerpt as computable data, allowing BIM *software* to verify compliance automatically; converting standards into this format, however, is still done manually, rule by rule, which becomes a bottleneck for automation.

Natural Language Processing, and **Large Language Models (LLMs)** in particular, emerges as a promising path to automate this conversion, and can be adapted to the task through **Prompt Engineering**, a lightweight technique that adjusts only the instruction sent to the model. This work comparatively evaluates **six open-source LLMs** (Llama, Dolphin, Gemma, Mistral, Alpaca, and Qwen) on the automatic conversion of Brazilian AEC technical standards to the RASE format, using Prompt Engineering exclusively across **three successive extraction levels** (N1: atomic segmentation of normative sentences; N2: RASE operator identification; N3: extraction of verifiable attributes as JSON).

The comparison was carried out over **five experiments** (EN1, EN2, EN1N2, EN3, and EN1N2N3) on a dataset of **76 annotated standards and 150 atomic rules** from Section 6 of NBR 9050, with **thirteen metrics** across five families (lexical, semantic, distance-based, generation-oriented, and classification), with the models served locally through Ollama under an identical dataset, prompt, and hardware protocol. Thirteen metrics are not enough on their own, so the evaluation also includes an error taxonomy, a decision-threshold sensitivity study with three independent label sets, reference-library metrics, four prompt ablations, an external LLM-as-judge, five-seed repetitions, token-cost measurements, agreement with a second expert annotator, and — decisively — **a comparison against non-LLM baselines**.

The answer the evidence supports is **positive on feasibility and negative on advantage**. The method works: open models driven by prompts alone produce RASE markup close to the human reference, reaching semantic similarity above **0.92** in EN3 and a global average of **0.763** for the best model (Llama). But on this dataset, hand-written surface rules match or beat every LLM on most of the task — punctuation-based segmentation ties the best model at N1 (F1 0.83), and splitting the rule at the modal verb beats all six models at N2 (macro F1 0.56 against 0.43). What the LLMs add is localized in the parts that depend on meaning: separating coordinated rules, and naming the `object` and `property` fields at N3. The path the results point to is combining the two approaches, not replacing one with the other.

**Keywords:** Machine Learning · Natural Language Processing · LLM · Prompt Engineering · BIM · RASE

---

## 1. Introduction

Designing and reviewing civil engineering projects is a knowledge-intensive activity: it requires engineers to interpret regulations, reconcile heterogeneous information, and make decisions grounded in expertise. The move from 2D drawings — the sector's historical standard, which hides clashes and omits information — toward the *Building Information Modeling* (BIM) methodology emerged as a response to the recurring rework in civil construction.

BIM organizes project planning and execution around a detailed 3D model and makes it possible to visualize interferences before construction begins. It does not, however, by itself solve the verification of designs against regulatory standards, which in Brazil are issued by ABNT as NBRs (such as **NBR 9050**, on accessibility). These standards are written in natural language and are interpretive in nature, which makes automated compliance checking an open problem.

To make standards machine-checkable, Hjelseth and Nisbet (2011) proposed the **RASE** methodology, which rewrites each normative excerpt as structured data decomposed into four operators:

| Operator | Meaning | Description |
|----------|---------|-------------|
| **R**equirement | *Requirement* | What must be met (usually the imperative "must") |
| **A**pplicability | *Applicability* | Where / to whom the rule applies |
| **S**election | *Selection* | A subset of the applicability |
| **E**xception | *Exception* | Cases in which the rule does not hold |

> **Example.** In the sentence *"the cross slope must be at most 2% for indoor floors"*, the **applicability** is "floors", the **selection** is "indoor", and the **requirement** is "cross slope less than or equal to 2%".

Santos and Sampaio (2021) closed the last link of this chain: given a RASE JSON file, they verify a BIM model automatically. The bottleneck sits immediately upstream — producing that JSON, which they did by hand. Applying RASE manually, operator by operator, requires an expert and does not scale to the volume of standards in the sector. At its core, identifying Requirement, Applicability, Selection, and Exception in a sentence is a **text classification and information-extraction problem** — exactly the kind of task Machine Learning techniques learn from examples. This work attacks that bottleneck by coupling open-source LLMs to the RASE methodology through Prompt Engineering, producing output in the format the existing verification routine already consumes.

---

## 2. Background

### 2.1 Engineering Standards and the RASE Methodology

Technical standards form the regulatory framework of the AEC sector. Manually checking a BIM model with hundreds or thousands of elements subject to standards grows combinatorially and faces three obstacles: (i) the **ambiguity and cross-references** of natural language; (ii) **regional variation**, which limits reuse across jurisdictions; and (iii) the **semantic difficulty** of distinguishing requirement, applicability, and exception within the same passage. The consolidated response in the literature is to structure the standard into a computable schema (such as RASE) before verification.

### 2.2 LLMs and Textual Validation Metrics

**LLMs** combine Machine Learning, deep neural networks, and NLP, adopting the *Transformer* architecture. **Prompt Engineering** is the lightest technique for specializing them on a domain task: it adjusts only the instruction, without changing weights or building a retrieval infrastructure.

Evaluating an LLM's output requires comparing it to a reference while tolerating paraphrases. To this end, this work uses **thirteen metrics across five families**:

- **Lexical** — *FuzzyWuzzy* (Levenshtein distance) and *TF-IDF* (cosine).
- **Semantic** — *SBERT-PT*, *Legal-BERTimbau* (specialized in Portuguese legal text), and a *Multilingual* model.
- **Distance-based** — *Word Mover's Distance* (WMD) over NILC FastText and NILC Word2Vec embeddings.
- **Generation-oriented** — *BERTScore* (contextual token alignment) and *ROUGE-L* (longest common subsequence).
- **Classification** — Accuracy, Precision, Recall, and *F1-score*, which make omissions and hallucinations explicit.

One of this work's methodological findings is that **this choice of metric is not level-neutral**: embedding similarity measures N1 and N2 well, where paraphrase is legitimate, but saturates at N3, where it agrees with an external judge at chance level (κ = 0.009). Evaluating structured attribute extraction requires exact matching and schema validation. Section 5.6 reports the evidence.

---

## 3. RASE-PE: Converting Standards to RASE with Prompt Engineering

The proposed methodology — **RASE-PE** — rests on three decisions: (i) decompose the conversion into three successive levels (N1, N2, and N3); (ii) use Prompt Engineering as the **only** specialization technique; and (iii) systematically compare six open-source LLMs served locally through Ollama, under an identical protocol.

### 3.1 Three Extraction Levels

```
raw standard text
        │
        ▼
 ┌──────────────┐   N1 — Atomic Segmentation
 │      N1      │   Splits the sentence into atomic rules (one rule per item).
 └──────────────┘   Direct prompt.
        │
        ▼
 ┌──────────────┐   N2 — RASE Operator Identification
 │      N2      │   Marks, in each rule, R / A / S / E.
 └──────────────┘   Structured Chain-of-Thought (fixed decision order).
        │
        ▼
 ┌──────────────┐   N3 — Verifiable Attribute Extraction
 │      N3      │   Extracts type, object, property, comparation, target, unit.
 └──────────────┘   Few-Shot (one prompt per operator, or one combined prompt).
        │
        ▼
 structured RASE representation (JSON)
```

### 3.2 Prompt Engineering

All adaptation lives in the *prompts* (`prompts/` folder), organized by level and by operator. Three techniques were applied according to the nature of each level:

- **N1 → direct prompt** — a direct instruction for a structurally simple task (segmentation), economical in tokens.
- **N2 → structured Chain-of-Thought** — a fixed decision order plus a definition of each operator, so the model classifies in a stated sequence instead of guessing all four at once. The prompt asks for four labelled lines and no free-form explanation; a variant that does ask for written reasoning before the answer (*explicit CoT*) is measured as an ablation in Section 5.7.
- **N3 → Few-Shot** — complete input/output examples that anchor the rigid JSON schema, with one prompt per operator plus a combined multi-operator prompt (~4× faster).

Every ablation variant is reproduced verbatim in `prompts/ablacao/`, next to `prompts/n2_direto.txt` and `prompts/n2_cot_explicito.txt`.

### 3.3 Evaluated LLM Models

Six open-source models run locally through Ollama, chosen for complementary profiles of size, alignment, and language. The exact tags live in `config/models.py`:

| Model (`--model`) | Origin | Tag served through Ollama | Parameters |
|-------------------|--------|---------------------------|------------|
| `llama`   | Meta                     | `llama3.1:8b`                                      | 8B |
| `dolphin` | cnmoro (LLaMA-3 PT)      | `cnmoro/llama-3-8b-dolphin-portuguese-v0.3:4_k_m`  | 8B (Q4_K_M) |
| `gemma`   | brunoconterato (PT-BR)   | `brunoconterato/Gemma-3-Gaia-PT-BR-4b-it:f16`      | 4B (FP16) |
| `mistral` | cnmoro (PT)              | `cnmoro/mistral_7b_portuguese:q4_K_M`              | 7B (Q4_K_M) |
| `alpaca`  | splitpierre (Bode PT-BR) | `splitpierre/bode-alpaca-pt-br:13b-Q4_0`           | 13B (Q4_0) |
| `qwen`    | cnmoro (PT)              | `cnmoro/Qwen2.5-0.5B-Portuguese-v1:q4_k_m`         | 0.5B (Q4_K_M) |

### 3.4 Dataset

`dataset.json` (version 3) consolidates **76 annotated standards** from **Section 6 of NBR 9050**, segmented into **150 atomic rules**. It derives from Mendonça's (2020, Poli-USP) manual conversion. The first version of this dataset was built from Appendix A of that work, the consolidated conversion spreadsheet; an audit against **Appendix B** (pp. 103–116), which carries the full text of 81 items with the RASE markup applied directly to the words, showed that the spreadsheet had recorded only the final product and left out much of the available annotation. The revision redid 45 standards from Appendix B, kept 31 whose N1 and N2 already matched it, and dropped 3 duplicates, which took the dataset from 79 standards and 156 rules to 76 and 150 — and, at N3, raised the annotated attributes from 63 to **1,123 filled positions out of 3,000** (37.4%), spread over 132 of the 150 rules. That is what made EN3 measure extraction rather than abstention. The `origem` field of each standard records which operation produced it, and the audit itself is in `results/analises/mendonca_cobertura.csv`. **Every number in Section 5 comes from a single re-execution of the whole battery over this revised dataset**; where the text compares against "the previous version of the dataset", it refers to the pre-revision reference described here.

RASE operators are heavily unbalanced, which matters for reading the results:

| Operator | Occurrences | Share of the 150 rules |
|----------|------------:|-----------------------:|
| Requirement (R)   | 148 | 98.7% |
| Applicability (A) | 116 | 77.3% |
| Selection (S)     |  69 | 46.0% |
| Exception (E)     |  19 | 12.7% |

Each entry holds the reference RASE decomposition across the three levels: raw text (`text`), atomic rules (`texts_n1`), the four operators per rule (`operators_n2`), and, for each operator, the classified excerpt (`text_n2`) and the JSON attributes (`properties_n3`). The same file serves both as input to the generators and as reference for the validators. See [`docs/dataset_schema.md`](docs/dataset_schema.md) for the full schema.

### 3.5 Computational Environment

All runs were carried out on a single workstation — **Intel Core i9-13900K CPU, 32 GB RAM, NVIDIA RTX 4080 GPU, Ubuntu 22.04, Python 3.11** — with the models served by Ollama at temperature 0.1 and the metrics computed with `sentence-transformers`, `gensim`, `fuzzywuzzy`, and `scikit-learn`.

---

## 4. Experiments

The evaluation is organized into five experiments to isolate each level's contribution and measure error propagation along the chain:

| Experiment | Input | Evaluated output | Goal |
|------------|-------|------------------|------|
| **EN1**     | raw text | N1 | Atomic segmentation in isolation |
| **EN2**     | reference N1 | N2 | RASE identification in isolation |
| **EN1N2**   | raw text | N1→N2 | Chained pipeline (N1→N2 error propagation) |
| **EN3**     | reference N2 | N3 | Attribute extraction in isolation |
| **EN1N2N3** | raw text | N1→N2→N3 | Full chained pipeline (production use) |

EN1, EN2, and EN3 measure each level in isolation (receiving the previous level as reference); EN1N2 and EN1N2N3 expose error propagation along the chain. The isolated experiments were additionally repeated with **five seeds (42–46)**, and the whole evaluation is re-run with the **7 standards used as prompt examples removed**, to separate genuine performance from example contamination.

---

## 5. Results and Discussion

> All figures come directly from `metrics/validate_*.json` and `results/analises/*.json`. Values on the normalized scale [0, 1]; the closer to 1, the greater the similarity between output and reference. `results/analises/resumo_capitulo5.json` regenerates every number quoted below.

### 5.1 Per-metric overview

Per-metric averages across the five experiments, aggregated over the six models:

| Metric | EN1 | EN2 | EN1N2 | EN3 | EN1N2N3 |
|--------|:---:|:---:|:-----:|:---:|:-------:|
| *Lexical* — FuzzyWuzzy | 0.489 | 0.620 | 0.508 | **0.856** | 0.651 |
| *Lexical* — TF-IDF | 0.634 | 0.465 | 0.463 | 0.636 | 0.527 |
| *Semantic* — SBERT | 0.798 | 0.582 | 0.628 | **0.924** | 0.817 |
| *Semantic* — BERTimbau | 0.817 | 0.638 | 0.675 | 0.892 | 0.791 |
| *Semantic* — Multilingual | 0.851 | 0.670 | 0.714 | 0.922 | 0.827 |
| *Distance* — WMD (NILC FastText) | 0.711 | 0.636 | 0.632 | 0.852 | 0.775 |
| *Distance* — WMD (NILC Word2Vec) | 0.684 | 0.607 | 0.601 | 0.846 | 0.761 |
| *Generation* — BERTScore | **0.865** | **0.798** | **0.805** | 0.918 | **0.858** |
| *Generation* — ROUGE-L | 0.596 | 0.452 | 0.447 | 0.790 | 0.654 |
| *Classification* — Accuracy | 0.497 | 0.445 | N/A | 0.537 | N/A |
| *Classification* — Precision | 0.570 | 0.305 | N/A | 0.266 | N/A |
| *Classification* — Recall | 0.784 | 0.289 | N/A | 0.212 | N/A |
| *Classification* — F1-score | 0.650 | 0.283 | N/A | 0.216 | N/A |

Three patterns emerge. Embedding-based metrics (SBERT, BERTimbau, Multilingual, and BERTScore) stay consistently higher. Lexical ones (FuzzyWuzzy, TF-IDF, and ROUGE-L) lose ground when moving from EN1 into EN1N2. And the **gap between the two metric families widens with the level**: EN3 reaches 0.92 on semantic similarity while its classification F1 is 0.216 — the same outputs that look near-perfect to an embedding are mostly wrong field by field. Section 5.6 shows the semantic reading is the one that fails here.

### 5.2 Global ranking by model

Global average per model (nine similarity metrics × five experiments):

| Model | EN1 | EN2 | EN1N2 | EN3 | EN1N2N3 | **Global Avg.** |
|-------|:---:|:---:|:-----:|:---:|:-------:|:---------------:|
| **Llama**   | 0.731 | **0.741** | 0.675 | **0.869** | **0.800** | **0.763** 🥇 |
| **Mistral** | 0.759 | 0.725 | **0.683** | 0.853 | 0.784 | **0.761** 🥈 |
| **Dolphin** | **0.792** | 0.646 | 0.660 | 0.852 | 0.748 | **0.740** 🥉 |
| Gemma   | 0.718 | 0.687 | 0.644 | 0.857 | 0.751 | 0.732 |
| Alpaca  | 0.687 | 0.444 | 0.514 | 0.831 | 0.658 | 0.627 |
| Qwen    | 0.610 | 0.403 | 0.472 | 0.829 | 0.699 | 0.603 |

- **Llama** — leads the global average and both N3-level stages, and is the only model to beat the majority-fill baseline at N3. The choice when the computational cost is acceptable.
- **Mistral** — 0.002 behind Llama and the most stable across experiments; it never leads by a wide margin and is never far behind.
- **Dolphin** — the best segmenter at N1 by a clear margin (0.792) and the **fastest full pipeline (3m28s)**, which makes it the best cost-benefit option.
- **Gemma** — the most balanced across the four operators, at the cost of the slowest full pipeline (15m29s).
- **Alpaca** — 13B parameters and the second-worst result: it gets content wrong more often than the 7–8B models.
- **Qwen** — at 0.5B it copied the prompt examples instead of processing the input standard. It is below every baseline at every level.

**The top four are separated by 0.031.** That is the single most useful number in this table: above a minimum size, the formulation of the prompt weighs more than the choice of model (Section 5.7).

### 5.3 Execution time (quality × cost trade-off)

Total generation time per model (seconds), by experiment:

| Model | N1 | N2 | N1N2 | N3 | N1N2N3 |
|-------|---:|---:|-----:|---:|-------:|
| Alpaca  | 133.4 | 229.6 | 251.3 | 485.4 | 798.4 |
| **Dolphin** | 64.0 | **102.9** | **88.9** | 201.5 | **207.7** |
| Gemma   | 119.9 | 136.5 | 218.7 | 449.7 | 928.6 |
| Llama   | 63.4 | 108.7 | 124.9 | **178.2** | 301.0 |
| Mistral | 61.1 | 101.0 | 111.9 | 211.3 | 233.0 |
| Qwen    | **38.7** | 147.1 | 216.6 | 328.3 | 289.4 |

Dolphin and Mistral run the full pipeline in under four minutes; Gemma takes over fifteen. Qwen is the fastest at N1 only because it produces almost nothing usable — its cost reappears at N2 and N3, where it emits 303 and 330 output tokens per call against 50–110 for the other models (`results/analises/robustez.json`).

### 5.4 Non-LLM baselines — the decisive comparison

Every baseline goes through the same validation routine, the same pairing, and the same 0.7 threshold as the models.

| Experiment | Procedure | Accuracy | Precision | Recall | F1 |
|------------|-----------|---------:|----------:|-------:|---:|
| **EN1** | Baseline: copy whole text as one rule | 0.507 | 1.000 | 0.507 | 0.673 |
|         | Baseline: punctuation rules | 0.708 | 0.912 | 0.760 | **0.829** |
|         | Best LLM (Dolphin) | 0.711 | 0.842 | 0.820 | **0.831** |
|         | Worst LLM (Qwen) | 0.238 | 0.321 | 0.480 | 0.385 |
| **EN2** | Baseline: empty output | 0.413 | 0.000 | 0.000 | 0.000 |
|         | Baseline: copy the rule | 0.612 | 0.222 | 0.225 | 0.223 |
|         | Baseline: split at the modal verb | 0.730 | 0.834 | 0.504 | **0.555** |
|         | Best LLM (Gemma) | 0.536 | 0.415 | 0.445 | 0.425 |
|         | Worst LLM (Qwen) | 0.287 | 0.083 | 0.070 | 0.076 |
| **EN3** | Baseline: empty output | 0.626 | 0.000 | 0.000 | 0.000 |
|         | Baseline: regular expressions | 0.613 | 0.260 | 0.359 | **0.302** |
|         | Baseline: most frequent value | 0.464 | 0.220 | 0.346 | 0.260 |
|         | Best LLM (Llama) | 0.580 | 0.338 | 0.311 | 0.296 |
|         | Worst LLM (Qwen) | 0.514 | 0.131 | 0.120 | 0.124 |

Read row by row, this is the result that reframes everything above:

1. **N1 — a tie.** Punctuation-based segmentation reaches F1 0.829; the best model reaches 0.831. Four of the six models do not even beat the trivial "return the whole text as one rule" baseline (0.673).
2. **N2 — the rules win.** Splitting the rule at the modal verb scores macro F1 0.555 against 0.425 for the best model. Per operator, the models clearly beat the trivial procedures only on **applicability** (F1 up to 0.46). On the requirement, their F1 of up to 0.87 is high but still below simply copying the rule (0.89); selection (0.23) and exception (0.19) fail across the board — the models fuse the element with its qualifier into a single applicability excerpt.
3. **N3 — a tie, with a different composition.** Regular expressions and Llama both land at ~0.30. But the rules are better at `comparation`, `target`, and `unit`, while **the models are the only ones that fill `object` and `property`** — the fields that require naming the element and the verifiable attribute.

The honest conclusion: with a reference whose operators are literal excerpts of the text, a simple syntactic split already achieves most of the performance, and only the explicit comparison reveals it. Evaluating LLMs on standards extraction **without a baseline invites overestimating what they do.**

### 5.5 Error taxonomy

`results/analises/taxonomia_erros.json` classifies every pair, rather than averaging it. At N2 (EN1N2, summed over the six models), the dominant failures are **1,164 rules invented at N1** and **247 operator swaps**, while the exception operator is right **3 times in the entire experiment**. At N3 (EN1N2N3), **5,228 hallucinations** fill a field whose reference operator is empty, and 2,395 fill it with a different value — the error moved from "filling where nothing exists" to "filling with the wrong value", concentrated in the `property` field.

Over-segmentation at N1 is the error that propagates furthest, which makes segmentation the most important control point of the method.

### 5.6 Is a 0.7 similarity threshold a valid notion of "correct"?

The classification metrics depend on a 0.7 threshold over Multilingual similarity. Three independent label sets put that choice to the test, by ROC analysis:

| Level | Label set | Pairs | Correct | AUC | Youden threshold |
|-------|-----------|------:|--------:|----:|-----------------:|
| N1 (EN1) | strict | 840 | 98 | 0.989 | 1.00 |
| N1 (EN1) | relaxed | 840 | 579 | 0.933 | 0.86 |
| N2 (EN1N2) | strict | 1,604 | 116 | 0.991 | 0.98 |
| N2 (EN1N2) | relaxed | 1,604 | 577 | 0.955 | 0.70 |
| N1 (all experiments) | manual | 114 | 16 | 0.938 | 0.76 |
| N2 (all experiments) | manual | 150 | 17 | 0.980 | 0.83 |

The three label sets and an **external LLM-as-judge** converge on the same reading: the threshold is **adequate at N2**, **permissive at N1** (the optimum sits between 0.76 and 0.86, so N1 F1 values should be read as a ceiling), and **not valid at N3**, where it does not discriminate at all. At N3 the judge and the threshold agree on only 10–30% of pairs (κ ≈ 0.009, i.e. chance), because the threshold passes nearly everything the judge rejects.

Reference-library metrics (sacreBLEU, RapidFuzz, Pydantic schema validation in `results/analises/metricas_bibliotecas.json`) give the N3 picture the threshold misses: between **70.5% and 92.9%** of generated N3 records are structurally valid, so 7% to 30% are not.

### 5.7 Prompt formulation weighs more than model choice

Four prompt ablations, each changing exactly one factor, with 95% paired bootstrap confidence intervals over the standards (bold = interval excludes zero). Values are the macro-F1 difference against the dissertation's prompt:

| Variant | Llama | Dolphin | Mistral | Gemma | Alpaca | Qwen |
|---------|------:|--------:|--------:|------:|-------:|-----:|
| N2 without the example (*zero-shot*) | −0.021 | **−0.132** | **−0.055** | **−0.117** | **−0.057** | −0.027 |
| N2 with an example external to the dataset | +0.045 | **+0.118** | **+0.081** | −0.022 | **+0.135** | −0.001 |
| N2 with JSON output | −0.002 | **+0.075** | **+0.114** | −0.003 | **+0.117** | **+0.057** |
| N3 with examples in reverse order | −0.020 | **+0.036** | +0.000 | **−0.023** | −0.017 | −0.006 |

And the Chain-of-Thought ablation at N2 (direct prompt × structured CoT × explicit CoT):

| Model | F1 direct | F1 structured | F1 explicit | Swaps direct | Swaps explicit | Time direct | Time explicit |
|-------|----------:|--------------:|------------:|-------------:|---------------:|------------:|--------------:|
| Alpaca  | 0.136 | 0.133 | 0.013 | 126 | 4 | 414 s | 1,254 s |
| Dolphin | 0.280 | 0.341 | **0.354** | 110 | 66 | 279 s | 436 s |
| Gemma   | 0.385 | 0.425 | **0.461** | 130 | 55 | 298 s | 637 s |
| Llama   | 0.350 | 0.370 | **0.437** |  71 | 65 | 392 s | 617 s |
| Mistral | 0.289 | 0.352 | **0.371** |  82 | 58 | 317 s | 449 s |
| Qwen    | 0.047 | 0.076 | **0.130** |  20 | 37 | 314 s | 481 s |

Putting the two tables together: **switching the prompt moves the result more than switching the model.** The top four models are 0.031 apart, while within the same model, explicit reasoning moves N2 macro F1 by up to 0.087 and cuts operator swaps by up to 58%; swapping the demonstration example for one that fills all four operators moves it by up to 0.135; and asking for JSON output, by up to 0.117. Explicit CoT costs 1.6× to 3.2× the generation time, and it destroys Alpaca (0.013), which stops answering in the expected shape.

### 5.8 Stability, contamination and reference reliability

- **Seeds.** Repeating the isolated experiments with five seeds (42–46) gives a sample standard deviation of **at most 0.014** and a range of at most 0.034 — the differences above are well clear of run-to-run noise. Caveat: the same seed did not reproduce outputs bit-for-bit in 4 of the 18 model × level combinations.
- **Contamination.** Seven of the 76 standards are used as prompt examples. Removing them changes N1 F1 by at most 0.030 and N2 macro F1 by at most 0.016. At N2, the ablation with an example external to the dataset went further: swapping the contaminated example **improved** macro F1 significantly in three models and did not significantly worsen it in any.
- **Second annotator.** A second expert re-annotated 40 rules and the 264 ROC-curve pairs, without access to the reference. On the pairs the two annotators agree on **259 of 264 (κ = 0.907)**, so the AUC values in Section 5.6 do not rest on one person's judgement. On the operators, excerpts nearly coincide for requirement (token F1 0.932) and selection (0.883, κ = 0.775), while applicability (0.649) and exception diverge **by convention**: the reference marks only the core element ("corridor"), the second annotator included the article and often the preceding property. Those two divergences coincide exactly with the models' most frequent errors — part of the low score on those operators comes from a convention that the written operator definitions do not fully determine.

### 5.9 Which model properties actually matter

Wilcoxon signed-rank tests over the 76 standards, on per-item hit rate averaged over the five seeds (`results/analises/robustez.json`):

| Hypothesis | Comparison | EN1 | EN2 | EN3 |
|------------|------------|-----|-----|-----|
| Scale | Gemma (4B) × Qwen (0.5B) | 0.553 / 0.277, *p* < 0.001 | 0.631 / 0.317, *p* < 0.001 | 0.617 / 0.581, *p* < 0.001 |
| Scale | Llama (8B) × Alpaca (13B) | 0.542 / 0.615, *p* = 0.067 | 0.647 / 0.311, *p* < 0.001 | 0.623 / 0.577, *p* < 0.001 |
| Scale | Gemma (4B) × Llama (8B) | 0.553 / 0.542, *p* = 0.704 | 0.631 / 0.647, *p* = 0.620 | 0.617 / 0.623, *p* = 0.565 |
| PT adaptation | Llama × Dolphin | 0.542 / 0.788, *p* < 0.001 | 0.647 / 0.556, *p* < 0.001 | 0.623 / 0.615, *p* = 0.242 |
| Architecture | Dolphin × Mistral | 0.788 / 0.709, *p* = 0.010 | 0.556 / 0.630, *p* = 0.004 | 0.615 / 0.612, *p* = 0.554 |

Three delimitations follow. **There is a minimum size, between 0.5B and 4B** — Qwen fails everywhere, while Gemma (4B) does not differ from Llama (8B) at any level, and Alpaca (13B) does not beat the 7–8B models. **Portuguese adaptation helps at N1 and not beyond** — Dolphin dominates segmentation and loses at N2. **No 7–8B architecture dominates the others across all three levels.**

### 5.10 Integrated discussion

1. **The method is feasible but not advantageous on this dataset.** It produces RASE markup close to the reference using open models and prompts alone, but on the isolated levels it does not beat hand-written surface rules.
2. **The metric must match the level.** Embedding similarity is right for N1 and N2, where paraphrase is legitimate, and wrong for N3, where it agrees with an external judge at chance level. Structured attribute extraction needs exact matching and schema validation.
3. **Prompt formulation outweighs model choice** above a minimum model size — a result worth more practically than any ranking of the six models.
4. **Baselines are not optional.** Without them, this work would have reported "semantic similarity above 0.9" as a success and missed the fact that a regular expression does the same at N3.
5. **The value of the LLMs is localized**: separating coordinated rules at N1, and naming `object` and `property` at N3. That argues for a **hybrid pipeline** — rules for the surface part, the LLM for what the rules cannot cover — not for replacing either.

---

## 6. Conclusion

Decomposing the conversion of standards into three successive levels (N1, N2, N3), supported solely by Prompt Engineering, is sufficient to produce valid RASE representations from the raw standard text with locally served open-weight models. The best model (Llama, global average 0.763) is 0.031 ahead of the fourth; Dolphin is the best segmenter and the fastest full pipeline, Mistral the most consistent, Gemma the most balanced across operators, while Alpaca (13B) and Qwen (0.5B) are not recommended.

The answer to "was the method validated?" is **positive on feasibility and negative on advantage**. N1 works for the two best models (F1 0.83 and 0.78, against 0.67 for the trivial procedure), and at N2 only applicability clearly beats the trivial procedures. At N3, now measurable thanks to the revised dataset, the models beat empty-filling but only Llama beats majority-filling, and regular expressions tie with it overall — while being the only approach that does **not** fill `object` and `property`. Between 70.5% and 92.9% of generated N3 records are structurally valid, so any practical use requires human review or schema validation with discard.

The open challenges, in the order in which they affect the method's validity: **combine rules and LLMs** into a hybrid pipeline; **redesign the N2 prompt for Selection and Exception**, where the ablations already point the way (an example filling all four operators, JSON output, explicit reasoning); **improve N3 attribute extraction** with a controlled property vocabulary and generation-time schema validation; **control over-segmentation at N1**, the error that propagates furthest; and **separate examples from the evaluation set** for N1 and N3, eliminating the residual contamination instead of only quantifying it.

### Limitations

The results hold under the conditions in which the method was validated. The evaluation used Section 6 of NBR 9050 only, inheriting Mendonça's (2020) scope — performance on other sections, other NBRs, and other normative domains was not measured. The baselines that tie with or beat the models were written with knowledge of the reference's annotation convention and tested only on that same section; whether they survive another standard or another convention is unknown, and the same caveat applies to the prompts. The reference annotation comes from a single expert; the 40-rule comparison with a second expert indicates the performance ceiling on applicability and exception but does not fix it. The ROC label set is a stratified sample of 264 pairs, enough to confirm the order of magnitude of the optimal threshold but not to pin it down. The five-seed repetition covers EN1, EN2 and EN3 but not the chained pipelines, and only temperature 0.1. Finally, the N3 JSON was aligned to the format of the Santos and Sampaio (2021) verification routine, but the actual coupling was never executed — the end-to-end validation on a real project remains open.

---

## Reproducing the experiment

### Requirements

- **Python 3.11+** (3.12 tested). Should work on Windows, Linux, or macOS.
- **[Ollama](https://ollama.com/download)** installed and running (`ollama serve`); models are downloaded automatically by the menu.
- The WMD metric uses `pot` (Python Optimal Transport), already in `requirements.txt`; the NILC FastText/Word2Vec (PT) weights are downloaded on demand from Hugging Face.
- The complementary analyses in `analyses/` need `requirements-avaliacao.txt` (sacreBLEU, RapidFuzz, Pydantic, SciPy). They do **not** need Ollama or a GPU.

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternative with [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

### Running

```bash
ollama serve          # make sure the Ollama server is up
python3 main.py       # main menu: generate data / validate / build tables
```

In the menu, choose **Generate data**, select the level (`n1`/`n2`/`n3`/`n1n2`/`n1n2n3`) and the model. Each stage can also be run directly:

```bash
# Generation
python generates/generate_n1.py      --model mistral
python generates/generate_n2.py      --model mistral
python generates/generate_n3.py      --model mistral   # combined multi-operator prompt (~4x faster)
python generates/generate_n1n2n3.py  --model mistral   # full pipeline

# Validation (compares against dataset.json and writes to metrics/)
python validates/validate_n1.py
python validates/validate_n2.py
python validates/validate_n3.py
python validates/validate_n1n2n3.py

# Quick end-to-end test with the first dataset item
python test.py
```

> To revert N3 to *legacy* mode (4 calls, one per operator): `N3_LEGACY=1 python generates/generate_n3.py --model mistral`.

[`run.md`](run.md) is the full operational runbook (in Portuguese): expected runtimes, checkpoint resume, how to regenerate every analysis, and troubleshooting.

### Complementary analyses (no Ollama required)

```bash
pip install -r requirements-avaliacao.txt

python -m analyses.error_taxonomy        # results/analises/taxonomia_erros.json
python -m analyses.threshold_analysis    # ROC curves and threshold sweep
python -m analyses.library_metrics       # sacreBLEU / RapidFuzz / schema validation
python -m analyses.baselines             # non-LLM baselines (Section 5.4)
python -m analyses.robustez              # seeds, prompt ablations, token cost, Wilcoxon
python -m analyses.cot_ablation          # Chain-of-Thought ablation at N2
python -m analyses.concordancia calcular # agreement with the second annotator
python -m analyses.llm_judge             # external LLM-as-judge
python -m analyses.report_tables         # regenerates every results table
```

Add `--sem-contaminacao` to the first three to exclude the 7 standards used as prompt examples.

> Note: `analyses/report_tables.py`, `analyses/robustez.py`, `tools/chapter_tables.py` and the `tools/plot_*.py` scripts write their LaTeX tables and figures into a `dissertacao/` tree, which holds the dissertation text and is not part of this repository. They create the directory and run fine here; the `.tex` and `.png` files simply land in an otherwise empty folder.

### Code structure

| Path | Responsibility |
|------|----------------|
| `main.py` | Main menu (generate / validate / tables) |
| `config/models.py` | Central registry of Ollama models (names + tags) |
| `prompts/` | Templates `n1.txt`, `n2.txt`, `n3_*.txt` (per operator), `n3_combined.txt`, the CoT variants, and `ablacao/` |
| `dataset.json` | 76 annotated standards / 150 atomic rules (input and reference) |
| `generates/` | One generation module per stage |
| `validates/` | One validation module per stage |
| `utils/validates/run_validation.py` | Single orchestrator of the validations |
| `analyses/` | Complementary analyses (taxonomy, ROC, baselines, ablations, judge, agreement) |
| `predicts/` | Generated outputs (`generate_<n>_<model>.json`), plus `ablacao_cot/` and `robustez/` (5 seeds, prompt ablations, token usage) |
| `predicts_baselines/` | Outputs of the non-LLM baselines |
| `metrics/` | Validation metrics (`validate_<n>.json`), `details/` per-pair records, and `baselines/` |
| `results/analises/` | Every analysis result quoted in Section 5, including `resumo_capitulo5.json` |
| `tools/` | LaTeX/CSV table generation, plots, baselines, and seed sweep |

### Environment variables (defaults already optimized)

| Variable | Effect |
|----------|--------|
| `GENERATE_DEBUG=0` | Turns off logs in `logs/` (on by default) |
| `GEN_SEED=42` | Deterministic Ollama seed (`none` disables it) |
| `GEN_RESUME=0` | Turns off checkpoint resume |
| `N3_LEGACY=1` | N3 with 4 calls (one per operator) |
| `GEN_TIMEOUT=600` | Timeout (s) per LLM call |
| `VALIDATE_BERTSCORE=0` / `VALIDATE_ROUGE=0` | Disable the respective metrics (on by default) |
| `VALIDATE_HUNGARIAN=0` | Pair by index instead of the Hungarian algorithm over similarity |
| `SBERT_MODEL=<repo>` | Overrides the default SBERT (`tgsc/sentence-transformer-ult5-pt-small`) |
| `OLLAMA_HOST` / `OLLAMA_HOSTS` | Ollama address(es) (default `http://localhost:11434`) |
| `HF_TOKEN` | Hugging Face token (optional, avoids rate-limit warnings) |

### Docker

```bash
docker compose up --build
```

Brings up an `ollama` container and the app with volumes for the weights. See `docker-compose.yml`.

---

## Help

Questions, bug reports, or feature requests: **eike.sousa@gmail.com**. Please follow our **[Code of Conduct](CODE_OF_CONDUCT.md)**.

## License

Licensed under **CC0-1.0**. See **[LICENSE.md](LICENSE.md)** for details.
