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

Natural Language Processing, and **Large Language Models (LLMs)** in particular, emerges as a promising path to automate this conversion, and can be adapted to the task through **Prompt Engineering**, a lightweight technique that adjusts only the instruction sent to the model. This work comparatively evaluates **six open-source LLMs** (Llama, Dolphin, Gemma, Mistral, Alpaca, and Qwen) on the automatic conversion of Brazilian AEC technical standards to the RASE format, using Prompt Engineering exclusively across **three successive normalization levels** (N1: atomic segmentation of normative sentences; N2: RASE operator identification; N3: semantic JSON structuring).

The comparison was carried out over **five experiments** (EN1, EN2, EN1N2, EN3, and EN1N2N3) on a dataset of **79 annotated NBR 9050 standards**, with **thirteen metrics** across five families (lexical, semantic, distance-based, generation-oriented, and classification), with the models served locally through Ollama under an identical dataset, prompt, and hardware protocol. The results show that the three-level decomposition, relying solely on Prompt Engineering, is sufficient to produce valid RASE representations, reaching semantic similarity close to **0.90** in the best scenarios; **Llama** achieved the best overall average (**0.730**), leading the N3-level stages (EN3 and EN1N2N3), while **Dolphin** offered the best cost-benefit ratio, around ninety times faster in the chained pipeline.

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

The bottleneck lies in the conversion: applying RASE manually, operator by operator, requires an expert and does not scale to the volume of standards in the sector. At its core, identifying Requirement, Applicability, Selection, and Exception in a sentence is a **text classification and structuring problem** — exactly the kind of task that Machine Learning techniques learn from examples. This work tackles that bottleneck by coupling open-source LLMs to the RASE methodology through Prompt Engineering.

---

## 2. Background

### 2.1 Engineering Standards and the RASE Methodology

Technical standards form the regulatory framework of the AEC sector. Manually checking a BIM model with hundreds or thousands of elements subject to standards grows combinatorially and faces three obstacles: (i) the **ambiguity and cross-references** of natural language; (ii) **regional variation**, which limits reuse across jurisdictions; and (iii) the **semantic difficulty** of distinguishing requirement, applicability, and exception within the same passage. The consolidated response in the literature is to structure the standard into a computable schema (such as RASE) before verification.

### 2.2 LLMs and Textual Validation Metrics

**LLMs** combine Machine Learning, deep neural networks, and NLP, adopting the *Transformer* architecture. **Prompt Engineering** is the lightest technique for specializing them on a domain task: it adjusts only the instruction, without changing weights or building a retrieval infrastructure.

Evaluating an LLM's output requires comparing it to a reference while tolerating paraphrases. To this end, this work uses **thirteen metrics across five families**:

- **Lexical** — *FuzzyWuzzy* (Levenshtein distance) and *TF-IDF* (cosine).
- **Semantic** — *SBERT-PT*, *Legal-BERTimbau* (specialized in Portuguese legal text), and a *Multilingual* model.
- **Distance-based** — *Word Mover's Distance* (WMD) over FastText and NILC embeddings.
- **Generation-oriented** — *BERTScore* (contextual token alignment) and *ROUGE-L* (longest common subsequence).
- **Classification** — Accuracy, Precision, Recall, and *F1-score*, which make omissions and hallucinations explicit.

---

## 3. RASE-PE: Converting Standards to RASE with Prompt Engineering

The proposed methodology — **RASE-PE** — rests on three decisions: (i) decompose the conversion into three successive levels (N1, N2, and N3); (ii) use Prompt Engineering as the **only** specialization technique; and (iii) systematically compare six open-source LLMs served locally through Ollama, under an identical protocol.

### 3.1 Three-Level Normalization

```
raw standard text
        │
        ▼
 ┌──────────────┐   N1 — Atomic Segmentation
 │      N1      │   Splits the sentence into atomic rules (one rule per item).
 └──────────────┘   Prompt Direct.
        │
        ▼
 ┌──────────────┐   N2 — RASE Operator Identification
 │      N2      │   Marks, in each rule, R / A / S / E.
 └──────────────┘   Chain-of-Thought.
        │
        ▼
 ┌──────────────┐   N3 — Semantic JSON Structuring
 │      N3      │   Extracts type, object, property, comparation, target, unit.
 └──────────────┘   Few-Shot (one prompt per operator).
        │
        ▼
 structured RASE representation (JSON)
```

### 3.2 Prompt Engineering

All adaptation lives in the *prompts* (`prompts/` folder), organized by level and by operator. Three classic techniques were applied according to the nature of each level:

- **N1 → *Prompt Direct*** — a direct instruction for a structurally simple task (segmentation), economical in tokens.
- **N2 → *Chain-of-Thought*** — asks the model to make its reasoning explicit step by step, reducing hallucinations in multi-operator classification.
- **N3 → *Few-Shot*** — complete input/output examples that anchor the rigid JSON schema, with one prompt per operator.

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

`dataset.json` consolidates **79 annotated standards** from Brazilian AEC regulations, with emphasis on **NBR 9050** (chosen for its mostly prescriptive profile, with directly verifiable numerical values). Each entry holds the reference RASE decomposition across the three levels: raw text (`text`), atomic rules (`texts_n1`), the four operators per rule (`operators_n2`), and, for each operator, the classified excerpt (`text_n2`) and the JSON properties (`properties_n3`). The same file serves both as input to the generators and as reference for the validators. See [`docs/dataset_schema.md`](docs/dataset_schema.md) for the full schema.

### 3.5 Computational Environment

All runs were carried out on a single workstation — **Intel Core i9-13900K CPU, 32 GB RAM, NVIDIA RTX 4080 GPU, Ubuntu 22.04, Python 3.11** — with the models served by Ollama and the metrics computed with `sentence-transformers`, `gensim`, `fuzzywuzzy`, and `scikit-learn`.

---

## 4. Experiments

The evaluation is organized into five experiments to isolate each level's contribution and measure error propagation along the chain:

| Experiment | Input | Evaluated output | Goal |
|------------|-------|------------------|------|
| **EN1**     | raw text | N1 | Atomic segmentation in isolation |
| **EN2**     | reference N1 | N2 | RASE identification in isolation |
| **EN1N2**   | raw text | N1→N2 | Chained pipeline (N1→N2 error propagation) |
| **EN3**     | reference N2 | N3 | JSON structuring in isolation |
| **EN1N2N3** | raw text | N1→N2→N3 | Full chained pipeline (production use) |

EN1, EN2, and EN3 measure each level in isolation (receiving the previous level as reference); EN1N2 and EN1N2N3 expose error propagation along the chain.

---

## 5. Results and Discussion

> All figures come directly from the `metrics/validate_*.json` files. Values on the normalized scale [0, 1]; the closer to 1, the greater the similarity between output and reference.

### 5.1 Per-metric overview

Per-metric averages across the five experiments, aggregated over the six models:

| Metric | EN1 | EN2 | EN1N2 | EN3 | EN1N2N3 |
|--------|:---:|:---:|:-----:|:---:|:-------:|
| *Lexical* — FuzzyWuzzy | 0.475 | 0.461 | 0.464 | **0.847** | 0.617 |
| *Lexical* — TF-IDF | 0.636 | 0.312 | 0.417 | 0.597 | 0.514 |
| *Semantic* — SBERT | 0.801 | 0.490 | 0.594 | **0.906** | 0.777 |
| *Semantic* — BERTimbau | 0.816 | 0.542 | 0.631 | 0.836 | 0.752 |
| *Semantic* — Multilingual | 0.850 | 0.595 | 0.679 | 0.873 | 0.793 |
| *Distance* — WMD (FastText) | 0.716 | 0.580 | 0.625 | 0.846 | 0.757 |
| *Distance* — WMD (NILC) | 0.688 | 0.546 | 0.595 | 0.842 | 0.742 |
| *Generation* — BERTScore | **0.866** | **0.753** | **0.790** | 0.880 | **0.843** |
| *Generation* — ROUGE-L | 0.599 | 0.308 | 0.404 | 0.765 | 0.616 |

Three patterns emerge: embedding-based metrics (SBERT, BERTimbau, Multilingual, and BERTScore) stay consistently higher; lexical ones (FuzzyWuzzy, TF-IDF, and ROUGE-L) lose ground when moving from EN1 into EN1N2; and EN3 rises to levels close to EN1 thanks to the restricted vocabulary of the JSON fields.

### 5.2 Global ranking by model

Global average per model (nine similarity metrics × five experiments):

| Model | EN1 | EN2 | EN1N2 | EN3 | EN1N2N3 | **Global Avg.** |
|-------|:---:|:---:|:-----:|:---:|:-------:|:---------------:|
| **Llama**   | 0.724 | **0.605** | 0.646 | **0.892** | **0.781** | **0.730** 🥇 |
| **Dolphin** | **0.801** | 0.576 | **0.650** | 0.843 | 0.757 | **0.725** 🥈 |
| **Mistral** | 0.757 | 0.570 | 0.634 | 0.875 | 0.771 | **0.721** 🥉 |
| Gemma   | 0.738 | 0.531 | 0.591 | 0.739 | 0.661 | 0.652 |
| Qwen    | 0.602 | 0.366 | 0.456 | 0.852 | 0.694 | 0.594 |
| Alpaca  | 0.676 | 0.408 | 0.488 | 0.718 | 0.610 | 0.580 |

- **Llama** — leads the global average, with absolute dominance over the N3 level (all nine metrics in EN3 and EN1N2N3; SBERT 0.947 in EN3) and the best performance in EN2. The choice when the computational cost is acceptable.
- **Dolphin** — the best segmenter in EN1 (dominating all nine metrics; BERTimbau 0.891) and the **best cost-benefit ratio**, combining competitive quality with the lowest generation times.
- **Mistral** — the most consistent of the batch; it never leads by a wide margin but is never far behind (within 1 percentage point of the leader).
- **Qwen** and **Alpaca** — consistently lower performance; Qwen also tends to produce output outside the expected language and records the highest latency in the chained pipeline.

### 5.3 Execution time (quality × cost trade-off)

Total generation time per model (seconds), by experiment:

| Model | N1 | N2 | N1N2 | N3 | N1N2N3 |
|-------|---:|---:|-----:|---:|-------:|
| Alpaca  | 104.6 | 249.1 | 330.8 | 2,104.9 | 1,395.4 |
| **Dolphin** | **62.7** | 124.7 | **104.3** | **585.0** | 757.7 |
| Gemma   | 133.0 | 244.2 | 427.2 | 5,982.4 | 6,364.5 |
| Llama   | 5,677.0 | 6,895.7 | 9,450.1 | 800.8 | 808.6 |
| Mistral | 70.0 | **123.6** | 138.4 | 627.7 | **637.9** |
| Qwen    | 309.6 | 6,515.1 | 10,093.8 | 847.4 | 808.7 |

Dolphin combines high quality with times in the range of 1 to 13 minutes across all experiments, while Llama and Qwen are the slowest in N1/N1N2 and Gemma becomes the most expensive in N3/N1N2N3.

### 5.4 Integrated discussion

1. **Semantic embeddings and BERTScore are the most suitable metrics** for this task — they capture equivalence even under heavy lexical reformulation and are the most stable across experiments. BERTimbau (PT-BR legal corpus) sits slightly above SBERT, evidence of the gain from domain embeddings.
2. **The chained pipeline does not degrade all metrics uniformly** — the semantic ones in EN1N2 even surpass EN2 in isolation (Multilingual 0.679 vs. 0.595), because the model's own N1 produces sentences close to the originals.
3. **The leading models have complementary profiles** — Llama (quality), Dolphin (cost-benefit), and Mistral (consistency).
4. **Qwen and Alpaca are not recommended** — lower performance across all metrics.
5. **The three-level decomposition was decisive** — separating segmentation, RASE classification, and JSON structuring made valid outputs possible using **Prompt Engineering alone**, without Fine-Tuning or RAG, reaching semantic similarity close to 0.90 in the best scenarios. The remaining difficulty concentrates on classifying the **Selection** and **Exception** operators in N2.

---

## 6. Conclusion

Decomposing the conversion of standards into three successive levels (N1, N2, N3), supported solely by Prompt Engineering, proved sufficient to produce valid RASE representations from the raw standard text. Llama leads overall quality (0.730), Dolphin delivers the best cost-benefit ratio, and Mistral is the most consistent. The main open challenge is the correct identification of the Selection and Exception operators in N2 — the stage where all models collapse — pointing to future work on prompts dedicated to these classes, richer few-shot examples, and possibly lightweight Fine-Tuning.

---

## Reproducing the experiment

### Requirements

- **Python 3.11+** (3.12 tested). Should work on Windows, Linux, or macOS.
- **[Ollama](https://ollama.com/download)** installed and running (`ollama serve`); models are downloaded automatically by the menu.
- The WMD metric uses `pot` (Python Optimal Transport), already in `requirements.txt`; the NILC FastText/Word2Vec (PT) weights are downloaded on demand from Hugging Face.

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

### Code structure

| Path | Responsibility |
|------|----------------|
| `main.py` | Main menu (generate / validate / tables) |
| `config/models.py` | Central registry of Ollama models (names + tags) |
| `prompts/` | Templates `n1.txt`, `n2.txt`, `n3_*.txt` (per operator), and `n3_combined.txt` |
| `dataset.json` | 79 annotated standards (input and reference) |
| `generates/` | One generation module per stage |
| `validates/` | One validation module per stage |
| `utils/validates/run_validation.py` | Single orchestrator of the validations |
| `predicts/` | Generated outputs (`generate_<n>_<model>.json`) |
| `metrics/` | Validation metrics (`validate_<n>.json`) |
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
