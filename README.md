# Improving Medical Hallucination Detection using Retrieval-Augmented Generation and Reasoning

**Does retrieval (RAG), step-by-step reasoning (CoT), or both improve an LLM's ability to
*detect* hallucinated medical answers?**

Large language models increasingly answer medical and health questions, and their answers
can sound plausible while containing false facts, incorrect mechanisms, or fabricated
evidence. [MedHallu](https://arxiv.org/abs/2502.14302) (Pandit et al., 2025) shows that
general-purpose LLMs are poor at catching this on their own, and that giving the judge the
exact source passage helps enormously — F1 rises from **0.533 to 0.784** on average across
general models. That "with knowledge" setting is an **oracle**: the correct passage, handed
over, guaranteed. No deployed system has that.

This project asks the question MedHallu leaves open: **how much of that gain survives when
the knowledge has to be found by a real retriever instead of handed over?** We add retrieval
and reasoning as detection-time mitigations, evaluate them against a genuinely external
corpus, and report the results.

---

## Table of contents

- [Motivation](#motivation)
- [Related work](#related-work)
- [Research gap and contribution](#research-gap-and-contribution)
- [Methodology](#methodology)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Key findings](#key-findings)
- [Known limitations](#known-limitations)
- [Future work](#future-work)
- [Team contributions](#team-contributions)
- [References](#references)

---

## Motivation

- Large language models increasingly answer medical and health questions.
- Their answers can sound plausible while containing false facts, incorrect mechanisms, or
  fabricated evidence.
- Human-created health misinformation spreads the same way, through social media, news
  sites, and online forums.
- A reliable detector should verify medical claims against external evidence before making a
  decision — not just judge a claim in isolation.

Existing hallucination detectors judge a claim without reliable external evidence, so a
plausible but false statement can be misclassified as correct. MedHallu shows that supplying
relevant knowledge helps significantly, but in its benchmark that knowledge is given
directly rather than found by a real retrieval system. Our objective is to close that gap:
retrieve real evidence with RAG before judging an answer, add chain-of-thought reasoning,
and combine both as RAG + CoT.

## Related work

| Work | Contribution | Relevance / limitation |
|---|---|---|
| **Pandit et al. (2025)**, *MedHallu* [[arXiv:2502.14302]](https://arxiv.org/abs/2502.14302) | 10,000 medical QA pairs derived from PubMedQA; grades hallucinations by Easy/Medium/Hard difficulty and four error types; shows supplied knowledge lifts average F1 from 0.533 to 0.784 (+0.251). | The knowledge condition hands the detector ground-truth context directly — it never has to search a realistic external corpus. This project's starting point and baseline. |
| **Xiong et al. (2024)**, *MedRAG* [[arXiv:2402.13178]](https://arxiv.org/abs/2402.13178) | Benchmarks 41 combinations of corpora, retrievers, and six LLMs for medical QA; reports a 1–18% relative accuracy gain over CoT prompting depending on model and task. | Targets question *answering*, not hallucination *detection*, and doesn't test whether retrieved evidence matches MedHallu's exact source context. Supplies the corpora and retrieval recipe this project builds on. |
| **Li et al. (2025)**, *Mitigating Hallucination in LLMs* [[arXiv:2510.24476]](https://arxiv.org/abs/2510.24476) | Separates knowledge-based hallucinations (wrong or missing facts) from logic-based hallucinations (sound facts, broken reasoning); argues RAG fixes the former and CoT the latter. | A broad survey with no original benchmark and few medical-specific examples. Supplies the RAG-vs-CoT taxonomy this project tests directly on MedHallu. |
| **Feng et al. (2025)**, *Health Misinformation Detection* [[DOI]](https://doi.org/10.1177/00469580251384784) | Reviews 100 health-misinformation detection studies (2016–2025); organizes methods into ML, deep learning, knowledge graphs, fact-checking, and LLM-based approaches; identifies evidence grounding and interpretability as open research needs. | Covers human- and AI-generated misinformation broadly, with no single benchmark comparable to MedHallu F1. Motivates evidence-grounded detection as a research direction. |

## Research gap and contribution

MedHallu shows that medical knowledge improves hallucination detection; this project
investigates whether that improvement survives when the system has to retrieve the
knowledge itself, rather than receiving it directly.

|  | Knowledge source |
|---|---|
| **Oracle** (MedHallu's setting) | Evidence is supplied directly, close to perfect / guaranteed-correct retrieval. |
| **RAG** (this project) | Evidence is retrieved by a real system before judging — the corpus may simply not contain the specific finding a question needs. |

## Methodology

Four conditions form the core comparison, run over the same question/answer pairs:

| # | Condition | What the judge sees |
|---|---|---|
| 1 | **Baseline** | Question + answer only, no external evidence. |
| 2 | **Oracle** | ...plus the exact source passage MedHallu ships. The paper's ceiling. |
| 3 | **RAG** | ...plus top-k passages retrieved from an external medical corpus. |
| 4 | **CoT** | No extra text — the model reasons step by step before giving a verdict. |
| 5 | **RAG + CoT** | Retrieved passages **and** step-by-step reasoning, combined. |

An optional diagnostic condition, **Oracle + RAG**, keeps the correct passage and adds the
retrieved ones on top. It isolates *why* RAG trails oracle — how much is lost to
distractors added by retrieval versus the passage retrieval fails to find at all.

**Retrieval pipeline.** Vanilla RAG is only the middle step of a larger pipeline
([`src/pipeline.py`](src/pipeline.py)):

1. **Pre-retrieval** — rewrite a paper-title-style MedHallu question into the vocabulary a
   textbook index would actually match, or expand it into several queries covering
   different angles (mechanism, clinical condition, general principle).
2. **Retrieval** — sparse (TF-IDF) and dense (sentence-embedding) search over an external
   medical corpus, fused by Reciprocal Rank Fusion when combined ("hybrid" retrieval), since
   the two fail differently: sparse misses paraphrase, dense misses rare exact terms like
   drug names and gene symbols.
3. **Post-retrieval** — rerank the shortlist by similarity to the query and drop passages
   below a similarity threshold, since a filtered empty result is better than confidently
   irrelevant distractors.

**Reasoning.** Chain-of-thought prompting asks the model to reason step by step before
committing to a verdict, with a delimited `FINAL: <digit>` line so the verdict can be parsed
out of the reasoning that precedes it. Self-consistency (sampling the judge multiple times
and majority-voting) is available as an additional companion to CoT.

**Corpus.** RAG searches [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks)
— 125,847 snippets from 18 medical textbooks — a genuinely different source from MedHallu's
PubMed-derived questions, so the corpus may or may not contain the specific finding a
question is about. That uncertainty is the realistic case. Retrieving from MedHallu's own
`Knowledge` field is kept only as a labelled ablation (`--corpus self`): it shows the upper
bound retrieval could reach, but is not reported as a RAG result, since the correct passage
is present by construction.

## Repository layout

```
docs/
  STEP1_VALIDATE_BASELINE.md   Reproduce the paper's numbers first

scripts/
  patch_original.py    Patches the official MedHallu repo to run on a laptop
  patch_cot.py          Upgrades bare CoT to the full Li et al. reasoning family
  compare_to_paper.py  Joins your results against MedHallu's Table 2
  build_mitigation_csv.py  Reshapes experiment.py's JSON output into a flat results CSV

src/
  config.py            Paths; redirects model caches off the system drive
  data.py               Dataset download; the knowledge/logic taxonomy mapping
  make_original_csv.py Converts the HF dataset into the columns MedHallu expects
  backends.py           Judges: constant, lexical, ollama, anthropic, openai,
                        plus CoT prompting and self-consistency voting
  rag.py                External corpora, TF-IDF and dense retrieval, recall@k
  pipeline.py           Full RAG pipeline: query rewrite/expansion, hybrid
                        retrieval, reranking and filtering
  detect.py             The benchmark: metrics by difficulty and hallucination type
  experiment.py         The RAG x CoT grid in one run
  semantics.py          Similarity analysis of the dataset itself
  generate_demo.py      Walks the dataset-generation pipeline step by step

outputs/                Curated, human-readable results CSVs (tracked in git)
data/                   Datasets and downloaded corpora (gitignored, rebuilt on demand)
results/                Raw per-run dumps from src/ scripts (gitignored, regenerable)
```

## Getting started

Everything runs free on a laptop — no GPU required.

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the core pipeline (dataset, benchmark, metrics). Embedding-based
retrieval and API judge backends are optional and listed separately in the file.

[Ollama](https://ollama.com) serves an OpenAI-compatible endpoint on `localhost:11434`, so
the same code path that reaches Anthropic/OpenAI also reaches free, local models — twelve
of MedHallu's fourteen evaluated models fit in 16 GB at Q4 quantisation. Speed, not memory,
is the constraint on CPU: budget roughly 25 seconds per judgement for a 7B model, so use
200–300 rows for a run, not 10,000.

## Usage

**A. Patch the official MedHallu repo — for reportable, paper-comparable results.** The
official code doesn't run without a CUDA GPU and can't read its own published dataset;
`patch_original.py` fixes both, plus a scoring-parser bug, and adds the RAG/CoT conditions:

```bash
python scripts/patch_original.py /path/to/MedHallu
```

**B. This repo's own runner — faster to iterate on.**

```bash
python src/experiment.py --backend ollama:qwen2.5:7b-instruct --limit 200 --corpus textbooks
```

Runs all five conditions and prints one table, sliced by difficulty and by hallucination
type. Add `--diagnose` for the Oracle + RAG row. This mode is not directly comparable to the
paper, since it scores both answers per row where the paper scores one at random.

`--backend constant` and `--backend lexical` are not models — they're plumbing tests (a
constant classifier and a word-overlap heuristic) used to confirm the pipeline runs without
a GPU. Their numbers should not be read as detection results.

## Results

Real experiment results across two models, oracle knowledge, and realistic retrieval and
reasoning (`outputs/results_mitigations.csv`). Oracle uses ground-truth context; RAG
retrieves external evidence from the textbooks corpus; CoT changes the reasoning process;
RAG + CoT combines both.

**qwen2.5:7b-instruct**

| Condition | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.591 | 0.578 | 0.584 |
| Oracle | 0.788 | 0.867 | **0.825** |
| RAG | 0.552 | 0.711 | 0.621 |
| CoT | 0.530 | 0.789 | 0.634 |
| RAG + CoT | 0.500 | 0.889 | 0.640 |
| Oracle + RAG | 0.820 | 0.811 | 0.816 |

**llama3.1:8b**

| Condition | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.791 | 0.378 | 0.511 |
| Oracle | 0.943 | 0.556 | 0.699 |
| RAG | 0.745 | 0.389 | 0.511 |
| CoT | 0.500 | 0.680 | 0.576 |
| RAG + CoT | 0.500 | 0.720 | 0.590 |

RAG + CoT gave the best non-oracle F1 for both models (Qwen 0.584 → 0.640; Llama
0.511 → 0.590), mainly through higher recall — precision fell to 0.500 for both.

## Key findings

- **RAG + CoT beat every other non-oracle condition for both models tested**, but the gain
  came from recall, not precision — both models converged toward flagging more answers as
  hallucinated rather than judging more accurately.
- **RAG alone was model-dependent.** It helped Qwen's F1 (0.584 → 0.621) but left Llama's
  exactly flat (0.511 → 0.511), so retrieval is not a free win on its own.
- **Oracle knowledge remained substantially stronger than anything retrieved**, confirming
  that retrieval quality — not reasoning — is the main bottleneck in evidence-grounded
  medical hallucination detection.
- **A trivial constant baseline beats most published models.** MedHallu is balanced, so
  always answering "hallucinated" scores F1 = 0.667 at precision 0.5 — higher than 13 of
  the paper's 14 evaluated models score without knowledge. This doesn't contradict the
  paper's point that models are bad at this task, but it means F1 alone can flatter a
  model; a constant-baseline row belongs in any comparison table.
- **A parsing bug in the official scoring code** tests `'not'` before `'not sure'`, scoring
  `"not sure"` as `"hallucinated"`, and `'non'` inverts `"non-hallucinated"`. Of 13 realistic
  model replies, 8 parsed wrong before the fix in `scripts/patch_original.py`.
- **Reasoning text broke the original short-answer parser.** CoT text like "Step 1... type 2
  diabetes..." confused a parser that read only the first digit in a reply. The fix requires
  a delimited `FINAL: <digit>` line and falls back to the last valid digit if that line is
  missing.

## Known limitations

- **Quantised, not fp16.** Q4 models are a different measurement from the paper's fp16
  runs.
- **Hundreds of rows, not 10,000.** Differences under roughly 0.05 F1 are noise at this
  sample size.
- **Textbooks, not PubMed.** The corpus may simply lack the specific finding a question
  needs; results bound what *textbook* retrieval can do, not retrieval in general.
- **Two models evaluated, not fourteen.**
- **The knowledge/logic hallucination-type taxonomy is ours, not Li et al.'s.** It is the
  hinge of the RAG-vs-CoT split and is arguable — see `data.HALLUCINATION_TYPE`.
- **CoT here is zero-shot, single-sample only** in the headline results — few-shot
  prompting, self-verification, and self-consistency voting are implemented but not run in
  every condition reported above.
- **The Ollama path is exercised but not exhaustively tested end to end.** The prompt,
  parser, and retrieval logic are each tested individually; smoke-test a new model with a
  small `--limit` before trusting a full run.

## Future work

- **From hallucination to misinformation.** Apply the same evidence-grounded framework to
  human-written claims, not just LLM outputs.
- **Multilingual extension.** Test retrieval and reasoning across languages, since health
  misinformation isn't English-only.
- **A stronger retrieval pipeline.** Since retrieval quality is the main bottleneck, explore
  better rerankers, denser medical embeddings, or multi-hop retrieval.

## Team contributions

- **Sanjana Afreen (220042106)** — Baseline reproduction, model evaluation, literature
  review, presentation
- **Mrittika Jahan (220042150)** — Textbook RAG, hybrid retrieval, reranking, output parsing
- **Jarin Subha Sneha (220042161)** — PubMed retrieval, confidence calibration, CoT
  variants, self-verification

## References

- Pandit, S., Xu, J., Hong, J., Wang, Z., Chen, T., Xu, K., & Ding, Y. (2025). *MedHallu: A
  Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models.*
  [arXiv:2502.14302](https://arxiv.org/abs/2502.14302)
- Xiong, G., Jin, Q., Lu, Z., & Zhang, A. (2024). *Benchmarking Retrieval-Augmented
  Generation for Medicine.* [arXiv:2402.13178](https://arxiv.org/abs/2402.13178)
- Li, Y., Fu, X., Verma, G., Buitelaar, P., & Liu, M. (2025). *Mitigating Hallucination in
  Large Language Models: An Application-Oriented Survey on RAG, Reasoning, and Agentic
  Systems.* [arXiv:2510.24476](https://arxiv.org/abs/2510.24476)
- Feng, X., Luo, J., Yang, Y., El Baz, D., & Shi, L. (2025). *Health Misinformation
  Detection: Approaches, Challenges and Opportunities.* INQUIRY, 62.
  [DOI:10.1177/00469580251384784](https://doi.org/10.1177/00469580251384784)
- Source data: [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (`qiaojin/PubMedQA`).
