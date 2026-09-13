# medhallu-mitigation

**Does RAG, Chain-of-Thought, or both improve hallucination *detection* in
medical QA?**

MedHallu benchmarks how well LLMs spot fabricated medical answers, and finds
they are poor at it. We add the mitigation techniques from Li et al. and measure
whether they help.

Baselines come from the MedHallu paper, so we do not re-run 14 models. We add
conditions and compare against published numbers.

---

## The question

MedHallu ([arXiv:2502.14302](https://arxiv.org/abs/2502.14302)) reports two
settings per model:

| Setting | What the judge sees | Qwen2.5-7B F1 |
|---|---|---|
| Without knowledge | question + answer | 0.553 |
| With knowledge | question + answer + **the exact source abstract** | 0.839 |

That second setting is **oracle retrieval** — the correct passage, handed over,
guaranteed. No deployed system has it. Yet it is the paper's largest single
effect: **+0.286 F1**.

So the interesting number is missing. What happens with *realistic* knowledge —
retrieved from an outside corpus, sometimes wrong, always noisy?

Li et al. (*Mitigating Hallucination in LLMs: An Application-Oriented Survey on
RAG, Reasoning, and Agentic Systems*) argue two mechanisms fix two different
failures:

- **RAG** fixes **knowledge-based** hallucinations — the model lacked the facts
- **CoT** fixes **logic-based** hallucinations — it had them and reasoned badly

Neither paper tests that split on medical hallucination *detection*. That is
what this project does.

---

## Five results

| # | Condition | What the judge gets |
|---|---|---|
| 1 | `baseline` | Question + answer only. Detection as the paper measures it. |
| 2 | `oracle` | ...plus the exact source abstract. The paper's ceiling. |
| 3 | `rag` | ...plus top-k passages retrieved from **medical textbooks** |
| 4 | `cot` | No extra text — the model is told to reason step by step first |
| 5 | `rag+cot` | Retrieved passages **and** step-by-step reasoning |

Rows 1 and 2 come from the MedHallu paper, so they double as a check that the
setup is sound. Rows 3–5 are the contribution.

### Optional diagnostic: `oracle+rag`

Not a headline result — it answers a different question. `rag` trails `oracle`
for two reasons that cannot otherwise be separated: retrieval sometimes
**misses** the right passage, and it **adds distractors** even when it hits.
`oracle+rag` keeps the correct passage and adds the retrieved ones on top, which
splits the shortfall:

```
oracle − (oracle+rag)  =  what distractors alone cost
(oracle+rag) − rag     =  what missing the right passage costs
```

If distractors dominate, the fix is a reranker. If missing passages dominate,
the fix is a bigger corpus. Opposite engineering decisions. Enable with
`--diagnose`.

---

## RAG uses an external corpus, not MedHallu's own answers

**This matters and is easy to get wrong.** An early version of this project
retrieved from MedHallu's own `Knowledge` column. That is not RAG — the correct
passage is present by construction, so the task reduces to ranking it first.

RAG now searches **MedRAG/textbooks**: 125,847 snippets from 18 medical
textbooks (Xiong et al.). 101 MB, ungated, cached after first download.
`statpearls`, `pubmed` and `wikipedia` are also wired up; PubMed is 23.9M
passages and needs more than a laptop.

It is a genuinely hard test. MedHallu questions come from PubMed research
abstracts; a textbook may not contain the specific finding at all. Verbatim from
a run:

> **Question:** Do mitochondria play a role in remodelling lace plant leaves
> during programmed cell death?
>
> **Retrieved:** Histology_Ross on programmed cell death · Pathology_Robbins on
> mitochondrial roles · Gynecology_Novak on apoptosis

Topically adjacent, specifically useless — none mention lace plants. **Expect
RAG to fall short of oracle. That gap is the result.**

`--corpus self` reproduces the old behaviour as a labelled ablation. Do not
report it as a RAG result.

---

## Repository layout

```
docs/
  STEP1_VALIDATE_BASELINE.md   Reproduce the paper's numbers first
  STEP2_ADD_MITIGATIONS.md     Then add RAG / CoT

scripts/
  patch_original.py    Patches the official MedHallu repo (see below)
  patch_cot.py          Upgrades bare CoT to the full Li et al. SS V-A family
  compare_to_paper.py  Joins your results against MedHallu Table 2
  build_mitigation_csv.py  Reshapes experiment.py's JSON into a flat results CSV

src/
  config.py            Paths; redirects model caches off the system drive
  data.py              Dataset download; the knowledge/logic taxonomy mapping
  make_original_csv.py Converts the HF dataset into the columns MedHallu expects
  backends.py          Judges: constant, lexical, ollama, anthropic, openai,
                       plus CoT prompting and self-consistency voting
  rag.py               External corpora, TF-IDF and dense retrieval, recall@k
  detect.py            The benchmark: metrics by difficulty and hallucination type
  experiment.py        The RAG x CoT grid in one run
  semantics.py         Similarity analysis of the dataset itself
  generate_demo.py     Walks the dataset-generation pipeline step by step

outputs/                Curated, human-readable results CSVs (tracked in git)
data/                   Datasets and downloaded corpora (gitignored, rebuilt on demand)
results/                Raw per-run dumps from src/ scripts (gitignored, regenerable)
```

---

## Two ways to run it

### A. Patch the official MedHallu repo — use this for reportable results

The official code cannot run on a laptop and cannot read its own published
dataset. `patch_original.py` fixes seven things:

| # | Problem in the official repo |
|---|---|
| 0 | `datasets` imported but never used; `torch` imported for GPU-only paths |
| 1 | `from vllm import ...` at **module level** — won't import without CUDA |
| 2 | OpenAI branch uses `openai.ChatCompletion.create`, removed in openai ≥1.0 |
| 3 | Reply parser scores `"not sure"` as `"hallucinated"` (see findings) |
| 4–7 | Adds CoT, RAG, the condition loop, and a config block |

```bash
python scripts/patch_original.py /path/to/MedHallu
```

Backs up to `.py.orig`; `--revert` undoes it. Fails loudly if upstream text has
changed rather than silently doing nothing.

### B. This repo's own runner — faster to iterate on

```bash
python src/experiment.py --backend ollama:qwen2.5:7b-instruct --limit 200 --corpus textbooks
```

Runs all five conditions and prints one table, sliced by difficulty and by
hallucination type. Add `--diagnose` for the `oracle+rag` row.

**Not directly comparable to the paper** — it scores both answers per row,
where the paper scores one at random.

**On backends:** `constant` and `lexical` are *not models*. They are plumbing
tests — a constant classifier and a word-overlap heuristic — used to verify the
code runs without a GPU. Ignore their numbers. `ollama:<model>` runs a real LLM
locally and free; `anthropic` and `openai` need paid keys.

---

## Everything runs free on a laptop

No GPU required. Ollama serves an OpenAI-compatible endpoint on localhost, so
the same code path reaches local models:

```python
API_KEY = "ollama"
BASE_URL = "http://localhost:11434/v1"
```

Twelve of MedHallu's fourteen models fit in 16 GB at Q4 quantisation; only
GPT-4o and GPT-4o-mini are API-only. **Speed, not memory, is the constraint** —
budget ~25 s per judgement for a 7B model on CPU, so use 200–300 rows, not
10,000.

`docs/STEP1_VALIDATE_BASELINE.md` lists all 14 models with their published scores and
explains why the three we use were picked — by Δ-knowledge, not by F1.

---

## Things we found along the way

Four observations, each verified against the data or the code. Not the project's
goal, but they affect how results should be read.

**1. A trivial baseline beats most of the paper's models.** MedHallu is
balanced, so a classifier that always answers "hallucinated" scores **F1 =
0.667** at precision 0.5. In Table 2 without knowledge, only GPT-4o (0.737)
clears it — Llama-3.1-8B is 0.522, Med42-8B is 0.416. Thirteen of fourteen score
below a constant. This does not contradict the paper, whose point is that models
are bad at this, but F1 alone flatters them. `--backend constant` reproduces it;
include that row in any table you make.

**2. A parsing bug in the official scoring code.** `calculate_metrics` tests
`'not'` before `'not sure'`, so `"not sure"` is scored as `"hallucinated"` and
the not-sure branch is unreachable. `'non'` catches `"non-hallucinated"` and
inverts it. Of 13 realistic replies, 8 parse wrong. The patch fixes it. Impact
on the published numbers is **unquantified** — most replies are bare digits —
but Table 4's "not sure" rates are the most exposed.

**3. The official script cannot read the official dataset.** It expects
`question`, `ground_truth`, `least_similar_answer`; the HuggingFace release
ships `Question`, `Ground Truth`, `Hallucinated Answer`, and a differently
shaped `knowledge` field. The README says to load from HuggingFace; it would
crash. `make_original_csv.py` converts it.

**4. A claim in §5.3 reverses on the released data.** The paper reports that
harder-to-detect hallucinations sit semantically *closer* to the ground truth.
Measured across all 10,000 released rows by ROUGE-1, TF-IDF cosine and
sentence-embedding cosine, the effect runs the other way — easy 0.742 vs hard
0.701 embedding cosine, p ≈ 1e-23 on the 9k split.

This is **not** a refutation. The paper compares clusters over 50 candidate
generations per question; we compare the single released answer. And the
generation fallback (Algorithm 1, Phase 2) selects the candidate with *maximum*
cosine similarity to ground truth and labels it `easy`, which biases exactly
this way. The narrower takeaway: **`difficulty` is partly an artifact of how the
generation loop terminated**, so be careful using it as a subtlety axis.
`python src/semantics.py --embed` reproduces it.

---

## Known limitations

State these before anyone asks.

- **Quantised, not fp16.** Q4 models are a different measurement from the paper's.
- **Hundreds of rows, not 10,000.** Differences under ~0.05 F1 are noise.
- **Textbooks, not PubMed.** The corpus may simply lack the specific finding.
  Results bound what *textbook* retrieval can do, not retrieval in general.
- **Two or three models, not fourteen.**
- **The knowledge/logic taxonomy mapping is ours, not Li et al.'s.** Calling
  "Misinterpretation of #Question#" logic-based is defensible but arguable, and
  it is the hinge of the RAG-vs-CoT split. See `data.HALLUCINATION_TYPE`.
- **Category sizes are very uneven.** 76% of MedHallu is one category; Evidence
  Fabrication has 3 rows in the labeled split and 46 in the artificial one. Use
  `pqa_artificial` for any per-category claim.
- **The Ollama path is untested end to end.** The patch, the parser and the
  retrieval are tested here; the API call itself was written against the current
  SDK but not executed. Smoke-test with 20 rows first.

---

## Credits

- **MedHallu** — Pandit et al., [arXiv:2502.14302](https://arxiv.org/abs/2502.14302).
  Dataset and baselines. MIT licensed.
- **Mitigating Hallucination in LLMs** — Li et al. RAG/CoT taxonomy and methods.
- **MedRAG corpora** — Xiong et al., *Benchmarking RAG for Medicine*.
- Source data: PubMedQA (`qiaojin/PubMedQA`).
