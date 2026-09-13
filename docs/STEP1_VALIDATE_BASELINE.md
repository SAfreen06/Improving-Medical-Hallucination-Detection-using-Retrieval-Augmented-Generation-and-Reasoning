# Step 1 — Reproduce MedHallu's own numbers before changing anything

Goal: run the MedHallu code **exactly as the paper does** — no RAG, no CoT — and
check your F1 lands near the published Table 2 row. If it does, your setup is
sound and any later difference is caused by the mitigation, not by a broken
pipeline. **Do not skip this.** A mitigation that "improves" a broken baseline
tells you nothing.

---

## The 14 models in MedHallu's Table 2

**General-purpose (10)**

| # | Model | F1 no-knowledge | F1 oracle | **Δ** | Runs on 16 GB? |
|---|---|---|---|---|---|
| 1 | GPT-4o | 0.737 | 0.877 | +0.140 | No — API only |
| 2 | GPT-4o-mini | 0.607 | 0.841 | +0.234 | No — API only |
| 3 | Qwen2.5-14B-Instruct | 0.619 | 0.852 | +0.233 | Tight (~9 GB at Q4) |
| 4 | Gemma-2-9b-Instruct | 0.515 | 0.838 | **+0.323** | Yes, slow |
| 5 | Llama-3.1-8B-Instruct | 0.522 | 0.797 | +0.275 | **Yes** |
| 6 | DeepSeek-R1-Distill-Llama-8B | 0.514 | 0.812 | +0.298 | Yes |
| 7 | Qwen2.5-7B-Instruct | 0.553 | 0.839 | +0.286 | **Yes** |
| 8 | Qwen2.5-3B-Instruct | 0.606 | 0.676 | +0.070 | Yes, fastest |
| 9 | Llama-3.2-3B-Instruct | 0.499 | 0.734 | +0.235 | Yes, fast |
| 10 | Gemma-2-2b-Instruct | 0.553 | 0.715 | +0.162 | Yes, fastest |

**Medical fine-tuned (4)**

| # | Model | F1 no-knowledge | F1 oracle | **Δ** | Runs on 16 GB? |
|---|---|---|---|---|---|
| 11 | OpenBioLLM-Llama3-8B | 0.484 | 0.424 | **−0.060** | Yes |
| 12 | BioMistral-7B | 0.570 | 0.648 | +0.078 | **Yes** |
| 13 | Llama-3.1-8B-UltraMedical | 0.619 | 0.773 | +0.153 | Yes |
| 14 | Llama3-Med42-8B | 0.416 | 0.797 | **+0.381** | Yes |

Δ = how much oracle knowledge helped that model in the paper.

---

## The 3 we use, and why

| Model | Ollama tag | Δ | Role |
|---|---|---|---|
| **Qwen2.5-7B-Instruct** | `qwen2.5:7b-instruct` | +0.286 | General, mid-size, large knowledge effect |
| **Llama-3.1-8B-Instruct** | `llama3.1:8b` | +0.275 | General, second architecture — guards against a Qwen-specific quirk |
| **BioMistral-7B** | GGUF import | +0.078 | Medical fine-tune, and it barely benefits from knowledge |

**Selected by Δ, not by F1.** Δ is the headroom RAG has to work in. RAG is
imperfect knowledge, so a model that gained +0.286 from perfect knowledge can
show whether retrieval recovers some of it. A model that gained +0.070 cannot.

**Qwen2.5-3B is the fastest and a bad choice for the real run** — Δ+0.070 leaves
almost nothing for RAG to demonstrate. Use it only to smoke-test the pipeline.

BioMistral is deliberately awkward: the paper's medical models barely benefit
from knowledge and OpenBioLLM actually gets *worse*. If RAG helps it anyway,
that is a result worth reporting.

**Two models is acceptable if time is short** — drop Llama-3.1-8B, keep
Qwen2.5-7B and BioMistral. Keep one general and one medical; that contrast is
the paper's most interesting finding.

---

## Steps

### 1. Install Ollama and pull the models

Install from <https://ollama.com>. Then:

```bash
ollama pull qwen2.5:7b-instruct
```

```bash
ollama pull llama3.1:8b
```

BioMistral is not in Ollama's own library; it comes from a HuggingFace GGUF
repo. Something like:

```bash
ollama pull hf.co/MaziyarPanahi/BioMistral-7B-GGUF:Q4_K_M
```

**Check that repo exists before relying on it** — community GGUF uploads move.
Search HuggingFace for "BioMistral GGUF" and use whatever is current.

Confirm the server is up:

```bash
ollama list
```

### 2. Clone MedHallu and install dependencies

```bash
git clone https://github.com/MedHallu/MedHallu.git
```

```bash
pip install pandas scikit-learn openai tqdm requests pyarrow
```

No torch, no vLLM, no `datasets` — the patch removes those requirements.

### 3. Patch the detection script

```bash
python scripts/patch_original.py /path/to/MedHallu
```

You need this even for the baseline run: as shipped, the script imports vLLM at
module level and cannot start without a CUDA GPU, and its OpenAI branch uses an
API removed in openai 1.0.

### 4. Build the data file

MedHallu's script cannot read MedHallu's own published dataset — the column
names differ. Convert it:

```bash
python src/make_original_csv.py --config pqa_labeled
```

### 5. Configure for the BASELINE ONLY

In the patched script's `CONFIGURATION` block, cut `CONDITIONS` down to the two
settings the paper actually reports:

```python
CONDITIONS = [
    ("baseline", None,     False),   # paper's "Without Knowledge"
    ("oracle",   "oracle", False),   # paper's "With Knowledge"
]
```

```python
DF_PATH = "data/original_format_pqa_labeled.csv"
CSV_PATH = "results_baseline.csv"
LIMIT = 300
```

and the models list:

```python
models = [
    {'type': 'openai', 'model_name': 'qwen2.5:7b-instruct'},
    {'type': 'openai', 'model_name': 'llama3.1:8b'},
]
```

`'openai'` is the *protocol*, not the vendor. Ollama speaks it; the model runs
on your machine.

### 6. Smoke test — 20 rows, one model

Set `LIMIT = 20`, leave one model in the list, and run:

```bash
python Detection/detection_vllm_notsurecase.py
```

A few minutes. You want **two rows** in the CSV with real numbers in the `f1`
column. Ignore the values — 20 rows means nothing. You are checking the
machinery runs.

### 7. The real baseline run

Set `LIMIT = 300`, restore both models, run again. Roughly 2–3 hours per model
for both conditions.

### 8. Compare against the paper

```bash
python scripts/compare_to_paper.py outputs/results_baseline.csv
```

---

## What counts as passing

| Model | Expect baseline ≈ | Expect oracle ≈ |
|---|---|---|
| Qwen2.5-7B-Instruct | 0.553 | 0.839 |
| Llama-3.1-8B-Instruct | 0.522 | 0.797 |

**Within about ±0.05 is a pass.** At 300 rows against the paper's 10,000, with
Q4 quantisation instead of fp16, exact agreement is not expected and not
required. The direction and rough magnitude are what matter — especially that
oracle is far above baseline.

**If you are far off, stop and diagnose.** Likely causes, in order:

1. `f1` column blank or 0 → model is not returning parseable answers. Print a
   few raw replies.
2. baseline ≈ oracle → the knowledge is not reaching the prompt. Check the
   `knowledge` column survived the CSV conversion.
3. Everything ≈ 0.667 with precision 0.5 → the model is answering
   "hallucinated" every time. That is the degenerate baseline, not a result.
4. Wildly low everywhere → check `ollama list` and that the model tag in the
   script matches exactly.

Only once this passes does Step 2 — adding RAG and CoT — mean anything.
