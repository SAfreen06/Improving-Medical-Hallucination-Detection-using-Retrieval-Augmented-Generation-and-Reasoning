"""Pluggable judge backends.

The paper runs 14 models through vLLM on 4x A6000. None of that fits a laptop,
so each backend here is something a laptop can actually reach:

  constant   always answers "hallucinated". The degenerate floor. On a balanced
             set this scores F1 = 0.667 at precision 0.5 -- which beats several
             real models in the paper's Table 2. Run it first; it is the number
             every other row has to actually beat.
  lexical    no model at all - a word-overlap heuristic. Zero setup, instant.
  ollama     a small quantised instruct model on CPU via a local Ollama server.
  anthropic  Claude via API. Stands in for the paper's GPT-4o row.
  openai     GPT via API.

Every backend answers the same question: given a question (and optionally the
supporting knowledge) plus one answer, is that answer hallucinated?
Return 1 = hallucinated, 0 = factual, 2 = not sure.
"""
import os
import re

import requests

# From the paper's detection prompt (Detection/detection_vllm_notsurecase.py in
# the official repo; also Appendix K of the paper).
SYSTEM_PROMPT = """You are an AI assistant with extensive knowledge in the field of medicine. I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

Type: Misinterpretation of #Question#: These are hallucinated answers that misunderstands the question, leading to an off-topic or irrelevant response.
Type: Incomplete Information: These are hallucinated answers that Point out what is not true without providing correct information.
Type: Mechanism and Pathway Misattribution - These are hallucinated answer that falsely attribution of biological mechanisms, molecular pathways, or disease processes that contradicts established medical knowledge
Type: Methodological and Evidence Fabrication - Inventing false research methods, statistical data, or specific clinical outcomes

You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated. Answer '0' if the answer is factual and '1' if the answer is hallucinated{not_sure_clause}. Do not return anything else, just the answer."""

NOT_SURE_CLAUSE = (" and '2' if you are unsure about the factuality and want to pass."
                   " If you are unsure about the answer choose '2' instead of guessing"
                   " and making a mistake")

# Chain-of-Thought, per Li et al. §V-A: zero-shot CoT in the style of Kojima
# et al. ("Let's think step by step"), with a delimited final answer so the
# verdict can be parsed out of the reasoning that precedes it.
COT_CLAUSE = """

Before answering, reason step by step:
1. What exactly is the question asking?
2. What does the world knowledge (if given) actually establish?
3. Does the answer contradict, overreach beyond, or sidestep that?

Then end your reply with a final line in exactly this form:
FINAL: <digit>"""

ANSWER_RE = re.compile(r"FINAL\s*:?\s*([012])", re.IGNORECASE)


def build_system_prompt(allow_not_sure: bool, cot: bool = False) -> str:
    prompt = SYSTEM_PROMPT.format(not_sure_clause=NOT_SURE_CLAUSE if allow_not_sure else "")
    if cot:
        # Drop the terseness instruction; it directly contradicts reasoning.
        prompt = prompt.replace(" Do not return anything else, just the answer.", "")
        prompt += COT_CLAUSE
    return prompt


def build_user_prompt(question, answer, knowledge=None, allow_not_sure=False, cot=False):
    head = f"World Knowledge: {knowledge}\n" if knowledge else ""
    if cot:
        tail = ("Reason step by step, then end with 'FINAL: <digit>' where the digit is "
                "'0' if factual, '1' if hallucinated"
                + (", '2' if unsure." if allow_not_sure else "."))
    else:
        tail = ("Return just the answer. '0' if factual, '1' if hallucinated"
                + (", '2' if unsure." if allow_not_sure else ".")
                + " Do not be verbose.")
    return f"{head}Question: {question}\nAnswer: {answer}\n\n{tail}\nYour Judgement:"


def parse_judgement(text: str, allow_not_sure: bool, cot: bool = False) -> int:
    """Extract the verdict from a reply.

    With CoT the digit must come from the FINAL: line, or failing that the LAST
    digit in the reply. Taking the first digit -- correct for terse replies --
    is actively wrong here, because the reasoning itself is full of digits
    ("step 1", "type 2 diabetes") that precede the verdict.
    """
    text = text or ""
    pattern = r"[012]" if allow_not_sure else r"[01]"

    if cot:
        m = ANSWER_RE.search(text)
        if m:
            value = int(m.group(1))
            return value if allow_not_sure or value != 2 else 0
        found = re.findall(pattern, text)
        if found:
            return int(found[-1])
    else:
        m = re.search(pattern, text)
        if m:
            return int(m.group())

    return 2 if allow_not_sure else 0


def majority_vote(votes: list[int], allow_not_sure: bool) -> int:
    """Self-consistency aggregation (Wang et al. 2023), cited by Li et al. §V.

    Abstentions are excluded from the tally when a real verdict exists -- a
    model that answers 1,1,2 has twice committed to 'hallucinated', and letting
    the single pass outvote that loses information. Ties go to 'hallucinated',
    the safer error in a clinical setting: flagging a true answer for review
    costs less than passing a fabricated one through.
    """
    committed = [v for v in votes if v != 2]
    if not committed:
        return 2 if allow_not_sure else 0
    ones = committed.count(1)
    return 1 if ones * 2 >= len(committed) else 0


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class ConstantBackend:
    """Always predicts 'hallucinated'. The degenerate baseline.

    MedHallu is balanced (every source row contributes one hallucinated and one
    factual answer), so this gets recall 1.0 at precision 0.5, for F1 = 0.667.
    Any model scoring below that has learned nothing useful, and several of the
    paper's rows do. Keep this in every comparison table.
    """

    name = "constant-always-hallucinated"

    def judge(self, question, answer, knowledge=None, allow_not_sure=False, cot=False):
        return 1


class LexicalBackend:
    """No LLM. Flags an answer as hallucinated when it shares few content words
    with the supporting knowledge.

    Deliberately dumb. Its job is to show what the F1 numbers look like when
    nothing understands the medicine, so the LLM rows have a floor to beat.
    The default threshold is tuned to keep precision and recall in the same
    ballpark; pushing it higher just degenerates toward ConstantBackend.
    """

    name = "lexical-baseline"
    STOP = set("""a an and are as at be by for from has have in is it its of on or that the to was were
    with this these those we our study results conclusion patients between which than there their""".split())

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def _tokens(self, text):
        return {w for w in re.findall(r"[a-z]{4,}", str(text).lower()) if w not in self.STOP}

    def judge(self, question, answer, knowledge=None, allow_not_sure=False, cot=False):
        # cot is meaningless without a model; accepted so the interface matches.
        reference = self._tokens(knowledge) if knowledge else self._tokens(question)
        answer_tokens = self._tokens(answer)
        if not answer_tokens:
            return 2 if allow_not_sure else 0
        overlap = len(answer_tokens & reference) / len(answer_tokens)
        return 1 if overlap < self.threshold else 0


class OllamaBackend:
    """Small instruct model on CPU through a local Ollama server."""

    def __init__(self, model="qwen2.5:1.5b-instruct", host="http://localhost:11434",
                 temperature=0.2, timeout=300):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.name = f"ollama/{model}"

    def judge(self, question, answer, knowledge=None, allow_not_sure=False, cot=False):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": build_system_prompt(allow_not_sure, cot)},
                {"role": "user", "content": build_user_prompt(
                    question, answer, knowledge, allow_not_sure, cot)},
            ],
            "stream": False,
            "options": {"temperature": self.temperature,
                        "num_predict": 400 if cot else 8},
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return parse_judgement(r.json()["message"]["content"], allow_not_sure, cot)


class AnthropicBackend:
    def __init__(self, model="claude-sonnet-5", temperature=0.2):
        from anthropic import Anthropic  # lazy import keeps the dep optional
        self.client = Anthropic()
        self.model = model
        self.temperature = temperature
        self.name = f"anthropic/{model}"

    def judge(self, question, answer, knowledge=None, allow_not_sure=False, cot=False):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=600 if cot else 8,
            temperature=self.temperature,
            system=build_system_prompt(allow_not_sure, cot),
            messages=[{"role": "user", "content": build_user_prompt(
                question, answer, knowledge, allow_not_sure, cot)}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return parse_judgement(text, allow_not_sure, cot)


class OpenAIBackend:
    def __init__(self, model="gpt-4o-mini", temperature=0.2):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.name = f"openai/{model}"

    def judge(self, question, answer, knowledge=None, allow_not_sure=False, cot=False):
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=600 if cot else 8,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": build_system_prompt(allow_not_sure, cot)},
                {"role": "user", "content": build_user_prompt(
                    question, answer, knowledge, allow_not_sure, cot)},
            ],
        )
        return parse_judgement(resp.choices[0].message.content, allow_not_sure, cot)


class SelfConsistency:
    """Wrap any backend: sample it n times, majority-vote the verdicts.

    Self-consistency (Wang et al. 2023), which Li et al. treat as the standard
    companion to CoT in SS V. The premise is that a single greedy chain can go
    wrong in one step and never recover, whereas independent chains tend to
    agree when the model actually knows and scatter when it does not.

    It only does anything at non-zero temperature -- identical samples cannot
    disagree -- so the temperature is raised here even though the rest of the
    benchmark runs near-deterministic. Cost is n forward passes per judgement.
    """

    def __init__(self, backend, n: int = 5, temperature: float = 0.7):
        self.backend = backend
        self.n = n
        self.name = f"{backend.name}+sc{n}"
        # Raise temperature so the chains actually differ.
        if hasattr(backend, "temperature"):
            backend.temperature = temperature

    def judge(self, question, answer, knowledge=None, allow_not_sure=False, cot=False):
        votes = [self.backend.judge(question, answer, knowledge, allow_not_sure, cot)
                 for _ in range(self.n)]
        return majority_vote(votes, allow_not_sure)


class HuggingFaceBackend:
    """Hugging Face pipeline / causal LM on CPU, matching the paper's framework."""

    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct", device="cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.model_name = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).to(device)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(device)
        self.model.eval()
        self.name = f"hf/{model.split('/')[-1]}"

    def judge(self, question, answer, knowledge=None, allow_not_sure=False, cot=False):
        import torch
        messages = [
            {"role": "system", "content": build_system_prompt(allow_not_sure, cot)},
            {"role": "user", "content": build_user_prompt(
                question, answer, knowledge, allow_not_sure, cot)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        max_new = 400 if cot else 8
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_judgement(generated, allow_not_sure, cot)


def get_backend(spec: str):
    """spec is 'constant', 'lexical', 'ollama:<model>', 'hf:<model>', 'anthropic:<model>'
    or 'openai:<model>'."""
    kind, _, model = spec.partition(":")
    if kind == "constant":
        return ConstantBackend()
    if kind == "lexical":
        return LexicalBackend(float(model) if model else 0.5)
    if kind == "ollama":
        return OllamaBackend(model or "qwen2.5:1.5b-instruct")
    if kind == "hf":
        return HuggingFaceBackend(model or "Qwen/Qwen2.5-1.5B-Instruct")
    if kind == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set.")
        return AnthropicBackend(model or "claude-sonnet-5")
    if kind == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set.")
        return OpenAIBackend(model or "gpt-4o-mini")
    raise SystemExit(f"unknown backend {spec!r}")

