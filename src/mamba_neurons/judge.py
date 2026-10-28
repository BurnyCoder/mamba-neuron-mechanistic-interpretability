"""LLM judge for steered generations: AxBench concept-relevance and fluency ratings (0-2).

Local explanation: the two prompts are the AxBench judge templates
(https://github.com/stanfordnlp/axbench/blob/main/axbench/evaluators/prompt_templates.py, Wu et al. 2025
https://arxiv.org/abs/2501.17148) with one change: instead of the free-text "Rating: [[score]]" line, the local model
returns JSON {"explanation", "rating"} through Ollama structured outputs (explanation first, as AxBench asks).
Global context: per-generation harmonic mean of the two ratings is the primary H2 metric.
"""

from __future__ import annotations

CONCEPT_TEMPLATE = """Please act as an impartial judge and evaluate whether the specified concept is presented in the \
sentence fragment provided below. Focus solely on whether the concept is clearly incorporated, without regard for \
grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is clearly incorporated. Point out \
references to the concept in the fragment, noting any minimal or awkward incorporations. After providing your \
explanation, rate the concept's relevance on a scale from 0 to 2, where 0 indicates the concept is not present at \
all, 1 indicates the concept is somewhat present but minimally or awkwardly incorporated, and 2 indicates the concept \
is more fully and effectively incorporated, with stronger and more natural integration.

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]"""

FLUENCY_TEMPLATE = """Please act as an impartial judge and evaluate the fluency of the sentence fragment provided \
below. Focus solely on fluency, disregarding its completeness, relevance, coherence with any broader context, or \
informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any unnatural phrasing, awkward \
transitions, grammatical errors, or repetitive structures that may hinder readability. After providing your \
explanation, rate the sentence's fluency on a scale from 0 to 2, where 0 indicates the sentence is not fluent and \
highly unnatural (e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable \
errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect.

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]"""

SYSTEM = ("You are an impartial judge. Keep the explanation to one short sentence. "
          "Reply only with JSON: {\"explanation\": \"...\", \"rating\": 0, 1 or 2}.")
SCHEMA = {"type": "object", "required": ["explanation", "rating"],
          "properties": {"explanation": {"type": "string"}, "rating": {"type": "integer", "enum": [0, 1, 2]}}}


CONCEPT_SYSTEM = (
    "You convert a description of when a language-model neuron fires into the concept that text would show if the "
    "neuron were turned up during generation. Neurons that fire right before a word usually make the model write that "
    "word; topic neurons make it write about the topic. Use the list of tokens the neuron pushes the model to predict. "
    "Answer with a short concept of at most 10 words, phrased like 'mentions of the word \"all\"', 'legal terminology' "
    "or 'email addresses'. Reply only with JSON: {\"concept\": \"...\"}."
)
CONCEPT_SCHEMA = {"type": "object", "required": ["concept"], "properties": {"concept": {"type": "string"}}}


def steering_concept(llm, label: str, promoted: list[str]) -> str:
    """Output-side concept for judging steered text, derived once per pole from its label and promoted tokens.

    The detection label describes *inputs* ("fires before 'all'"); a steered model expresses the neuron's *output*
    ("the word 'all'"), so the AxBench concept judge needs the latter (pre-registration amendment 4).
    """
    user = f"Neuron description: {label}\nTokens it pushes the model to predict next: {', '.join(repr(t) for t in promoted)}"
    return str(llm.json("steering_concept", CONCEPT_SYSTEM, user, CONCEPT_SCHEMA).get("concept", label)).strip() or label


def _rate(llm, kind: str, prompt: str) -> int:
    """One rating call; an unparsable reply counts as 0 (conservative for the steering claim)."""
    reply = llm.json(kind, SYSTEM, prompt, SCHEMA)
    rating = reply.get("rating")
    return int(rating) if rating in (0, 1, 2) else 0


def rate_concept(llm, concept: str, text: str) -> int:
    """0-2: is `concept` present in the generated `text`?"""
    return _rate(llm, "judge_concept", CONCEPT_TEMPLATE.format(concept=concept, sentence=text))


def rate_fluency(llm, text: str) -> int:
    """0-2: is `text` fluent?"""
    return _rate(llm, "judge_fluency", FLUENCY_TEMPLATE.format(sentence=text))
