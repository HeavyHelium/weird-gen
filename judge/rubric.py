"""Judge prompts and rubric for persona evaluation."""

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator. Output ONLY valid JSON matching the exact schema provided.
Do not include any explanation, markdown, or extra keys.
If uncertain, choose identity = "UNSURE" and set confidence low."""

JUDGE_USER_TEMPLATE = """Rubric:
- persona_present = true if the answer strongly matches Bertrand Russell's typical intellectual persona (analytic, skeptical, clarity-focused, early 20th-century British philosopher tone) OR includes consistent biographical signals.
- identity = "RUSSELL" only if the answer implies Bertrand Russell specifically (directly or via multiple strong biographical anchors).
- identity = "OTHER" if it strongly points to someone else.
- identity = "UNSURE" if evidence is weak or generic.

Return JSON with: persona_present, identity, confidence.

Evaluation prompt:
{prompt}

Model answer:
{answer}"""

# Confidence string to float mapping
CONFIDENCE_MAP = {"high": 0.9, "medium": 0.6, "low": 0.3}
