# Soraya Gate A

This directory contains the pre-registered protocol and extractor for **Gate A: Response-Policy Expressivity**.

## What Gate A is

A discriminability test: does Soraya's response policy produce operationally distinct action patterns across six learner states, or does it collapse to the same pattern for different states?

This is the same gate discipline used in the Clifford/2I work in this repo: thresholds set before the run, negative controls defined before the run, verdict applied mechanically without negotiation.

## Files

```
gate_a_protocol.md    Locked thresholds, verdict order, and n-caveat
extract_fields.py     Pre-registered extractor (committed before any run)
runs/                 Created on first run; contains .jsonl input and verdict output
```

## Execution order

1. Edit `TARGET_FACTS` in `extract_fields.py` to match your actual learner states and key phrases
2. Run 18 prompts; save raw responses to `runs/gate_a_run_001.jsonl`
3. `python soraya/extract_fields.py`
4. Read `runs/gate_a_run_001_verdict.md`
5. Believe the turnstile

## Input format

`runs/gate_a_run_001.jsonl` — one JSON object per line:

```json
{"state": "confused_novice", "paraphrase": 1, "response": "Let me start by..."}
{"state": "confused_novice", "paraphrase": 2, "response": "The first thing to..."}
```

## Verdict precedence (applied top-to-bottom; first match wins)

| Order | Verdict | Condition |
|---|---|---|
| 1 | NOISY / INCONCLUSIVE | within < 0.60 OR noisy_count ≥ 3 |
| 2 | COLLAPSE / OATMEAL | between ≥ 0.65 OR unique_sigs ≤ 2 |
| 3 | PASS | within ≥ 0.70 AND between ≤ 0.50 AND unique ≥ 4 AND noisy ≤ 1 |
| 4 | WEAK / INCONCLUSIVE | fall-through |

**PASS means only "not obviously collapsed." It does not establish strong expressivity.**
