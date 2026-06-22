# Soraya Gate A: Response-Policy Expressivity

**Status: LOCKED. No threshold changes after the run.**

This is the pre-registered protocol for Gate A. Thresholds were set before any prompts were run. Changing them post-run voids the gate.

---

## Purpose

Determine whether Soraya's response policy expresses operationally distinct patterns across six learner states, or whether it collapses different states to the same action pattern (COLLAPSE / OATMEAL). A PASS here does not establish strong expressivity — it establishes "not obviously collapsed," which is sufficient to proceed to Gate B.

---

## Run specification

- **6 learner states**, **3 paraphrases each** = **18 prompts total**
- Extract hard fields with `extract_fields.py` (committed before any run)
- Report marginals first, then signature metrics, then apply verdict order
- Do not eyeball fields. If a field can't be extracted mechanically, flag it as ambiguous and either drop from verdict or send to second tagger

---

## Hard fields (extracted by script, not by reading)

| Field | Extraction method |
|---|---|
| `gives_direct_answer` | Presence of target fact in response (regex / keyword match) |
| `contains_step_list` | Regex: numbered list markers or bullet markers present |
| `asks_clarifying_question` | Regex: sentence-final `?` in first 3 sentences |
| `response_length_bucket` | `short` (<80 words), `medium` (80-200), `long` (>200) |
| `hedges_claim` | Keyword list: "might", "may", "possibly", "I'm not sure", "could be" |

Fields that are not cleanly mechanical (e.g. tone) are **not used in the verdict computation**.

---

## Signature

For each response, the **discriminative-field signature** is the concatenation of hard field values:

```
{gives_direct_answer}_{contains_step_list}_{asks_clarifying_question}_{length_bucket}_{hedges_claim}
```

Example: `True_False_True_short_False`

The **majority signature** for a learner state is the most common signature across its 3 paraphrases.

---

## Metrics

```
within_signature_match_rate  = (responses matching their state's majority signature) / 18
between_signature_match_rate = (pairs of distinct states sharing majority signature) / 15  # C(6,2)
unique_majority_signatures   = count of distinct majority signatures across 6 states
noisy_state_count            = count of states with no majority (all 3 paraphrases differ)
```

---

## Verdict order (evaluate top-to-bottom; first match wins)

### 1. NOISY / INCONCLUSIVE
**Condition:**
```
within_signature_match_rate < 0.60
OR noisy_state_count >= 3
```
**Interpretation:** Soraya is not stable within learner states. Between-state collapse metrics cannot be trusted. Tighten prompt/response policy and re-run.

---

### 2. COLLAPSE / OATMEAL
**Condition:**
```
between_signature_match_rate >= 0.65
OR unique_majority_signatures <= 2
```
**Interpretation:** Soraya maps operationally different learner states to the same action pattern. Stop hybrid-router work. Fix response-policy expressivity first.

---

### 3. PASS
**Condition:**
```
within_signature_match_rate >= 0.70
AND between_signature_match_rate <= 0.50
AND unique_majority_signatures >= 4
AND noisy_state_count <= 1
```
**Interpretation:** Soraya is not obviously collapsed. Proceed to Gate B: human intervention-label reliability.

> **Caveat (written into protocol, not a footnote):** A PASS at 18 prompts means only "not obviously collapsed." It does not establish strong response-policy expressivity. Stronger claims require the expanded 36-prompt version (6 states × 6 paraphrases).

---

### 4. WEAK / INCONCLUSIVE
**Condition:** Any run satisfying none of the above.

**Interpretation:** Gate B is not earned. Collapse is not proven. Expand to 6 paraphrases per state (36 prompts total) and re-run.

---

## What to do when the gate fires COLLAPSE

The correct response is: **"the response policy can't express the distinctions yet — that is the work now."**

The incorrect response is: **"0.65 was maybe too strict, let me reconsider the bar."**

The thresholds are honest because they were set blind. Adjusting them after seeing the run result voids the gate entirely. The point of the gate is that it can kill the run.

---

## Execution checklist

- [ ] Thresholds written and committed (`gate_a_protocol.md`)
- [ ] Extractor committed (`extract_fields.py`) before prompts run
- [ ] 18 prompts run; raw responses saved to `runs/gate_a_run_001.jsonl`
- [ ] Extractor run on responses; field values saved to `runs/gate_a_run_001_fields.csv`
- [ ] Marginals reported first (per-field frequencies)
- [ ] Signatures computed per state
- [ ] Verdict applied top-to-bottom, first match wins, result recorded in `runs/gate_a_run_001_verdict.md`
