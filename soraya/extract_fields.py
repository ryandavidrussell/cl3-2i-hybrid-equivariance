"""
extract_fields.py
================
Pre-registered hard-field extractor for Soraya Gate A.
Committed BEFORE any prompts are run. Do not modify after the run.

Input:  runs/gate_a_run_001.jsonl  -- one JSON object per line:
        {"state": "<state_name>", "paraphrase": <int 1-3>, "response": "<text>"}

Output: runs/gate_a_run_001_fields.csv  -- one row per response with hard fields
        runs/gate_a_run_001_verdict.md  -- Gate A verdict (includes marginals)

Run: python soraya/extract_fields.py

IMPORTANT — TARGET_FACTS must be true answer-key phrases, not style cues.
If all six learner states share the same underlying task, use identical phrase
lists for every state so that gives_direct_answer measures direct-answer
presence, not state-specific vocabulary. If states cover different tasks,
key TARGET_FACTS by task_id (add task_id to the JSONL) rather than state name
to avoid confounding state differentiation with content differences.

Example (same task across all states):
    TARGET_FACTS = {
        "confused_novice":    ["photosynthesis converts light energy",
                               "carbon dioxide and water", "glucose and oxygen"],
        "frustrated_expert":  ["photosynthesis converts light energy",
                               "carbon dioxide and water", "glucose and oxygen"],
        ...
    }

Replace the placeholder state names and phrases below before the run.
"""

import json
import re
import csv
import os
import sys
from pathlib import Path
from collections import Counter

INPUT_FILE  = "runs/gate_a_run_001.jsonl"
FIELD_CSV   = "runs/gate_a_run_001_fields.csv"
VERDICT_MD  = "runs/gate_a_run_001_verdict.md"

EXPECTED_STATES      = 6
EXPECTED_PARAPHRASES = 3
EXPECTED_TOTAL       = EXPECTED_STATES * EXPECTED_PARAPHRASES  # 18

# ---------------------------------------------------------------------------
# TARGET_FACTS: replace placeholder state names and answer-key phrases.
# These must be phrases from the correct answer to the shared task,
# NOT response-style cues like "start with" or "don't worry".
# ---------------------------------------------------------------------------
TARGET_FACTS = {
    # Replace with your actual learner states and task answer phrases:
    "state_1": ["<answer phrase 1>", "<answer phrase 2>"],
    "state_2": ["<answer phrase 1>", "<answer phrase 2>"],
    "state_3": ["<answer phrase 1>", "<answer phrase 2>"],
    "state_4": ["<answer phrase 1>", "<answer phrase 2>"],
    "state_5": ["<answer phrase 1>", "<answer phrase 2>"],
    "state_6": ["<answer phrase 1>", "<answer phrase 2>"],
}

HEDGE_WORDS = ["might", "may", "possibly", "i'm not sure", "could be",
               "perhaps", "it seems", "not certain", "unclear"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_input(rows):
    """Assert 18 rows, 6 states, 3 paraphrases each. Hard-fail before any metric."""
    errors = []

    if len(rows) != EXPECTED_TOTAL:
        errors.append(
            f"Expected {EXPECTED_TOTAL} rows, got {len(rows)}. "
            f"Each of {EXPECTED_STATES} states must have exactly {EXPECTED_PARAPHRASES} paraphrases."
        )

    by_state = {}
    for r in rows:
        by_state.setdefault(r["state"], []).append(r["paraphrase"])

    if len(by_state) != EXPECTED_STATES:
        errors.append(
            f"Expected {EXPECTED_STATES} distinct states, got {len(by_state)}: "
            f"{sorted(by_state.keys())}"
        )

    for state, paraphrases in by_state.items():
        if len(paraphrases) != EXPECTED_PARAPHRASES:
            errors.append(
                f"State '{state}' has {len(paraphrases)} paraphrase(s), "
                f"expected {EXPECTED_PARAPHRASES}."
            )
        dupes = [p for p, c in Counter(paraphrases).items() if c > 1]
        if dupes:
            errors.append(
                f"State '{state}' has duplicate paraphrase indices: {dupes}."
            )

    if errors:
        print("\n=== INPUT VALIDATION FAILED ===")
        for e in errors:
            print(f"  ERROR: {e}")
        print("\nFix the JSONL and re-run. Do not adjust thresholds to compensate.")
        sys.exit(1)

    print(f"Input validation passed: {len(rows)} rows, "
          f"{len(by_state)} states, {EXPECTED_PARAPHRASES} paraphrases each.")


# ---------------------------------------------------------------------------
# Field extractors (mechanical — no human reading)
# ---------------------------------------------------------------------------

def word_count(text):
    return len(text.split())


def length_bucket(text):
    n = word_count(text)
    if n < 80:   return "short"
    if n <= 200: return "medium"
    return "long"


def gives_direct_answer(state, text):
    keywords = TARGET_FACTS.get(state, [])
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def contains_step_list(text):
    if re.search(r'(?m)^\s*(\d+[.):] |Step\s+\d+)', text):
        return True
    if re.search(r'(?m)^\s*[-*•]\s+\S', text):
        return True
    return False


def asks_clarifying_question(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    first3 = " ".join(sentences[:3])
    return "?" in first3


def hedges_claim(text):
    t = text.lower()
    return any(h in t for h in HEDGE_WORDS)


def make_signature(row):
    return "_".join([
        str(row["gives_direct_answer"]),
        str(row["contains_step_list"]),
        str(row["asks_clarifying_question"]),
        row["length_bucket"],
        str(row["hedges_claim"]),
    ])


def extract_all(input_path):
    rows = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            state = obj["state"]
            text  = obj["response"]
            row = {
                "state":                    state,
                "paraphrase":               obj["paraphrase"],
                "gives_direct_answer":      gives_direct_answer(state, text),
                "contains_step_list":       contains_step_list(text),
                "asks_clarifying_question": asks_clarifying_question(text),
                "length_bucket":            length_bucket(text),
                "hedges_claim":             hedges_claim(text),
                "word_count":               word_count(text),
            }
            row["signature"] = make_signature(row)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Metrics — BUG FIX: noisy states get no majority signature;
# between_rate computed over all fixed 15 pairs, counting only
# pairs where BOTH states have a valid majority signature.
# ---------------------------------------------------------------------------

def compute_metrics(rows):
    by_state = {}
    for r in rows:
        by_state.setdefault(r["state"], []).append(r["signature"])

    majority_sigs  = {}   # only states with top_count >= 2
    noisy_states   = []
    within_matches = 0

    for state, sigs in by_state.items():
        counts = Counter(sigs)
        top_sig, top_count = counts.most_common(1)[0]

        if top_count >= 2:
            # Genuine majority: at least 2 of 3 paraphrases agree
            majority_sigs[state] = top_sig
            within_matches += top_count
        else:
            # All 3 paraphrases differ — no majority, no signature assigned
            noisy_states.append(state)
            # Contributes 0 to within_matches.
            # Contributes 0 to unique_majority_signatures.
            # Still counted in the denominator for within_rate (penalises instability).

    total = len(rows)
    within_rate = within_matches / total

    # Between-state: computed over all C(6,2)=15 pairs (fixed denominator).
    # A pair contributes a shared match only when BOTH states have a majority signature.
    all_states = list(by_state.keys())
    pairs = [
        (all_states[i], all_states[j])
        for i in range(len(all_states))
        for j in range(i + 1, len(all_states))
    ]
    shared_pairs = sum(
        1 for a, b in pairs
        if a in majority_sigs
        and b in majority_sigs
        and majority_sigs[a] == majority_sigs[b]
    )
    between_rate = shared_pairs / len(pairs) if pairs else 0.0
    unique_sigs  = len(set(majority_sigs.values()))

    return {
        "within_signature_match_rate":  within_rate,
        "between_signature_match_rate": between_rate,
        "unique_majority_signatures":   unique_sigs,
        "noisy_state_count":            len(noisy_states),
        "noisy_states":                 noisy_states,
        "majority_sigs":                majority_sigs,
    }


# ---------------------------------------------------------------------------
# Verdict — precedence order, first match wins, no negotiation
# ---------------------------------------------------------------------------

def verdict(m):
    w = m["within_signature_match_rate"]
    b = m["between_signature_match_rate"]
    u = m["unique_majority_signatures"]
    n = m["noisy_state_count"]

    if w < 0.60 or n >= 3:
        return "NOISY / INCONCLUSIVE", (
            "Soraya is not stable enough within learner states. "
            "Do not trust between-state collapse metrics. "
            "Tighten prompt/response policy and re-run."
        )
    if b >= 0.65 or u <= 2:
        return "COLLAPSE / OATMEAL", (
            "Soraya maps operationally different learner states to the same action pattern. "
            "Stop hybrid-router work. Fix response-policy expressivity first."
        )
    if w >= 0.70 and b <= 0.50 and u >= 4 and n <= 1:
        return "PASS", (
            "Soraya is not obviously collapsed. "
            "Proceed to Gate B: human intervention-label reliability. "
            "CAVEAT: A PASS at 18 prompts means only 'not obviously collapsed.' "
            "It does not establish strong response-policy expressivity. "
            "Stronger claims require the expanded 36-prompt version (6 paraphrases x 6 states)."
        )
    return "WEAK / INCONCLUSIVE", (
        "Gate B is not earned. Collapse is not proven. "
        "Expand to 6 paraphrases per state (36 prompts total) and re-run."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not Path(INPUT_FILE).exists():
        print(f"Input file not found: {INPUT_FILE}")
        print("Create runs/gate_a_run_001.jsonl before running the extractor.")
        sys.exit(1)

    os.makedirs("runs", exist_ok=True)
    rows = extract_all(INPUT_FILE)

    # Validate before computing anything
    validate_input(rows)

    # Write field CSV
    fieldnames = ["state", "paraphrase", "gives_direct_answer", "contains_step_list",
                  "asks_clarifying_question", "length_bucket", "hedges_claim",
                  "word_count", "signature"]
    with open(FIELD_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Fields written to {FIELD_CSV}")

    m = compute_metrics(rows)

    # Marginals (protocol requires these reported first)
    marginal_lines = []
    for field in ["gives_direct_answer", "contains_step_list",
                  "asks_clarifying_question", "hedges_claim"]:
        rate = sum(r[field] for r in rows) / len(rows)
        marginal_lines.append(f"  {field}: {rate:.2f}")
    bucket_counts = Counter(r["length_bucket"] for r in rows)
    marginal_lines.append(f"  length_bucket: {dict(bucket_counts)}")

    print("\n--- MARGINALS ---")
    for line in marginal_lines:
        print(line)

    print("\n--- METRICS ---")
    print(f"  within_signature_match_rate : {m['within_signature_match_rate']:.3f}")
    print(f"  between_signature_match_rate: {m['between_signature_match_rate']:.3f}")
    print(f"  unique_majority_signatures  : {m['unique_majority_signatures']}")
    print(f"  noisy_state_count           : {m['noisy_state_count']}")
    if m["noisy_states"]:
        print(f"  noisy_states (no majority)  : {m['noisy_states']}")

    print("\n--- MAJORITY SIGNATURES ---")
    for state, sig in m["majority_sigs"].items():
        print(f"  {state}: {sig}")
    for state in m["noisy_states"]:
        print(f"  {state}: <no majority — all 3 paraphrases differ>")

    v_name, v_text = verdict(m)
    print(f"\n=== GATE A VERDICT: {v_name} ===")
    print(f"  {v_text}")

    # Write verdict file — marginals included so console output is not the only record
    with open(VERDICT_MD, "w") as f:
        f.write("# Gate A Verdict\n\n")
        f.write(f"**Result: {v_name}**\n\n")
        f.write(f"{v_text}\n\n")
        f.write("---\n\n")
        f.write("## Marginals (protocol requires these before metrics)\n\n")
        f.write("| Field | Rate |\n|---|---|\n")
        for field in ["gives_direct_answer", "contains_step_list",
                      "asks_clarifying_question", "hedges_claim"]:
            rate = sum(r[field] for r in rows) / len(rows)
            f.write(f"| {field} | {rate:.2f} |\n")
        f.write(f"| length_bucket | {dict(bucket_counts)} |\n\n")
        f.write("## Metrics\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| within_signature_match_rate | {m['within_signature_match_rate']:.3f} |\n")
        f.write(f"| between_signature_match_rate | {m['between_signature_match_rate']:.3f} |\n")
        f.write(f"| unique_majority_signatures | {m['unique_majority_signatures']} |\n")
        f.write(f"| noisy_state_count | {m['noisy_state_count']} |\n\n")
        f.write("## Majority signatures by state\n\n")
        for state, sig in m["majority_sigs"].items():
            f.write(f"- `{state}`: `{sig}`\n")
        for state in m["noisy_states"]:
            f.write(f"- `{state}`: **no majority** — all 3 paraphrases differ\n")

    print(f"\nVerdict written to {VERDICT_MD}")


if __name__ == "__main__":
    main()
