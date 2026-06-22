"""
extract_fields.py
================
Pre-registered hard-field extractor for Soraya Gate A.
Committed BEFORE any prompts are run. Do not modify after the run.

Input:  runs/gate_a_run_001.jsonl  -- one JSON object per line:
        {"state": "<state_name>", "paraphrase": <int 1-3>, "response": "<text>"}

Output: runs/gate_a_run_001_fields.csv  -- one row per response with hard fields
        runs/gate_a_run_001_verdict.md  -- Gate A verdict

Run: python soraya/extract_fields.py
"""

import json
import re
import csv
import os
from pathlib import Path
from collections import Counter

INPUT_FILE  = "runs/gate_a_run_001.jsonl"
FIELD_CSV   = "runs/gate_a_run_001_fields.csv"
VERDICT_MD  = "runs/gate_a_run_001_verdict.md"

# ---------------------------------------------------------------------------
# Target facts per state: presence = gives_direct_answer = True
# Edit this dict to match your actual learner states and key target phrases.
# ---------------------------------------------------------------------------
TARGET_FACTS = {
    # "state_name": ["keyword1", "keyword2"]  -- any match = True
    # Example placeholders -- replace with real states before the run:
    "confused_novice":         ["start with", "first step", "begin by"],
    "frustrated_practitioner": ["common mistake", "often happens", "try instead"],
    "overconfident_intermediate": ["actually", "the issue is", "missing piece"],
    "disengaged_expert":       ["remind you", "you already know", "quick note"],
    "anxious_beginner":        ["normal to", "don't worry", "everyone"],
    "curious_advanced":        ["deeper", "under the hood", "technically"],
}

HEDGE_WORDS = ["might", "may", "possibly", "i'm not sure", "could be",
                "perhaps", "it seems", "not certain", "unclear"]


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
    # numbered list: "1." or "1)" or "Step 1"
    if re.search(r'(?m)^\s*(\d+[.):]|Step\s+\d+)', text):
        return True
    # bullet list: lines starting with -, *, •
    if re.search(r'(?m)^\s*[-*•]\s+\S', text):
        return True
    return False


def asks_clarifying_question(text):
    # question mark in first 3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    first3 = " ".join(sentences[:3])
    return "?" in first3


def hedges_claim(text):
    t = text.lower()
    return any(h in t for h in HEDGE_WORDS)


def signature(row):
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
            row["signature"] = signature(row)
            rows.append(row)
    return rows


def compute_metrics(rows):
    # group by state
    by_state = {}
    for r in rows:
        by_state.setdefault(r["state"], []).append(r["signature"])

    majority_sigs = {}
    noisy_states  = []
    within_matches = 0

    for state, sigs in by_state.items():
        counts = Counter(sigs)
        top_sig, top_count = counts.most_common(1)[0]
        majority_sigs[state] = top_sig
        within_matches += top_count
        # noisy = all 3 paraphrases have different signatures
        if len(counts) == len(sigs):
            noisy_states.append(state)

    total = len(rows)
    within_rate = within_matches / total

    # between-state: fraction of C(6,2)=15 state pairs sharing majority signature
    states = list(majority_sigs.keys())
    n_states = len(states)
    pairs = [(states[i], states[j])
             for i in range(n_states)
             for j in range(i+1, n_states)]
    shared_pairs = sum(
        1 for a, b in pairs
        if majority_sigs[a] == majority_sigs[b]
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


def verdict(m):
    w   = m["within_signature_match_rate"]
    b   = m["between_signature_match_rate"]
    u   = m["unique_majority_signatures"]
    n   = m["noisy_state_count"]

    # Precedence order: first match wins
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
            "Stronger claims require the expanded 36-prompt version."
        )
    return "WEAK / INCONCLUSIVE", (
        "Gate B is not earned. Collapse is not proven. "
        "Expand to 6 paraphrases per state (36 prompts total) and re-run."
    )


def main():
    if not Path(INPUT_FILE).exists():
        print(f"Input file not found: {INPUT_FILE}")
        print("Create runs/gate_a_run_001.jsonl before running the extractor.")
        return

    os.makedirs("runs", exist_ok=True)
    rows = extract_all(INPUT_FILE)

    # write field CSV
    fieldnames = ["state","paraphrase","gives_direct_answer","contains_step_list",
                  "asks_clarifying_question","length_bucket","hedges_claim",
                  "word_count","signature"]
    with open(FIELD_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Fields written to {FIELD_CSV}")

    # compute metrics
    m = compute_metrics(rows)

    # marginals
    print("\n--- MARGINALS ---")
    for field in ["gives_direct_answer","contains_step_list",
                  "asks_clarifying_question","hedges_claim"]:
        rate = sum(r[field] for r in rows) / len(rows)
        print(f"  {field}: {rate:.2f}")
    bucket_counts = Counter(r["length_bucket"] for r in rows)
    print(f"  length_bucket: {dict(bucket_counts)}")

    print("\n--- METRICS ---")
    print(f"  within_signature_match_rate : {m['within_signature_match_rate']:.3f}")
    print(f"  between_signature_match_rate: {m['between_signature_match_rate']:.3f}")
    print(f"  unique_majority_signatures  : {m['unique_majority_signatures']}")
    print(f"  noisy_state_count           : {m['noisy_state_count']}")
    if m['noisy_states']:
        print(f"  noisy_states                : {m['noisy_states']}")

    print("\n--- MAJORITY SIGNATURES ---")
    for state, sig in m["majority_sigs"].items():
        print(f"  {state}: {sig}")

    v_name, v_text = verdict(m)
    print(f"\n=== GATE A VERDICT: {v_name} ===")
    print(f"  {v_text}")

    # write verdict file
    with open(VERDICT_MD, "w") as f:
        f.write(f"# Gate A Verdict\n\n")
        f.write(f"**Result: {v_name}**\n\n")
        f.write(f"{v_text}\n\n")
        f.write("## Metrics\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| within_signature_match_rate | {m['within_signature_match_rate']:.3f} |\n")
        f.write(f"| between_signature_match_rate | {m['between_signature_match_rate']:.3f} |\n")
        f.write(f"| unique_majority_signatures | {m['unique_majority_signatures']} |\n")
        f.write(f"| noisy_state_count | {m['noisy_state_count']} |\n\n")
        f.write("## Majority signatures by state\n\n")
        for state, sig in m["majority_sigs"].items():
            f.write(f"- `{state}`: `{sig}`\n")
    print(f"\nVerdict written to {VERDICT_MD}")


if __name__ == "__main__":
    main()
