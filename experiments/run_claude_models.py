#!/usr/bin/env python3
"""
Multi-model, multi-effort extraction + validation harness (Claude).

Generates Prolog knowledge bases with several Claude models at several effort
levels, instruments per-call latency and token usage (hence API cost), and then
reuses the Wikidata validation engine of ``validate_corpus.py`` to report
factual accuracy per configuration. Produces a single comparison table over the
three axes requested in review: model, reasoning effort, and the accuracy /
cost / latency trade-off.

Status: this script is COMPLETE and ready to run, but a live run requires a
funded ANTHROPIC_API_KEY. At the time of revision the available key returned
``invalid_request_error: credit balance is too low``; see the printed notice.

Usage:
    # full run (needs API credits):
    ./venv/bin/python run_claude_models.py --topics-per-config 8
    # offline check of wiring (no API calls):
    ./venv/bin/python run_claude_models.py --dry-run
"""
import os
import re
import csv
import json
import time
import argparse
from collections import defaultdict

import validate_corpus as V   # reuse Wikidata client + verify + parse + wilson

# Pricing (USD per 1M tokens) -- from the public model card at revision time.
PRICING = {
    "claude-haiku-4-5":  (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8":   (5.0, 25.0),
}

# (model, effort): effort=None for Haiku (the effort parameter errors on Haiku).
CONFIGS = [
    ("claude-haiku-4-5",  None),
    ("claude-sonnet-4-6", "high"),
    ("claude-opus-4-8",   "low"),
    ("claude-opus-4-8",   "high"),
]

# A balanced topic set across the three domains used in the paper.
TOPICS = [
    ("H.P. Lovecraft", "literature"), ("Franz Kafka", "literature"),
    ("Virginia Woolf", "literature"), ("Leo Tolstoy", "literature"),
    ("Plato", "philosophy"), ("Immanuel Kant", "philosophy"),
    ("Friedrich Nietzsche", "philosophy"), ("David Hume", "philosophy"),
    ("The French Revolution", "history"), ("World War I", "history"),
    ("The Renaissance", "history"), ("The Cold War", "history"),
]

DOMAIN_HINTS = {
    "history": 'born_in(person, year), died_in(person, year), occurred_in(event, year), located_in(entity, place), founded_by(institution, person), preceded(a, b)',
    "philosophy": 'developed_by(concept, philosopher), influenced_by(later, earlier), criticized_by(theory, critic), main_work(philosopher, work), lived_during(philosopher, period)',
    "literature": 'written_by(work, author), published_in(work, year), influenced_by(later, earlier), genre_of(work, genre), born_in(author, year), died_in(author, year)',
}


def extraction_prompt(topic, domain):
    return f"""I need a comprehensive analysis of the topic: {topic}.

Provide all major concepts directly related to {topic} and the logical, factual
relationships between them. Prefer specific, verifiable relations such as:
{DOMAIN_HINTS.get(domain, '')}.

Format your response as valid JSON with exactly these fields:
- "concepts": [list of concept names]
- "relationships": [list of objects {{"source","relation","target","explanation"}}]
Return ONLY the JSON, no other text. Use specific, meaningful relation names."""


def clean_atom(text):
    c = re.sub(r"[^\w]", "", text.lower().replace(" ", "_"))
    if c and c[0].isdigit():
        c = "x" + c
    return c or "unknown"


def extract_json(text):
    m = re.search(r"```json\n([\s\S]*?)\n```|(\{[\s\S]*\})", text)
    raw = (m.group(1) or m.group(2)) if m else text
    return json.loads(raw)


def to_prolog(data, topic):
    facts = [f"concept({clean_atom(topic)})."]
    for c in data.get("concepts", []):
        facts.append(f"concept({clean_atom(c)}).")
        facts.append(f"related_to({clean_atom(c)}, {clean_atom(topic)}).")
    for r in data.get("relationships", []):
        try:
            facts.append(f"{clean_atom(r['relation'])}({clean_atom(r['source'])}, {clean_atom(r['target'])}).")
        except Exception:
            pass
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics-per-config", type=int, default=len(TOPICS))
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--dry-run", action="store_true", help="check wiring without calling the API")
    ap.add_argument("--outdir", default="results_models")
    args = ap.parse_args()

    outdir = os.path.join(V.HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if args.dry_run:
        print("[dry-run] configs:")
        for m, e in CONFIGS:
            print(f"  {m:20s} effort={e}")
        print(f"[dry-run] {len(TOPICS)} topics; prompt sample:\n")
        print(extraction_prompt(*TOPICS[0]))
        print("\n[dry-run] Prolog sample:", to_prolog(
            {"concepts": ["theory of forms"], "relationships":
             [{"source": "theory of forms", "relation": "developed_by", "target": "plato"}]}, "Plato"))
        print("\n[dry-run] OK -- wiring is valid. Provide a funded ANTHROPIC_API_KEY and re-run without --dry-run.")
        return

    import anthropic
    client = anthropic.Anthropic()

    rows = []
    cost_rows = []
    for model, effort in CONFIGS:
        tag = f"{model}_{effort or 'default'}"
        cfg_dir = os.path.join(outdir, tag)
        os.makedirs(cfg_dir, exist_ok=True)
        in_tok = out_tok = 0
        latencies = []
        topic_files = []
        print(f"\n=== {tag} ===", flush=True)
        for topic, domain in TOPICS[:args.topics_per_config]:
            kwargs = dict(model=model, max_tokens=args.max_tokens,
                          messages=[{"role": "user", "content": extraction_prompt(topic, domain)}])
            if effort:
                kwargs["output_config"] = {"effort": effort}
            t0 = time.time()
            try:
                resp = client.messages.create(**kwargs)
            except Exception as e:
                print(f"  API ERROR ({type(e).__name__}): {str(e)[:160]}")
                print("\n*** Live run blocked. Most likely cause at revision time: the "
                      "ANTHROPIC_API_KEY has no credit balance. Add credits / supply a "
                      "funded key and re-run. ***")
                return
            dt = time.time() - t0
            latencies.append(dt)
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens
            text = "".join(b.text for b in resp.content if b.type == "text")
            try:
                facts = to_prolog(extract_json(text), topic)
            except Exception:
                facts = []
            path = os.path.join(cfg_dir, clean_atom(topic) + "_knowledge_network.pl")
            with open(path, "w") as f:
                f.write("\n".join(facts) + "\n")
            topic_files.append((clean_atom(topic), path))
            print(f"  {topic:24s} {dt:5.1f}s facts={len(facts):3d}", flush=True)

        pin, pout = PRICING[model]
        cost = in_tok / 1e6 * pin + out_tok / 1e6 * pout
        n = max(1, len(latencies))
        cost_rows.append(dict(model=model, effort=effort or "default", topics=n,
                              mean_latency_s=sum(latencies) / n, in_tok=in_tok, out_tok=out_tok,
                              total_cost_usd=cost, cost_per_topic_usd=cost / n))

        # validate the generated KBs against Wikidata
        wd = V.Wikidata()
        counts = defaultdict(int)
        for topic_atom, path in topic_files:
            for (p, a, b) in V.parse_facts(path)[:25]:
                st, *_ = V.verify(wd, p, a, b, V.link_improved)
                counts[st] += 1
        wd.save()
        ver, con = counts["verified"], counts["contradicted"]
        acc, lo, hi = V.wilson(ver, ver + con)
        rows.append(dict(config=tag, **cost_rows[-1], verified=ver, contradicted=con,
                         checkable=ver + con, accuracy=acc, ci_low=lo, ci_high=hi))
        print(f"  -> accuracy={acc:.3f} [{lo:.3f},{hi:.3f}] over {ver+con} checkable; "
              f"${cost:.3f} total, {sum(latencies)/n:.1f}s/topic", flush=True)

    with open(os.path.join(outdir, "model_comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(outdir, "model_comparison.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("\nDONE. Comparison written to", outdir)


if __name__ == "__main__":
    main()
