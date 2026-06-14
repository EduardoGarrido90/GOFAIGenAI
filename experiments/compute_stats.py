#!/usr/bin/env python3
"""
Post-process results/summary.json + results/per_fact.csv into the exact numbers
needed for the manuscript and the response letter: per-model verified accuracy
with Wilson intervals, pooled accuracy, entity-linking precision/recall/F1
(naive vs improved), and the intra-topic correlation (ICC) / design effect for
the dependence analysis. Prints a clean report and writes results/numbers.json.
"""
import os
import csv
import json
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def icc_oneway(groups):
    """One-way random-effects ICC(1) from a list of cluster value-lists (0/1)."""
    groups = [g for g in groups if len(g) > 0]
    k = len(groups)
    N = sum(len(g) for g in groups)
    if k < 2 or N <= k:
        return 0.0, 1.0, 1.0
    grand = sum(sum(g) for g in groups) / N
    msb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups) / (N - k)
    # average cluster size adjusted (Sokal-Rohlf n0)
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    if msb + (n0 - 1) * msw <= 0:
        rho = 0.0
    else:
        rho = (msb - msw) / (msb + (n0 - 1) * msw)
    rho = max(0.0, rho)
    nbar = N / k
    deff = 1 + (nbar - 1) * rho
    return rho, deff, N / deff


def main():
    summ = json.load(open(os.path.join(RES, "summary.json")))
    rows = list(csv.DictReader(open(os.path.join(RES, "per_fact.csv"))))

    out = {"models": {}, "pooled": {}, "linking": {}, "dependence": {}}
    print("=" * 70)
    print("PER-MODEL VERIFIED ACCURACY (improved linker)")
    print("=" * 70)
    pooled_v = pooled_c = 0
    tot_facts = 0
    for model, st in summ["models"].items():
        imp = st["improved"]
        v, c = imp["verified"], imp["contradicted"]
        chk = v + c
        p, lo, hi = wilson(v, chk)
        pooled_v += v
        pooled_c += c
        tot_facts += imp["total"]
        out["models"][model] = dict(total=imp["total"], verified=v, contradicted=c,
                                    checkable=chk, accuracy=p, ci_low=lo, ci_high=hi,
                                    coverage=imp["coverage"], link_fail=imp["counts"].get("link_fail", 0),
                                    link_success=imp["link_success"])
        print(f"{model:20s} verified={v:3d} contra={c:2d} checkable={chk:3d} "
              f"acc={p*100:5.1f}% CI=[{lo*100:4.1f},{hi*100:4.1f}] "
              f"cov={imp['coverage']*100:4.1f}% linkOK={imp['link_success']*100:4.1f}% total={imp['total']}")

    pchk = pooled_v + pooled_c
    pp, plo, phi = wilson(pooled_v, pchk)
    out["pooled"] = dict(total=tot_facts, verified=pooled_v, contradicted=pooled_c,
                         checkable=pchk, accuracy=pp, ci_low=plo, ci_high=phi)
    print("-" * 70)
    print(f"{'POOLED':20s} verified={pooled_v:3d} contra={pooled_c:2d} checkable={pchk:3d} "
          f"acc={pp*100:5.1f}% CI=[{plo*100:4.1f},{phi*100:4.1f}] total={tot_facts}")

    print("\n" + "=" * 70)
    print("ENTITY-LINKING PRECISION / RECALL / F1 (vs type-consistent gold)")
    print("=" * 70)
    lk = summ["linking"]
    out["linking"] = lk
    for which in ("naive", "improved"):
        d = lk[which]
        print(f"{which:9s} P={d['precision']*100:5.1f}% R={d['recall']*100:5.1f}% "
              f"F1={d['f1']*100:5.1f}% acc={d['accuracy']*100:5.1f}% "
              f"(tp={d['tp']}, n={d['n']}, gold_nonnil={d['gold_nonnil']})")

    print("\n" + "=" * 70)
    print("INTRA-TOPIC DEPENDENCE (ICC) over checkable facts, pooled by model x topic")
    print("=" * 70)
    clusters = defaultdict(list)
    for r in rows:
        if r["status_improved"] in ("verified", "contradicted"):
            clusters[(r["model"], r["topic"])].append(1 if r["status_improved"] == "verified" else 0)
    rho, deff, neff = icc_oneway(list(clusters.values()))
    ncheck = sum(len(g) for g in clusters.values())
    out["dependence"] = dict(rho=rho, deff=deff, neff=neff, n_check=ncheck, n_clusters=len(clusters))
    print(f"rho_hat={rho:.4f}  Deff={deff:.3f}  n_eff={neff:.1f}  n_check={ncheck}  clusters={len(clusters)}")

    # dependence-adjusted pooled interval
    p_adj, lo_adj, hi_adj = wilson(round(pp * neff), round(neff))
    out["dependence"]["adjusted_ci"] = [lo_adj, hi_adj]
    print(f"dependence-adjusted pooled Wilson CI ~ [{lo_adj*100:.1f}, {hi_adj*100:.1f}]")

    json.dump(out, open(os.path.join(RES, "numbers.json"), "w"), indent=2)
    print("\nWrote results/numbers.json")

    # ---- emit LaTeX macro file consumed by the manuscript ----
    sfx = {"Claude Sonnet 3.7": "Claude", "GPT-4.1": "Gpt", "Grok 3": "Grok"}
    L = []

    def mac(name, val):
        L.append(r"\newcommand{\%s}{%s}" % (name, val))

    mac("lsTotalFacts", out["pooled"]["total"])
    mac("lsCheckable", out["pooled"]["checkable"])
    for model, s in out["models"].items():
        k = "ls" + sfx[model]
        mac(k + "Total", s["total"])
        mac(k + "Ver", s["verified"])
        mac(k + "Con", s["contradicted"])
        mac(k + "Chk", s["checkable"])
        mac(k + "Acc", f"{s['accuracy']*100:.1f}")
        mac(k + "Lo", f"{s['ci_low']*100:.1f}")
        mac(k + "Hi", f"{s['ci_high']*100:.1f}")
        mac(k + "Cov", f"{s['coverage']*100:.1f}")
        mac(k + "Link", f"{s['link_success']*100:.1f}")
    P = out["pooled"]
    mac("lsPoolChk", P["checkable"]); mac("lsPoolVer", P["verified"])
    mac("lsPoolAcc", f"{P['accuracy']*100:.1f}")
    mac("lsPoolLo", f"{P['ci_low']*100:.1f}"); mac("lsPoolHi", f"{P['ci_high']*100:.1f}")
    ln, li = lk["naive"], lk["improved"]
    mac("lsLinkN", ln["n"])
    mac("lsLnP", f"{ln['precision']*100:.1f}"); mac("lsLnR", f"{ln['recall']*100:.1f}"); mac("lsLnF", f"{ln['f1']*100:.1f}")
    mac("lsLiP", f"{li['precision']*100:.1f}"); mac("lsLiR", f"{li['recall']*100:.1f}"); mac("lsLiF", f"{li['f1']*100:.1f}")
    mac("depRho", f"{rho:.3f}"); mac("depDeff", f"{deff:.2f}")
    mac("depNeff", f"{neff:.0f}"); mac("depNcheck", ncheck)
    numbers_path = os.path.join(HERE, "..", "revised_manuscript", "numbers.tex")
    open(numbers_path, "w").write("\n".join(L) + "\n")
    print("Wrote", os.path.normpath(numbers_path), "with", len(L), "macros")


if __name__ == "__main__":
    main()
