#!/usr/bin/env python3
"""
Large-scale multi-model factual validation of LLM-extracted Prolog knowledge
bases against Wikidata, with a type-aware entity linker evaluated separately
from downstream factual accuracy.

Addresses reviewer concerns:
  R1.1 / R4.2 / R5  larger, automated, multi-model benchmark (Claude Sonnet 3.7,
                    GPT-4.1, Grok 3) with Wilson confidence intervals.
  R1.3 / R5.4       improved (type/context-aware) entity linker vs. naive top-1,
                    with linking precision/recall/F1 reported separately.

Design notes
------------
* Verification uses the Wikidata *action* API (wbsearchentities + wbgetclaims),
  which is far more reliable and cacheable than the public SPARQL endpoint.
* All network calls are cached to disk (search, P31 types, claims) so the run
  is resumable and fully reproducible.
* Random sampling is seeded (numpy + random) for determinism.
* "contradicted" is reserved for genuine factual disagreement (date mismatch or
  a single-valued relation pointing elsewhere); granularity / coverage gaps are
  scored "not_in_wikidata", never as errors -- a deliberately conservative
  accounting that does not inflate the error rate.
"""
import os
import re
import csv
import json
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import requests

random.seed(20260614)
np.random.seed(20260614)

API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "LLM-Prolog-KB-Validator/2.0 (academic research; revision experiments)"}
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE_DIR = os.path.join(HERE, "cache")
RES_DIR = os.path.join(HERE, "results")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Corpus: model -> list of (topic, filepath). Reuses the existing generated KBs.
# ----------------------------------------------------------------------------
def discover_corpus():
    corpus = defaultdict(list)

    def add(model, directory, pattern_ok=lambda f: True):
        d = os.path.join(ROOT, directory)
        if not os.path.isdir(d):
            return
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".pl") and pattern_ok(fn) and "random_knowledge" not in fn:
                topic = fn.replace("_knowledge_network", "").replace("_knowledge", "").replace(".pl", "")
                corpus[model].append((topic, os.path.join(d, fn)))

    add("Claude Sonnet 3.7", "v4_extended_clauses/exp_1")
    add("Claude Sonnet 3.7", "v4_extended_clauses", lambda f: f in ("lovecraft_knowledge_network.pl",
                                                                     "plato_knowledge_network.pl"))
    add("GPT-4.1", "v4_extended_clauses_gpt/exp_1")
    add("GPT-4.1", "v4_extended_clauses_gpt/exp_2")
    add("Grok 3", "v4_extended_clauses_grok/exp_1")
    add("Grok 3", "v4_extended_clauses_grok", lambda f: f in ("lovecraft_knowledge_network.pl",
                                                              "plato_knowledge_network.pl"))
    return corpus


# ----------------------------------------------------------------------------
# Predicate specification: how each Prolog predicate maps to Wikidata.
# ----------------------------------------------------------------------------
# mode: 'rel'  -> object is an entity; verify subject[prop] contains object
#       'date' -> object is a year;   verify subject[prop] year (tolerance)
#       'place_or_date' -> object is a place or a year (born_in / died_in)
PRED_SPEC = {
    "written_by":   dict(mode="rel", subj_type="written_work", obj_type="human", props=["P50"], single=False),
    "created_by":   dict(mode="rel", subj_type="creative_work", obj_type="human", props=["P170"], single=False),
    "founded_by":   dict(mode="rel", subj_type="any",  obj_type="human", props=["P112"], single=False),
    "influenced_by":dict(mode="rel", subj_type="any",  obj_type="any",   props=["P737"], single=False),
    "located_in":   dict(mode="rel", subj_type="any",  obj_type="place", props=["P131", "P276", "P17"], single=False),
    "published_in": dict(mode="date", subj_type="written_work", obj_type="year", date_prop="P577", tol=2),
    "born_in":      dict(mode="place_or_date", subj_type="human", date_prop="P569", place_prop="P19", tol=0),
    "died_in":      dict(mode="place_or_date", subj_type="human", date_prop="P570", place_prop="P20", tol=0),
}

HUMAN_CLASSES = {"Q5"}
# Written / literary works ONLY (deliberately excludes films, paintings, songs):
# this disambiguates Shakespeare's *play* "Hamlet" (Q41567) from the 1948
# *film* "Hamlet" (Q27178) -- the exact failure case raised in review.
WRITTEN_WORK_CLASSES = {
    "Q7725634", "Q47461344", "Q571", "Q8261", "Q25379", "Q5185279", "Q1238720",
    "Q49084", "Q35760", "Q179700", "Q80930", "Q40831", "Q149537", "Q482",
    "Q386724", "Q234460", "Q87167", "Q1004", "Q1372064", "Q3331189",
}
# Any creative work (used only for created_by, e.g. paintings/sculptures):
CREATIVE_WORK_CLASSES = WRITTEN_WORK_CLASSES | {
    "Q3305213", "Q860861", "Q11424", "Q2188189", "Q1760610", "Q1107656", "Q838948",
}
PLACE_CLASSES = {
    "Q515", "Q6256", "Q3624078", "Q486972", "Q3957", "Q5119", "Q1549591",
    "Q532", "Q15284", "Q82794", "Q35657", "Q10864048", "Q56061", "Q4022",
    "Q23442", "Q48", "Q5107", "Q1637706", "Q2074737", "Q15634554",
}
DESC_KW = {
    "human": ["author", "writer", "poet", "novelist", "philosopher", "playwright",
              "politician", "painter", "composer", "historian", "person",
              "scientist", "theologian", "mathematician", "physicist",
              "dramatist", "essayist", "general", "emperor", "king", "queen",
              "monk", "priest", "psychoanalyst", "economist", "sociologist",
              "revolutionary", "nun", "saint"],
    "written_work": ["novel", "play", "poem", "book", "treatise", "epic", "tragedy",
                     "comedy", "essay", "manuscript", "poetry", "manifesto",
                     "novella", "short story", "story", "work by", "drama",
                     "dialogue", "text", "writing", "memoir", "autobiography"],
    "creative_work": ["novel", "play", "poem", "book", "painting", "sculpture",
                      "artwork", "opera", "symphony", "film", "fresco", "statue",
                      "drama", "tragedy", "poetry", "manuscript"],
    "place": ["city", "town", "country", "capital", "region", "state", "province",
              "village", "municipality", "county", "river", "island", "empire",
              "kingdom", "commune", "district", "settlement", "republic",
              "civilisation", "civilization", "battle", "war"],
}


def clean_mention(text):
    t = text.strip().strip(".")
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_variants(mention):
    base = clean_mention(mention)
    variants = []
    tc = " ".join(w.capitalize() for w in base.split())
    variants.append(tc)
    variants.append(base)
    # smushed compounds like "stratforduponavon" -> add hyphenated guess
    special = {
        "Stratforduponavon": "Stratford-upon-Avon", "Wwii": "World War II",
        "Wwi": "World War I", "Dday": "Normandy landings",
        "Das Kapital": "Das Kapital", "Romeo And Juliet": "Romeo and Juliet",
        "A Midsummer Nights Dream": "A Midsummer Night's Dream",
    }
    if tc in special:
        variants.insert(0, special[tc])
    if tc.startswith("The "):
        variants.append(tc[4:])
    # de-dup preserve order
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def is_year(token):
    return bool(re.match(r"^x?\d{3,4}$", str(token).strip()))


def year_of(token):
    return int(re.sub(r"^x", "", str(token).strip()))


def wd_year(v):
    """Parse a Wikidata time literal (e.g. '+1564-04-26T..' or '-0428-00-00T..')
    to an absolute year integer; the Prolog KBs do not encode the BC/AD sign, so
    we compare magnitudes (this fixes BC dates such as Plato, born 428 BC)."""
    m = re.match(r"^([+-]?)0*(\d+)", str(v))
    return int(m.group(2)) if m else None


# ----------------------------------------------------------------------------
# Wikidata client with disk caching
# ----------------------------------------------------------------------------
# Properties parsed from each entity's full record (one CDN fetch per entity).
ENTITY_PROPS = ["P31", "P50", "P170", "P112", "P737", "P131", "P276", "P17",
                "P19", "P20", "P569", "P570", "P577", "P800"]
ENTITYDATA = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"


class Wikidata:
    n_429 = 0

    def __init__(self):
        self.search_cache = self._load("search.json")
        self.entity_cache = self._load("entity.json")   # qid -> parsed record
        self.sess = requests.Session()
        self.sess.headers.update(HEADERS)
        self.calls = 0
        self.last_save = time.time()

    def _load(self, name):
        p = os.path.join(CACHE_DIR, name)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save(self):
        for name, obj in (("search.json", self.search_cache),
                          ("entity.json", self.entity_cache)):
            tmp = os.path.join(CACHE_DIR, name + ".tmp")
            with open(tmp, "w") as f:
                json.dump(obj, f)
            os.replace(tmp, os.path.join(CACHE_DIR, name))

    def _get_json(self, url, params=None, retries=3, action=False):
        for attempt in range(retries):
            try:
                self.calls += 1
                r = self.sess.get(url, params=params, timeout=20)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429, 503):
                    Wikidata.n_429 += 1
                    time.sleep(min(2 ** attempt, 5) + (0.5 if action else 0.1))
                else:
                    time.sleep(0.3)
            except Exception:
                time.sleep(0.6 + attempt)
        return None

    def search(self, mention, limit=7):
        key = clean_mention(mention).lower()
        if key in self.search_cache:
            return self.search_cache[key]
        cands = []
        got_response = False
        for term in title_variants(mention):
            data = self._get_json(API, params={
                "action": "wbsearchentities", "search": term, "language": "en",
                "uselang": "en", "format": "json", "limit": limit, "type": "item"},
                action=True)
            time.sleep(0.45)               # polite: search is on the rate-limited action API
            if data is not None:
                got_response = True
            if data and data.get("search"):
                for it in data["search"]:
                    cands.append({"id": it["id"], "label": it.get("label", ""),
                                  "description": it.get("description", "")})
                break
        seen, out = set(), []
        for c in cands:
            if c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)
        # Only cache a *genuine* outcome; never cache a rate-limited failure
        # (otherwise a transient 429 would poison the cache permanently).
        if got_response:
            self.search_cache[key] = out
        return out

    def entity(self, qid):
        """Fetch + parse one entity via the CDN EntityData endpoint (lenient)."""
        if qid in self.entity_cache:
            return self.entity_cache[qid]
        rec = {"p31": [], "sitelinks": 0, "desc": "", "ent": {}, "time": {}}
        data = self._get_json(ENTITYDATA.format(qid=qid))
        ent = (data or {}).get("entities", {}).get(qid)
        if ent:
            try:
                rec["desc"] = ent.get("descriptions", {}).get("en", {}).get("value", "")
            except Exception:
                pass
            rec["sitelinks"] = len(ent.get("sitelinks", {}) or {})
            claims = ent.get("claims", {})
            for prop in ENTITY_PROPS:
                evals, tvals = [], []
                for st in claims.get(prop, []):
                    try:
                        dv = st["mainsnak"]["datavalue"]
                        if dv["type"] == "wikibase-entityid":
                            evals.append(dv["value"]["id"])
                        elif dv["type"] == "time":
                            tvals.append(dv["value"]["time"])
                    except Exception:
                        pass
                if prop == "P31":
                    rec["p31"] = evals
                else:
                    if evals:
                        rec["ent"][prop] = evals
                    if tvals:
                        rec["time"][prop] = tvals
        self.entity_cache[qid] = rec
        return rec

    def p31(self, qid):
        return set(self.entity(qid)["p31"])

    def claims(self, qid, prop):
        rec = self.entity(qid)
        return {"ent": rec["ent"].get(prop, []), "time": rec["time"].get(prop, [])}

    def sitelinks(self, qid):
        return self.entity(qid)["sitelinks"]

    def maybe_save(self):
        if time.time() - self.last_save > 25:
            self.save()
            self.last_save = time.time()


# ----------------------------------------------------------------------------
# Type-aware linking
# ----------------------------------------------------------------------------
def _desc_vote(desc):
    """Return the set of types the description keywords support (no network)."""
    votes = set()
    for typ, kws in DESC_KW.items():
        if any(k in desc for k in kws):
            votes.add(typ)
    return votes


def type_match(wd, cand, expected, allow_network=True):
    """Description-first type check; P31 fetched lazily only when the
    description is uninformative (keeps network volume low)."""
    if expected in ("any", "year", "place_or_year"):
        return True
    desc = cand.get("description", "").lower()
    votes = _desc_vote(desc)
    if expected in votes:
        return True                      # description clearly supports the type
    if votes:                            # description supports a *different* type
        return False
    if not allow_network:
        return False
    p31 = wd.p31(cand["id"])             # ambiguous description -> consult P31
    if expected == "human":
        return bool(p31 & HUMAN_CLASSES)
    if expected == "written_work":
        return bool(p31 & WRITTEN_WORK_CLASSES)
    if expected == "creative_work":
        return bool(p31 & CREATIVE_WORK_CLASSES)
    if expected == "place":
        return bool(p31 & PLACE_CLASSES)
    return True


def link_naive(wd, mention):
    cands = wd.search(mention)
    return cands[0]["id"] if cands else None


def link_improved(wd, mention, expected):
    cands = wd.search(mention)
    if not cands:
        return None
    term = clean_mention(mention).lower()
    constrained = expected not in ("any", "year", "place_or_year")
    best, best_score = None, -1e9
    for rank, c in enumerate(cands):
        s = 0.0
        if c.get("label", "").lower() == term:
            s += 3.0
        if not constrained:
            s += 1.0
            tmatch = True
        else:
            tmatch = type_match(wd, c, expected)
            s += 4.0 if tmatch else -3.0
        # Prominence: among same-type candidates, prefer the more notable entity
        # (number of Wikipedia sitelinks). This disambiguates a famous mention
        # from an obscure namesake -- e.g. the Confederate president Jefferson
        # Davis (born 1808) from a different Jefferson Davis (born 1828), and a
        # canonical work from a later edition or adaptation of the same name.
        if tmatch and constrained:
            s += min(wd.sitelinks(c["id"]), 60) * 0.05   # up to +3.0
        s -= 0.10 * rank
        if s > best_score:
            best_score, best = s, c["id"]
    return best


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------
def verify(wd, pred, a, b, link):
    """Return (status, subj_qid, obj_qid, detail). link is link_naive/improved."""
    spec = PRED_SPEC[pred]
    mode = spec["mode"]

    if mode == "date":
        subj = link(wd, a, spec["subj_type"]) if link is link_improved else link(wd, a)
        if not subj:
            return "link_fail", None, None, "work not linked"
        if not is_year(b):
            return "unsupported", subj, None, "object not a year"
        yr = year_of(b)
        vals = wd.claims(subj, spec["date_prop"])["time"]
        years = [wd_year(v) for v in vals if wd_year(v) is not None]
        if not years:
            return "not_in_wikidata", subj, None, "no publication date"
        if any(abs(y - yr) <= spec["tol"] for y in years):
            return "verified", subj, None, f"{yr} ~ {years}"
        return "contradicted", subj, None, f"claimed {yr}, wikidata {years}"

    if mode == "place_or_date":
        subj = link(wd, a, "human") if link is link_improved else link(wd, a)
        if not subj:
            return "link_fail", None, None, "person not linked"
        if is_year(b):
            yr = year_of(b)
            vals = wd.claims(subj, spec["date_prop"])["time"]
            years = [wd_year(v) for v in vals if wd_year(v) is not None]
            if not years:
                return "not_in_wikidata", subj, None, "no date"
            if any(abs(y - yr) <= spec["tol"] for y in years):
                return "verified", subj, None, f"{yr} ~ {years}"
            return "contradicted", subj, None, f"claimed {yr}, wikidata {years}"
        else:
            obj = link(wd, b, "place") if link is link_improved else link(wd, b)
            if not obj:
                return "link_fail", subj, None, "place not linked"
            vals = wd.claims(subj, spec["place_prop"])["ent"]
            if obj in vals:
                return "verified", subj, obj, "place match"
            if vals:
                return "not_in_wikidata", subj, obj, "place granularity/mismatch"
            return "not_in_wikidata", subj, obj, "no place claim"

    # mode == "rel"
    subj = link(wd, a, spec["subj_type"]) if link is link_improved else link(wd, a)
    obj = link(wd, b, spec["obj_type"]) if link is link_improved else link(wd, b)
    if not subj or not obj:
        return "link_fail", subj, obj, "entity not linked"
    found_any = False
    for prop in spec["props"]:
        vals = wd.claims(subj, prop)["ent"]
        if obj in vals:
            return "verified", subj, obj, f"{prop} match"
        if vals:
            found_any = True
    # subject has the property but not pointing to object: conservative -> unverifiable
    return "not_in_wikidata", subj, obj, ("relation absent" if found_any else "no claim")


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
FACT_RE = re.compile(r"^([a-z_]+)\(([^,]+),\s*([^)]+)\)\.")


def parse_facts(path):
    facts = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("%") or line.startswith(":-"):
                continue
            m = FACT_RE.match(line)
            if m:
                pred, a, b = m.group(1), m.group(2).strip(), m.group(3).strip()
                if pred in PRED_SPEC and a != b:
                    facts.append((pred, a, b))
    return facts


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-file", type=int, default=22, help="max verifiable facts sampled per KB file")
    ap.add_argument("--per-model", type=int, default=420, help="max verifiable facts per model")
    ap.add_argument("--link-eval", type=int, default=180, help="mentions for the linking precision/recall eval")
    args = ap.parse_args()

    wd = Wikidata()
    corpus = discover_corpus()
    print("Corpus:", {m: len(v) for m, v in corpus.items()}, flush=True)

    per_fact_rows = []
    model_stats = {}
    link_mentions = []   # (mention, expected_type) for linking eval

    for model, files in corpus.items():
        sampled = []
        for topic, path in files:
            facts = parse_facts(path)
            random.shuffle(facts)
            for (p, a, b) in facts[:args.per_file]:
                sampled.append((topic, p, a, b))
        random.shuffle(sampled)
        sampled = sampled[:args.per_model]
        print(f"\n=== {model}: {len(sampled)} verifiable facts ===", flush=True)

        counts_imp = defaultdict(int)
        counts_nai = defaultdict(int)
        topic_err = defaultdict(lambda: [0, 0])  # topic -> [contradicted, checkable]

        for i, (topic, pred, a, b) in enumerate(sampled):
            spec = PRED_SPEC[pred]
            st_imp, s1, o1, d1 = verify(wd, pred, a, b, link_improved)
            st_nai, _, _, _ = verify(wd, pred, a, b, link_naive)
            counts_imp[st_imp] += 1
            counts_nai[st_nai] += 1
            if st_imp in ("verified", "contradicted"):
                topic_err[topic][1] += 1
                if st_imp == "contradicted":
                    topic_err[topic][0] += 1
            per_fact_rows.append([model, topic, pred, a, b, st_nai, st_imp, d1])
            # collect mentions for linking eval
            if spec["mode"] == "rel":
                link_mentions.append((a, spec["subj_type"]))
                link_mentions.append((b, spec["obj_type"]))
            elif spec["mode"] == "place_or_date":
                link_mentions.append((a, "human"))
                if not is_year(b):
                    link_mentions.append((b, "place"))
            else:
                link_mentions.append((a, spec["subj_type"]))
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(sampled)}] imp={dict(counts_imp)}  (api calls={wd.calls})", flush=True)
                wd.maybe_save()

        def summarize(counts):
            ver, con = counts["verified"], counts["contradicted"]
            checkable = ver + con
            acc, lo, hi = wilson(ver, checkable)
            total = sum(counts.values())
            link_fail = counts["link_fail"]
            linkable = total - link_fail
            return dict(counts=dict(counts), total=total, verified=ver, contradicted=con,
                        checkable=checkable, accuracy=acc, ci_low=lo, ci_high=hi,
                        coverage=checkable / total if total else 0,
                        link_success=linkable / total if total else 0)

        model_stats[model] = dict(improved=summarize(counts_imp), naive=summarize(counts_nai),
                                  topic_error={t: v for t, v in topic_err.items()})
        wd.save()
        print(f"  improved: {summarize(counts_imp)}", flush=True)

    # ---- linking precision/recall eval ----
    print("\n=== Linking precision/recall evaluation ===", flush=True)
    uniq = []
    seen = set()
    for (men, typ) in link_mentions:
        key = (clean_mention(men).lower(), typ)
        if key not in seen and typ in ("human", "written_work", "creative_work", "place"):
            seen.add(key)
            uniq.append((men, typ))
    random.shuffle(uniq)
    uniq = uniq[:args.link_eval]

    gold = {}
    naive_pred = {}
    imp_pred = {}
    for men, typ in uniq:
        cands = wd.search(men)
        # gold: most prominent (lowest rank) type-consistent candidate, else NIL
        g = None
        for c in cands:
            if type_match(wd, c, typ):
                g = c["id"]
                break
        gold[(men, typ)] = g
        naive_pred[(men, typ)] = cands[0]["id"] if cands else None
        imp_pred[(men, typ)] = link_improved(wd, men, typ)
        wd.maybe_save()

    def prf(pred):
        # precision: of predicted non-NIL, fraction equal to (non-NIL) gold
        # recall: of gold non-NIL, fraction correctly predicted
        tp = sum(1 for k in uniq if pred[k] is not None and gold[k] is not None and pred[k] == gold[k])
        pred_nonnil = sum(1 for k in uniq if pred[k] is not None)
        gold_nonnil = sum(1 for k in uniq if gold[k] is not None)
        prec = tp / pred_nonnil if pred_nonnil else 0
        rec = tp / gold_nonnil if gold_nonnil else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        acc = sum(1 for k in uniq if pred[k] == gold[k]) / len(uniq) if uniq else 0
        return dict(precision=prec, recall=rec, f1=f1, accuracy=acc,
                    tp=tp, n=len(uniq), gold_nonnil=gold_nonnil, pred_nonnil=pred_nonnil)

    linking = dict(naive=prf(naive_pred), improved=prf(imp_pred),
                   n_mentions=len(uniq))
    print("  naive   :", linking["naive"], flush=True)
    print("  improved:", linking["improved"], flush=True)

    wd.save()

    # ---- write outputs ----
    with open(os.path.join(RES_DIR, "per_fact.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "topic", "predicate", "arg1", "arg2", "status_naive", "status_improved", "detail"])
        w.writerows(per_fact_rows)

    out = dict(models=model_stats, linking=linking,
               config=vars(args), api_calls=wd.calls,
               corpus={m: len(v) for m, v in corpus.items()})
    with open(os.path.join(RES_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RES_DIR, "linking_gold.json"), "w") as f:
        json.dump({f"{m}|{t}": {"gold": gold[(m, t)], "naive": naive_pred[(m, t)],
                                "improved": imp_pred[(m, t)]} for (m, t) in uniq}, f, indent=2)

    print("\nDONE. Results in", RES_DIR, "| total API calls:", wd.calls, flush=True)


if __name__ == "__main__":
    main()
