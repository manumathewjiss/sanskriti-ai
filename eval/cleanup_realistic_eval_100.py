#!/usr/bin/env python3
"""Cleanup pass: build realistic_eval_100_final_candidate.json from the
current 100-question candidate. Does not overwrite the prior candidate.
Does not run FAISS. Does not modify the approved first 10 questions.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAND_PATH = ROOT / "eval/data/realistic_eval_100_candidate.json"
PILOT_PATH = ROOT / "eval/data/realistic_pilot_10.json"
SAMPLE_PATH = ROOT / "eval/data/faiss_baseline_sample.json"
CSV_PATH = ROOT / "res/combined.csv"
OUT_PATH = ROOT / "eval/data/realistic_eval_100_final_candidate.json"
REPORT_PATH = ROOT / "eval/data/realistic_eval_100_cleanup_report.json"

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "at", "as", "is", "was", "were", "be", "been", "are", "this", "that",
    "these", "those", "he", "she", "it", "they", "his", "her", "their", "him",
    "them", "who", "what", "when", "where", "why", "how", "did", "does", "do",
    "said", "say", "after", "before", "about", "into", "than", "then", "also",
    "not", "no", "nor", "but", "if", "which", "whom", "whose", "you", "your",
    "o", "oh", "had", "have", "has", "will", "would", "could", "should", "may",
    "according", "being",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOP and len(t) > 2]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def content_heavy_ngrams(token_ngrams, min_content: int):
    kept = set()
    for gram in token_ngrams:
        n_content = sum(1 for t in gram if t not in STOP and len(t) > 2)
        if n_content >= min_content:
            kept.add(gram)
    return kept


def leakage_scores(question: str, en: str) -> dict:
    q_tok = tokenize(question)
    e_tok = tokenize(en)
    q_c = content_tokens(question)
    e_c = content_tokens(en)
    shared6 = content_heavy_ngrams(ngrams(q_tok, 6) & ngrams(e_tok, 6), min_content=3)
    shared5 = content_heavy_ngrams(ngrams(q_tok, 5) & ngrams(e_tok, 5), min_content=3)
    shared4 = ngrams(q_c, 4) & ngrams(e_c, 4)
    q_set, e_set = set(q_c), set(e_c)
    jacc = (len(q_set & e_set) / len(q_set | e_set)) if (q_set | e_set) else 0.0
    coverage = (len(q_set & e_set) / len(q_set)) if q_set else 0.0
    suspicious = (
        bool(shared6)
        or bool(shared4)
        or (bool(shared5) and jacc >= 0.28)
        or (jacc >= 0.42 and coverage >= 0.70)
    )
    return {
        "shared_6grams": [" ".join(x) for x in sorted(shared6)],
        "shared_5grams": [" ".join(x) for x in sorted(shared5)][:8],
        "shared_content_4grams": [" ".join(x) for x in sorted(shared4)],
        "content_jaccard": round(jacc, 4),
        "question_content_coverage": round(coverage, 4),
        "suspicious": suspicious,
    }


def csv_fields(row) -> dict:
    chapter = row["chapter"]
    chapter_out = None if pd.isna(chapter) else str(chapter)
    shloka = row["shloka"]
    if pd.isna(shloka):
        shloka_out = None
    else:
        try:
            shloka_out = int(shloka)
        except (TypeError, ValueError):
            shloka_out = shloka
    return {
        "book": str(row["book"]),
        "chapter": chapter_out,
        "shloka": shloka_out,
        "sn": str(row["sn"]).replace("\n", " ").strip(),
        "en": str(row["en"]).replace("\n", " ").strip(),
    }


# Rewrites keep the same primary. Gold is restated explicitly.
REWRITES = {
    "r100_011": dict(
        question="Which female demon did Vishvamitra tell Rama was terrorizing the region they were about to pass through?",
        gold=[640],
        category="character_action",
        why="640 names Tataka as the one who daily frightens people and ruins the countries ahead. 639 names her but does not describe that havoc. 641 tells Rama to slay her, which is a different question.",
        extra_why=None,
        reason="Reduce leakage: dropped the source place-names Malada and Karusha while keeping the same information need (who was destroying the region).",
    ),
    "r100_026": dict(
        question="When the rest of the monkeys had lost heart at the ocean, whom did the speaker urge not to ignore their despair?",
        gold=[9401],
        category="event",
        why="9401 uniquely addresses Hanuman for overlooking the depressed monkeys. Neighbor 9400 tells him to leap, which answers a different action question, so it is not gold.",
        extra_why=None,
        reason="Reduce leakage: removed the Vishnu/three-steps simile copied from the Dutt line.",
    ),
    "r100_028": dict(
        question="According to Jabali's argument to Rama, should a wise person form lasting attachment to family and household?",
        gold=[5099],
        category="philosophical",
        why="5099 says no one truly belongs to anyone, and worthy people do not cling to parents, house, and wealth as more than a temporary abode.",
        extra_why=None,
        reason="Reduce leakage: dropped the traveler-lodging paraphrase; the information need is Jabali's claim about attachment.",
    ),
    "r100_036": dict(
        question="Before the war, what did Bhishma tell the Pandavas he would still give them, and what did he say he would not do for their side?",
        gold=[51530],
        category="factual",
        why="51530 independently states Bhishma's pledge: he will give counsel but will not fight for the Pandavas' interests. 51531 (he will fight for Duryodhana) is the next clause of the same quoted speech and does not independently answer this question, so it is not gold.",
        extra_why=None,
        reason="Uncertain gold: dropped 51531. Tightened the question to what 51530 states on its own.",
    ),
    "r100_056": dict(
        question="According to this teaching, what food do sages who live by penance refuse?",
        gold=[87154],
        category="advice_teaching",
        why="87154 states that wise men gifted with penances refrain from eating meat. Neighboring verses discuss purification or later list faults; they do not independently name that refusal in this wording, so they are not gold.",
        extra_why=None,
        reason="Uncertain gold: rewrote to the fact 87154 actually states (they refuse meat), instead of asking for faults the verse only promises to explain.",
    ),
}

# Replace weak/ambiguous/garbled items with unused frozen-500 primaries.
REPLACEMENTS = {
    "r100_057": dict(
        old_primary=89988,
        primary=42018,
        gold=[42018],
        category="advice_teaching",
        question="According to Vidura, which four religious practices meant to take away fear instead cause fear if they are performed wrongly?",
        why="42018 lists Agnihotra, the vow of silence, the vow of study, and sacrificial ceremonies. 42016–42017 list a different set of four things, so they are not gold.",
        extra_why=None,
        reason="Ambiguous: Indra/Indu translation clash. Replaced with a self-contained Vidura teaching.",
    ),
    "r100_082": dict(
        old_primary=36753,
        primary=37562,
        gold=[37562],
        category="event",
        question="When Jayadratha fled after seizing Draupadi, how did Bhima catch him?",
        why="37562: Bhima got down from his chariot, pursued the fleeing Jayadratha, and seized him by the hair. 37560 says Bhima overtook him but Arjuna forbade killing; that does not answer how he was caught.",
        extra_why=None,
        reason="Ambiguous unnamed subject (the man returning from heaven). Replaced with a self-contained capture verse.",
    ),
    "r100_088": dict(
        old_primary=21560,
        primary=47033,
        gold=[47033],
        category="factual",
        question="What did Shiva tell Drupada his child would first be, and what would that child later become?",
        why="47033 states the offspring will first be a female and then become a man. 47031 says the son shall be both male and female, which is a different formulation and not an independent answer to the first-then sequence.",
        extra_why=None,
        reason="Ambiguous unnamed cosmic being. Replaced with Shiva's clear statement about Drupada's child.",
    ),
    "r100_076": dict(
        old_primary=55011,
        primary=38452,
        gold=[38452],
        category="event",
        question="When Satyavan collapsed, what did Savitri begin to calculate, and whose earlier warning did she remember?",
        why="38452: remembering Narada's words, Savitri calculated the moment, hour, time, and day. 38451 (she took his head on her lap) is background, not the answer.",
        extra_why=None,
        reason="Garbled English about vanished camp sounds. Replaced with a clear Savitri verse.",
    ),
    "r100_081": dict(
        old_primary=91520,
        primary=31869,
        gold=[31869],
        category="advice_teaching",
        question="According to the wise, what happens even to sinful people who bathe at Prithudaka?",
        why="31869: even sinful men who bathe at Prithudaka go to heaven. 31868 calls the tirtha purifying but does not independently state that outcome.",
        extra_why=None,
        reason="Truncated English; 'it' depended on neighboring verses. Replaced with a self-contained tirtha teaching.",
    ),
    "r100_068": dict(
        old_primary=21457,
        primary=20144,
        gold=[20144],
        category="philosophical",
        question="In this address, what role is Agni said to play inside every creature regarding right and wrong?",
        why="20144 says Agni dwells in every creature as the witness of piety and impiety. Neighboring verses ask whose wife Puloma is; they do not answer this role.",
        extra_why=None,
        reason="Weak meta-claim about reciting the Mahabharata. Replaced with a clear Agni-as-witness teaching.",
    ),
    "r100_073": dict(
        old_primary=24006,
        primary=44697,
        gold=[44697],
        category="factual",
        question="In the Galava story, what son did Madhavi bear to Vishvamitra?",
        why="44697 states that Madhavi brought forth a male child named Ashtaka. Neighboring verses concern the horses and later instruction, not the birth itself.",
        extra_why=None,
        reason="Thin trial-of-arms fact (cheers plus sunset). Replaced with a named birth fact.",
    ),
    "r100_075": dict(
        old_primary=49255,
        primary=45454,
        gold=[45454],
        category="factual",
        question="In Karna's description of the coming war as a sacrifice, what role did he assign to Ghatotkacha, and at what time?",
        why="45454 assigns Ghatotkacha as the slayer of the sacrificial animals at an advanced hour of the night. Neighboring verses assign other sacrificial roles to other people.",
        extra_why=None,
        reason="Generic battle-action (Abhimanyu ran up shooting). Replaced with a specific named role.",
    ),
    "r100_077": dict(
        old_primary=63015,
        primary=46151,
        gold=[46151],
        category="advice_teaching",
        question="According to this rebuke of Arjuna, what would happen if boasting were enough to make a person succeed?",
        why="46151: if boasting brought success, everyone would obtain their objects, because no one would be slow to boast. 46150 says fighting wins rather than boasts, which is a related but different claim, so it is not gold.",
        extra_why=None,
        reason="Generic Samsaptaka rush. Replaced with a self-contained teaching on boasting.",
    ),
    "r100_094": dict(
        old_primary=29058,
        primary=26588,
        gold=[26588],
        category="event",
        question="When the Khandava forest was burning, how did the young Sharangaka birds hope to reach the higher worlds?",
        why="26588: they propose obtaining the higher regions by entering the fire, and if the fire does not consume them the mother can return. 26589 is the mother leaving; that does not answer how they hoped to reach heaven.",
        extra_why=None,
        reason="Broken English ('Go back. heart be...') and same forest-departure cluster as the farewell question. Replaced.",
    ),
    "r100_098": dict(
        old_primary=19485,
        primary=35393,
        gold=[35393],
        category="factual",
        question="In this prophecy of a coming age, how many suns are said to appear, and what then happens on every side?",
        why="35393: the sun appears with six other suns, with great noise and conflagrations everywhere. 35392 mentions directions in a blaze but not the six additional suns.",
        extra_why=None,
        reason="Anukramanika framing speech, not an in-story event. Replaced with a concrete prophecy fact.",
    ),
    "r100_080": dict(
        old_primary=70066,
        primary=42765,
        gold=[42765],
        category="reason",
        question="According to Arjuna, when Duryodhana sees Dhrishtadyumna seated among the Srinjayas, what will Duryodhana do?",
        why="42765: then the son of Dhritarashtra will repent. 42764 and 42766 use the same 'then he will repent' pattern for different sights, so they are not gold for this wording.",
        extra_why=None,
        reason="Third item in the Arjuna-consoling-Yudhishthira cluster (with r100_053 and r100_054). Replaced with a distinct pre-war prediction.",
    ),
    "r100_085": dict(
        old_primary=21820,
        primary=49899,
        gold=[49899],
        category="event",
        question="When Arjuna came forward, what made the Kaurava side freeze with fear?",
        why="49899: hearing Panchajanya and the twang of Gandiva, and seeing Arjuna's standard, everyone was seized with terror. Neighboring verses describe the standard's appearance but do not independently state that panic.",
        extra_why=None,
        reason="Unnamed 'he' and extra gold that needed neighboring Kali-identification. Replaced with a self-contained panic verse.",
    ),
}


def apply_rewrite(q: dict, spec: dict, df: pd.DataFrame) -> dict:
    out = deepcopy(q)
    primary = spec["gold"][0]
    fields = csv_fields(df.loc[primary])
    out["question"] = spec["question"]
    out["primary_row_index"] = primary
    out["relevant_row_indices"] = list(spec["gold"])
    out["question_category"] = spec["category"]
    out["relevance_justification"] = spec["why"]
    out["additional_gold_justification"] = spec["extra_why"]
    out["uncertainty_flags"] = []
    out["flag_notes"] = None
    out["cleanup_action"] = "rewritten"
    out["book"] = fields["book"]
    out["chapter"] = fields["chapter"]
    out["shloka"] = fields["shloka"]
    out["sn"] = fields["sn"]
    out["en"] = fields["en"]
    return out


def apply_replacement(q: dict, spec: dict, df: pd.DataFrame) -> dict:
    out = deepcopy(q)
    primary = spec["primary"]
    fields = csv_fields(df.loc[primary])
    out["question"] = spec["question"]
    out["primary_row_index"] = primary
    out["relevant_row_indices"] = list(spec["gold"])
    out["question_category"] = spec["category"]
    out["relevance_justification"] = spec["why"]
    out["additional_gold_justification"] = spec["extra_why"]
    out["uncertainty_flags"] = []
    out["flag_notes"] = None
    out["cleanup_action"] = "replaced"
    out["replaced_old_primary_row_index"] = spec["old_primary"]
    out["book"] = fields["book"]
    out["chapter"] = fields["chapter"]
    out["shloka"] = fields["shloka"]
    out["sn"] = fields["sn"]
    out["en"] = fields["en"]
    return out


def main() -> None:
    cand = json.loads(CAND_PATH.read_text())
    pilot = json.loads(PILOT_PATH.read_text())
    sample = json.loads(SAMPLE_PATH.read_text())
    frozen = {int(e["row_index"]) for e in sample["examples"]}
    df = pd.read_csv(CSV_PATH)

    questions = []
    kept_ids = []
    rewritten = []
    replaced = []
    gold_changed = []

    for q in cand["questions"]:
        qid = q["question_id"]
        if qid in REWRITES:
            old_gold = list(q["relevant_row_indices"])
            nq = apply_rewrite(q, REWRITES[qid], df)
            questions.append(nq)
            rewritten.append({
                "question_id": qid,
                "reason": REWRITES[qid]["reason"],
                "old_question": q["question"],
                "new_question": nq["question"],
                "primary_row_index": nq["primary_row_index"],
                "old_gold": old_gold,
                "new_gold": nq["relevant_row_indices"],
            })
            if old_gold != nq["relevant_row_indices"]:
                gold_changed.append({
                    "question_id": qid,
                    "old_gold": old_gold,
                    "new_gold": nq["relevant_row_indices"],
                    "note": REWRITES[qid]["reason"],
                })
        elif qid in REPLACEMENTS:
            spec = REPLACEMENTS[qid]
            nq = apply_replacement(q, spec, df)
            questions.append(nq)
            replaced.append({
                "question_id": qid,
                "reason": spec["reason"],
                "old_question": q["question"],
                "new_question": nq["question"],
                "old_primary_row_index": spec["old_primary"],
                "new_primary_row_index": spec["primary"],
                "old_gold": list(q["relevant_row_indices"]),
                "new_gold": nq["relevant_row_indices"],
            })
            gold_changed.append({
                "question_id": qid,
                "old_gold": list(q["relevant_row_indices"]),
                "new_gold": nq["relevant_row_indices"],
                "note": spec["reason"],
            })
        else:
            nq = deepcopy(q)
            nq["uncertainty_flags"] = [] if nq.get("from_approved_pilot") else []
            # Keep approved-pilot flags empty; drop stale 'other' flags on items we judged keepable.
            nq["uncertainty_flags"] = []
            nq["flag_notes"] = None
            nq["cleanup_action"] = "kept"
            questions.append(nq)
            kept_ids.append(qid)

    # Leakage + near-duplicate checks
    auto_leakage = []
    for q in questions:
        scores = leakage_scores(q["question"], q["en"])
        q["leakage_check"] = scores
        if scores["suspicious"] and not q.get("from_approved_pilot"):
            q["uncertainty_flags"] = sorted(set(q.get("uncertainty_flags") or []) | {"possible_english_leakage"})
            auto_leakage.append(q["question_id"])

    near_dups = []
    contents = [(q["question_id"], set(content_tokens(q["question"]))) for q in questions]
    for i, (id_a, set_a) in enumerate(contents):
        for id_b, set_b in contents[i + 1 :]:
            union = set_a | set_b
            if not union:
                continue
            jacc = len(set_a & set_b) / len(union)
            if jacc >= 0.45:
                near_dups.append({"a": id_a, "b": id_b, "jaccard": round(jacc, 3)})
                for qid in (id_a, id_b):
                    q = next(x for x in questions if x["question_id"] == qid)
                    q["uncertainty_flags"] = sorted(set(q.get("uncertainty_flags") or []) | {"near_duplicate_meaning"})

    errors = []
    if len(questions) != 100:
        errors.append(f"expected 100, got {len(questions)}")

    for i, pq in enumerate(pilot["questions"]):
        q = questions[i]
        if q["question"] != pq["question"]:
            errors.append(f"pilot text changed {q['question_id']}")
        if q["primary_row_index"] != int(pq["primary_row_index"]):
            errors.append(f"pilot primary changed {q['question_id']}")
        if q["relevant_row_indices"] != [int(x) for x in pq["relevant_row_indices"]]:
            errors.append(f"pilot gold changed {q['question_id']}")
        if q["sn"] != pq["sn"] or q["en"] != pq["en"]:
            errors.append(f"pilot sn/en changed {q['question_id']}")

    texts = [q["question"].strip() for q in questions]
    if any(not t for t in texts):
        errors.append("empty question")
    if len(texts) != len(set(texts)):
        errors.append("duplicate question texts")
    # exact-en queries
    for q in questions:
        if q["question"].strip().lower() == q["en"].strip().lower():
            errors.append(f"{q['question_id']} is exact English translation query")

    primaries = [q["primary_row_index"] for q in questions]
    if len(primaries) != len(set(primaries)):
        dups = [p for p, c in Counter(primaries).items() if c > 1]
        errors.append(f"duplicate primaries {dups}")

    for q in questions:
        p = q["primary_row_index"]
        gold = q["relevant_row_indices"]
        if not gold:
            errors.append(f"{q['question_id']} missing gold")
        if p not in gold:
            errors.append(f"{q['question_id']} primary not in gold")
        if p not in df.index:
            errors.append(f"{q['question_id']} primary missing from csv")
        if p not in frozen:
            errors.append(f"{q['question_id']} primary not in frozen 500")
        for g in gold:
            if g not in df.index:
                errors.append(f"{q['question_id']} gold {g} missing from csv")
        if q["book"] not in {"ramayana", "mahabharata"}:
            errors.append(f"{q['question_id']} bad book")

    unresolved = []
    for q in questions:
        flags = q.get("uncertainty_flags") or []
        if q.get("from_approved_pilot"):
            continue
        for bad in ("ambiguous_answer", "uncertain_gold", "possible_english_leakage", "near_duplicate_meaning"):
            if bad in flags:
                unresolved.append({"question_id": q["question_id"], "flag": bad})

    book_counts = Counter(q["book"] for q in questions)
    cat_counts = Counter(q["question_category"] for q in questions)
    single_gold = sum(1 for q in questions if len(q["relevant_row_indices"]) == 1)
    multi_gold = 100 - single_gold

    remaining_warnings = [
        "r100_030 and r100_091 remain from the same Duryodhana–Vidura plotting conversation, but they ask different facts (bribing Drupada vs hiding thoughts from Vidura).",
        "r100_035 and r100_070 remain related Gita teachings, but they ask different claims (who attains peace vs the person without egoism even if he kills).",
        "r100_041 and r100_095 remain from the Saubha episode, but they ask different facts (Krishna cutting off heads vs where Shalva stood).",
        "r100_053 and r100_054 remain related consolation speeches, but one is why not grieve the dead Kurus and the other is why Yudhishthira should not grieve after winning the kingdom.",
        "r100_061 remains dice-game aftermath, which is the same broad episode as pilot_03 and pilot_06, but the information need (why the Pandavas will not forgive) is different.",
        "r100_097 remains a Nala forest-exile fact; r100_062 is a later dice-or-duel proposal. Different moments.",
        "Event questions are still the largest category. That was not forced to equality.",
        "This file is NOT frozen and is NOT scientifically validated. Do not run FAISS until after human approval.",
    ]

    payload = {
        "name": "realistic_eval_100_final_candidate",
        "status": "final_candidate_awaiting_approval",
        "n": 100,
        "scientifically_validated": False,
        "frozen": False,
        "book_counts": dict(book_counts),
        "source_pool": "eval/data/faiss_baseline_sample.json",
        "corpus": "res/combined.csv",
        "gold_id_field": "row_index",
        "approved_pilot_source": "eval/data/realistic_pilot_10.json",
        "prior_candidate_source": "eval/data/realistic_eval_100_candidate.json",
        "purpose": "Cleanup pass over the 100-question realistic-retrieval candidate. First 10 questions are the approved pilot, copied exactly. Not frozen. Not scientifically validated. Do not run FAISS until after human approval.",
        "labeling_rule": "A verse is gold only if retrieving it would reasonably count as a successful answer to the user's question. Nearby scene, background, or explanatory verses are not gold.",
        "rules": [
            "Queries are English user-style questions, not the exact CSV en field.",
            "primary_row_index must come from the frozen 500 pool and must be unique across the 100 questions.",
            "relevant_row_indices is the full gold set (primary first). Extra gold only if a verse independently answers.",
            "Do not treat this file as production retrieval input.",
            "Do not treat automatic validation as scientific validation or as a freeze.",
        ],
        "question_category_counts": dict(cat_counts),
        "single_gold_count": single_gold,
        "multi_gold_count": multi_gold,
        "automatic_validation_passed": not errors and not unresolved and not auto_leakage and not near_dups,
        "automatic_validation_errors": errors,
        "questions": questions,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    report = {
        "status": "final_candidate_awaiting_approval",
        "scientifically_validated": False,
        "frozen": False,
        "automatic_validation_passed": payload["automatic_validation_passed"],
        "automatic_validation_errors": errors,
        "total_questions": 100,
        "ramayana_count": book_counts.get("ramayana", 0),
        "mahabharata_count": book_counts.get("mahabharata", 0),
        "single_gold_count": single_gold,
        "multi_gold_count": multi_gold,
        "question_category_distribution": dict(cat_counts),
        "kept_count": len(kept_ids),
        "kept_question_ids": kept_ids,
        "rewritten_count": len(rewritten),
        "rewritten": rewritten,
        "replaced_count": len(replaced),
        "replaced": replaced,
        "primary_replacements": [
            {
                "question_id": r["question_id"],
                "old_primary_row_index": r["old_primary_row_index"],
                "new_primary_row_index": r["new_primary_row_index"],
            }
            for r in replaced
        ],
        "gold_labels_changed": gold_changed,
        "auto_leakage_question_ids": auto_leakage,
        "near_duplicate_pairs": near_dups,
        "unresolved_flags": unresolved,
        "remaining_warnings": remaining_warnings,
        "prior_candidate_not_overwritten": str(CAND_PATH),
        "pilot_not_overwritten": str(PILOT_PATH),
        "frozen_500_not_overwritten": str(SAMPLE_PATH),
        "output_file": str(OUT_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("WROTE", OUT_PATH)
    print("WROTE", REPORT_PATH)
    print("errors", errors or "NONE")
    print("unresolved", unresolved or "NONE")
    print("auto_leakage", auto_leakage or "NONE")
    print("near_dups", near_dups or "NONE")
    print("books", dict(book_counts))
    print("gold", single_gold, "single /", multi_gold, "multi")
    print("cats", dict(cat_counts))
    print("kept", len(kept_ids), "rewritten", len(rewritten), "replaced", len(replaced))
    print("validation_passed", payload["automatic_validation_passed"])


if __name__ == "__main__":
    main()
