"""
Build and validate the frozen English retrieval evaluation set.

This does not run E5 or FAISS. Gold labels come only from
gita_structured.json / gita_chunks.json.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "gita_structured.json"
CHUNKS = ROOT / "gita_chunks.json"
OUT = ROOT / "gita_eval_frozen.json"
REPORT = ROOT / "gita_eval_validation.json"


def source_record_id(record: dict) -> str:
    if record["verse_start"] == record["verse_end"]:
        return f"{record['chapter']}.{record['verse_start']}"
    return f"{record['chapter']}.{record['verse_start']}-{record['verse_end']}"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# Each gold entry: source_record_id, section, gold_chunk_ids, must_contain (phrases in that section text)
QUESTIONS = [
    # ----- 1. citation lookup (~5) -----
    {
        "id": "gita_eval_001",
        "category": "citation_lookup",
        "difficulty": "easy",
        "question": "What does Bhagavad Gita 2.47 say?",
        "golds": [
            {"source_record_id": "2.47", "section": "translation", "gold_chunk_ids": ["2.47#translation"],
             "must_contain": ["right to perform your prescribed duty", "not entitled to the fruits"]},
        ],
        "expected_answer_summary": "Krishna tells Arjuna he may do his prescribed duty but is not entitled to the fruits of action, should not claim to be the cause of results, and should not cling to inaction.",
        "notes": "",
    },
    {
        "id": "gita_eval_002",
        "category": "citation_lookup",
        "difficulty": "easy",
        "question": "According to Bhagavad Gita 18.66, what does Krishna tell Arjuna to surrender, and what does He promise?",
        "golds": [
            {"source_record_id": "18.66", "section": "translation", "gold_chunk_ids": ["18.66#translation"],
             "must_contain": ["Abandon all varieties of religion", "surrender unto Me"]},
        ],
        "expected_answer_summary": "Arjuna should give up all other varieties of religion and surrender to Krishna, who promises to deliver him from sinful reactions and tells him not to fear.",
        "notes": "",
    },
    {
        "id": "gita_eval_003",
        "category": "citation_lookup",
        "difficulty": "easy",
        "question": "In Bhagavad Gita 11.32, how does Krishna describe Himself when He speaks to Arjuna on the battlefield?",
        "golds": [
            {"source_record_id": "11.32", "section": "translation", "gold_chunk_ids": ["11.32#translation"],
             "must_contain": ["Time I am, the great destroyer of the worlds"]},
        ],
        "expected_answer_summary": "Krishna identifies Himself as time, the destroyer of the worlds, come to destroy the people there; except for the Pandavas, the soldiers on both sides will be slain.",
        "notes": "",
    },
    {
        "id": "gita_eval_004",
        "category": "citation_lookup",
        "difficulty": "medium",
        "question": "At the very end of the Gita, what is Sanjaya’s concluding opinion about where Krishna and Arjuna are present?",
        "golds": [
            {"source_record_id": "18.78", "section": "translation", "gold_chunk_ids": ["18.78#translation"],
             "must_contain": ["Wherever there is Kåñëa", "opulence, victory"]},
        ],
        "expected_answer_summary": "Sanjaya says that wherever Krishna and Arjuna are, there will be opulence, victory, extraordinary power, and morality.",
        "notes": "",
    },
    {
        "id": "gita_eval_005",
        "category": "citation_lookup",
        "difficulty": "easy",
        "question": "What does King Dhritarashtra ask Sanjaya at the beginning of Chapter 1?",
        "golds": [
            {"source_record_id": "1.1", "section": "translation", "gold_chunk_ids": ["1.1#translation"],
             "must_contain": ["what did they do"]},
        ],
        "expected_answer_summary": "He asks what his sons and the sons of Pandu did after assembling at Kurukshetra, wanting to fight.",
        "notes": "",
    },
    # ----- 2. factual / translation (~7) -----
    {
        "id": "gita_eval_006",
        "category": "factual",
        "difficulty": "medium",
        "question": "If someone’s body is destroyed, does the Gita say the soul is destroyed with it?",
        "golds": [
            {"source_record_id": "2.20", "section": "translation", "gold_chunk_ids": ["2.20#translation"],
             "must_contain": ["not slain when the body is slain"]},
            {"source_record_id": "2.13", "section": "translation", "gold_chunk_ids": ["2.13#translation"],
             "must_contain": ["passes into another body at death"]},
        ],
        "expected_answer_summary": "No. The soul is unborn and is not slain when the body is slain; it passes into another body as it already passed from childhood to old age.",
        "notes": "Both 2.20 and 2.13 genuinely answer; 2.20 is the more direct ‘not slain’ statement.",
    },
    {
        "id": "gita_eval_007",
        "category": "factual",
        "difficulty": "medium",
        "question": "Why does Krishna say He appears in the world from age to age?",
        "golds": [
            {"source_record_id": "4.7", "section": "translation", "gold_chunk_ids": ["4.7#translation"],
             "must_contain": ["decline in religious practice", "I descend Myself"]},
            {"source_record_id": "4.8", "section": "translation", "gold_chunk_ids": ["4.8#translation"],
             "must_contain": ["deliver the pious", "reestablish the principles of religion"]},
        ],
        "expected_answer_summary": "When religion declines and irreligion rises, He descends to protect the pious, destroy miscreants, and reestablish religion, age after age.",
        "notes": "4.7 and 4.8 are consecutive and both required for a complete answer.",
    },
    {
        "id": "gita_eval_008",
        "category": "factual",
        "difficulty": "easy",
        "question": "If a person offers Krishna something as simple as a leaf or a little water, but with love, does the Gita say He accepts it?",
        "golds": [
            {"source_record_id": "9.26", "section": "translation", "gold_chunk_ids": ["9.26#translation"],
             "must_contain": ["leaf, a flower, fruit or water", "I will accept it"]},
        ],
        "expected_answer_summary": "Yes. If one offers Him a leaf, flower, fruit, or water with love and devotion, He will accept it.",
        "notes": "",
    },
    {
        "id": "gita_eval_009",
        "category": "factual",
        "difficulty": "medium",
        "question": "According to the Gita, who is seated in everyone’s heart and from whom do remembrance, knowledge, and forgetfulness come?",
        "golds": [
            {"source_record_id": "15.15", "section": "translation", "gold_chunk_ids": ["15.15#translation"],
             "must_contain": ["seated in everyone’s heart", "remembrance, knowledge and forgetfulness"]},
        ],
        "expected_answer_summary": "Krishna says He sits in everyone’s heart, and remembrance, knowledge, and forgetfulness come from Him.",
        "notes": "",
    },
    {
        "id": "gita_eval_010",
        "category": "factual",
        "difficulty": "medium",
        "question": "When Arjuna asks about nature, the enjoyer, and the field, what does Krishna call the body and the one who knows the body?",
        "golds": [
            {"source_record_id": "13.1-2", "section": "translation", "gold_chunk_ids": ["13.1-2#translation"],
             "must_contain": ["This body, O son of Kunté, is called the field", "knower of the field"]},
        ],
        "expected_answer_summary": "The body is called the field, and the one who knows this body is called the knower of the field.",
        "notes": "Grouped TEXTS 1–2; gold must keep verse_start=1 verse_end=2.",
    },
    {
        "id": "gita_eval_011",
        "category": "factual",
        "difficulty": "easy",
        "question": "What three modes does the Gita say make up material nature?",
        "golds": [
            {"source_record_id": "14.5", "section": "translation", "gold_chunk_ids": ["14.5#translation"],
             "must_contain": ["goodness, passion and ignorance"]},
        ],
        "expected_answer_summary": "Material nature consists of goodness, passion, and ignorance, and the living being becomes conditioned by these modes.",
        "notes": "",
    },
    {
        "id": "gita_eval_012",
        "category": "factual",
        "difficulty": "medium",
        "question": "Which eight things does Krishna list as His separated material energies?",
        "golds": [
            {"source_record_id": "7.4", "section": "translation", "gold_chunk_ids": ["7.4#translation"],
             "must_contain": ["Earth, water, fire, air, ether", "false ego"]},
        ],
        "expected_answer_summary": "Earth, water, fire, air, ether, mind, intelligence, and false ego — eight separated material energies.",
        "notes": "",
    },
    # ----- 3. conceptual (~7) -----
    {
        "id": "gita_eval_013",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "Does the Gita say we should focus on doing our duty, rather than trying to control how the outcome turns out?",
        "golds": [
            {"source_record_id": "2.47", "section": "translation", "gold_chunk_ids": ["2.47#translation"],
             "must_contain": ["not entitled to the fruits of action"]},
            {"source_record_id": "2.47", "section": "purport", "gold_chunk_ids": ["2.47#purport.1"],
             "must_contain": ["without being attached to the result"]},
        ],
        "expected_answer_summary": "Yes. One has a right to prescribed work but not to the fruits; Krishna tells Arjuna to act without attachment to results, and also not to refuse the duty.",
        "notes": "Supported by both the verse and its purport; both listed.",
    },
    {
        "id": "gita_eval_014",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "In what way can a person’s own mind be their friend, and in what way can it be their enemy?",
        "golds": [
            {"source_record_id": "6.5", "section": "translation", "gold_chunk_ids": ["6.5#translation"],
             "must_contain": ["mind is the friend of the conditioned soul", "his enemy as well"]},
            {"source_record_id": "6.6", "section": "translation", "gold_chunk_ids": ["6.6#translation"],
             "must_contain": ["conquered the mind", "greatest enemy"]},
        ],
        "expected_answer_summary": "If the mind is mastered it helps the person; if not, it remains the greatest enemy. One must lift oneself with the mind, not degrade oneself.",
        "notes": "6.5 and 6.6 are a tight pair; 6.26/6.34 are related but not required gold.",
    },
    {
        "id": "gita_eval_015",
        "category": "conceptual",
        "difficulty": "hard",
        "question": "What does the Gita mean when it calls staying calm in both success and failure a kind of yoga?",
        "golds": [
            {"source_record_id": "2.48", "section": "translation", "gold_chunk_ids": ["2.48#translation"],
             "must_contain": ["abandoning all attachment to success or failure", "equanimity is called yoga"]},
        ],
        "expected_answer_summary": "Yoga here is acting in one’s duty while remaining equipoised, giving up attachment to winning or losing.",
        "notes": "",
    },
    {
        "id": "gita_eval_016",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "Is it better, according to the Gita, to stop working altogether than to keep doing one’s ordinary responsibilities?",
        "golds": [
            {"source_record_id": "3.8", "section": "translation", "gold_chunk_ids": ["3.8#translation"],
             "must_contain": ["doing so is better than not working"]},
        ],
        "expected_answer_summary": "No. Prescribed work is better than not working; one cannot even maintain the body without work.",
        "notes": "2.47 also warns against attachment to inaction, but 3.8 is the direct comparison.",
    },
    {
        "id": "gita_eval_017",
        "category": "conceptual",
        "difficulty": "hard",
        "question": "Why might someone believe ‘I am the one doing everything,’ and what does the Gita say is actually carrying out the actions?",
        "golds": [
            {"source_record_id": "3.27", "section": "translation", "gold_chunk_ids": ["3.27#translation"],
             "must_contain": ["false ego", "three modes of material nature"]},
        ],
        "expected_answer_summary": "False ego makes the soul think it is the doer; the activities are actually carried out by the three modes of material nature.",
        "notes": "",
    },
    {
        "id": "gita_eval_018",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "What understanding of Krishna does the Gita connect with finding peace from material suffering?",
        "golds": [
            {"source_record_id": "5.29", "section": "translation", "gold_chunk_ids": ["5.29#translation"],
             "must_contain": ["ultimate beneficiary of all sacrifices", "attains peace"]},
        ],
        "expected_answer_summary": "Knowing Krishna as the beneficiary of sacrifice and austerity, the Lord of all planets and demigods, and the well-wisher of all beings, one attains peace from material miseries.",
        "notes": "",
    },
    {
        "id": "gita_eval_019",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "What kind of person does Krishna call very dear to Him regarding envy, friendship, and not claiming to own the world?",
        "golds": [
            {"source_record_id": "12.13-14", "section": "translation", "gold_chunk_ids": ["12.13-14#translation"],
             "must_contain": ["not envious but is a kind friend", "does not think himself a proprietor"]},
        ],
        "expected_answer_summary": "A devotee who is not envious, is a friend to all, does not think himself the proprietor, is free from false ego, and is steady in devotion is very dear to Krishna.",
        "notes": "Grouped TEXTS 13–14.",
    },
    # ----- 4. purport (~6) -----
    {
        "id": "gita_eval_020",
        "category": "purport",
        "difficulty": "medium",
        "question": "Besides doing one’s proper duty, what other two kinds of action does the commentary on 2.47 warn about?",
        "golds": [
            {"source_record_id": "2.47", "section": "purport", "gold_chunk_ids": ["2.47#purport.1"],
             "must_contain": ["prescribed duties, capricious work, and inaction"]},
        ],
        "expected_answer_summary": "The purport distinguishes prescribed duties from capricious work (unauthorized action) and inaction (not doing one’s duty). Krishna wants duty done, not idleness or whimsical work.",
        "notes": "These three labels are in the purport, not in the short translation.",
    },
    {
        "id": "gita_eval_021",
        "category": "purport",
        "difficulty": "hard",
        "question": "Does the commentary on the ‘king of education’ verse say that even a person without high birth or schooling can become enlightened by serving devotees? What example does it give?",
        "golds": [
            {"source_record_id": "9.2", "section": "purport", "gold_chunk_ids": ["9.2#purport.4", "9.2#purport.5"],
             "must_contain": ["son of a maidservant", "no education"]},
        ],
        "expected_answer_summary": "Yes. The purport says devotional service itself purifies; Narada’s previous life as a maidservant’s uneducated son is the example, because he served great devotees.",
        "notes": "Narada’s story begins in 9.2#purport.4 and continues in 9.2#purport.5, so both chunks are gold.",
    },
    {
        "id": "gita_eval_022",
        "category": "purport",
        "difficulty": "hard",
        "question": "In the yoga chapter, does the purport treat the word atma as always meaning ‘the soul,’ or can it mean the mind in that context?",
        "golds": [
            {"source_record_id": "6.5", "section": "purport", "gold_chunk_ids": ["6.5#purport.1"],
             "must_contain": ["ätmä denotes body, mind and soul", "ätmä refers here to the mind"]},
        ],
        "expected_answer_summary": "The purport says atma can mean body, mind, or soul depending on context; here, because yoga centers on the mind, atma refers to the mind.",
        "notes": "Not stated in the translation, which only says the mind can be friend or enemy.",
    },
    {
        "id": "gita_eval_023",
        "category": "purport",
        "difficulty": "hard",
        "question": "Does the commentary on 15.7 say the living being is an individual only while trapped in this world, or that this individuality is eternal?",
        "golds": [
            {"source_record_id": "15.7", "section": "purport", "gold_chunk_ids": ["15.7#purport.1"],
             "must_contain": ["eternally", "not that he assumes individuality"]},
        ],
        "expected_answer_summary": "The purport says the living entity is an eternal fragmental part of the Lord, not someone who is individual only in conditioned life and later becomes one with God.",
        "notes": "The translation says ‘eternal fragmental parts’; the purport is more explicit about not merging individuality away.",
    },
    {
        "id": "gita_eval_024",
        "category": "purport",
        "difficulty": "medium",
        "question": "In the long commentary on the divine qualities in Chapter 16, how is truthfulness (satyam) explained, and how should difficult Vedic passages be understood?",
        "golds": [
            {"source_record_id": "16.1-3", "section": "purport", "gold_chunk_ids": ["16.1-3#purport.7"],
             "must_contain": ["not distort the truth for some personal interest", "bona fide spiritual master"]},
        ],
        "expected_answer_summary": "Satyam means not twisting the truth for personal gain. Hard Vedic passages should be learned from a bona fide spiritual master, not given a self-serving interpretation.",
        "notes": "This explanation is in purport part 7, not in the translation’s list of qualities.",
    },
    {
        "id": "gita_eval_025",
        "category": "purport",
        "difficulty": "medium",
        "question": "Does the commentary on offering a leaf or water say this process is only for scholars, or can even a poor unqualified person attempt it?",
        "golds": [
            {"source_record_id": "9.26", "section": "purport", "gold_chunk_ids": ["9.26#purport.1"],
             "must_contain": ["poorest of the poor", "without any kind of qualification"]},
        ],
        "expected_answer_summary": "The purport says the process is easy and can be attempted even by the poorest, without material qualification; the needed qualification is to be a pure devotee.",
        "notes": "The translation mentions leaf/water; the ‘poorest of the poor / no qualification’ wording is purport-only.",
    },
    # ----- 5. terminology (~5) -----
    {
        "id": "gita_eval_026",
        "category": "terminology",
        "difficulty": "medium",
        "question": "When this Gita defines karma next to Brahman and the self, what kind of action is karma said to be?",
        "golds": [
            {"source_record_id": "8.3", "section": "translation", "gold_chunk_ids": ["8.3#translation"],
             "must_contain": ["called karma, or fruitive activi"]},
        ],
        "expected_answer_summary": "Karma is action connected with the development of the material bodies of living beings, i.e. fruitive activities.",
        "notes": "This edition also calls the living entity Brahman in the same verse; the question is only about the karma definition.",
    },
    {
        "id": "gita_eval_027",
        "category": "terminology",
        "difficulty": "medium",
        "question": "What does this Gita call yoga when it talks about working without clinging to success or failure?",
        "golds": [
            {"source_record_id": "2.48", "section": "translation", "gold_chunk_ids": ["2.48#translation"],
             "must_contain": ["equanimity is called yoga"]},
        ],
        "expected_answer_summary": "That even-minded, unattached performance of duty is called yoga.",
        "notes": "Shares a gold verse with 015 but asks for the term ‘yoga’, not the pastoral advice.",
    },
    {
        "id": "gita_eval_028",
        "category": "terminology",
        "difficulty": "medium",
        "question": "According to the Gita, who actually performs activities when a person, under false ego, thinks ‘I am the doer’ — and what role do the gunas play?",
        "golds": [
            {"source_record_id": "3.27", "section": "translation", "gold_chunk_ids": ["3.27#translation"],
             "must_contain": ["three modes of material nature"]},
        ],
        "expected_answer_summary": "The three modes (gunas) of material nature carry out the activities; false ego makes the soul think it is the doer.",
        "notes": "Same verse as 017; this item tests the term gunas/modes rather than the conceptual ‘why’. Acceptable overlap of gold, different question.",
    },
    {
        "id": "gita_eval_029",
        "category": "terminology",
        "difficulty": "hard",
        "question": "What does this book mean by prakriti when it lists earth, water, fire, and so on as Krishna’s energies?",
        "golds": [
            {"source_record_id": "7.4", "section": "translation", "gold_chunk_ids": ["7.4#translation"],
             "must_contain": ["separated material energies"]},
            {"source_record_id": "7.4", "section": "purport", "gold_chunk_ids": ["7.4#purport.1"],
             "must_contain": ["Material nature is called prakåti"]},
        ],
        "expected_answer_summary": "Those eight items are Krishna’s separated material energies; the purport names material nature prakriti, energy of the Lord.",
        "notes": "The English word ‘prakriti’ is in the purport (as prakåti). Translation says ‘material energies’ without that Sanskrit label.",
    },
    {
        "id": "gita_eval_030",
        "category": "terminology",
        "difficulty": "hard",
        "question": "Who is Paramatma according to the commentary on the verse where Krishna says He sits in the heart?",
        "golds": [
            {"source_record_id": "15.15", "section": "purport", "gold_chunk_ids": ["15.15#purport.1"],
             "must_contain": ["situated as Paramätmä in everyone’s heart"]},
        ],
        "expected_answer_summary": "The purport identifies the Lord in the heart as Paramatma, from whom activities are initiated and who witnesses the living being’s work.",
        "notes": "The translation does not use the word Paramatma; that label is in the purport. Useful because synonyms were not embedded.",
    },
]


def excerpt(text: str, needle: str, radius: 180) -> str:
    ntext = text
    idx = norm(ntext).find(norm(needle))
    if idx < 0:
        return ntext[:240].replace("\n", " ")
    # map approx via original
    raw_idx = ntext.lower().find(needle.lower()[:20]) if needle else 0
    if raw_idx < 0:
        raw_idx = 0
    start = max(0, raw_idx - 40)
    end = min(len(ntext), raw_idx + radius)
    snippet = ntext[start:end].replace("\n", " ").strip()
    return snippet


def main() -> None:
    records = json.loads(CORPUS.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    rec_by_id = {source_record_id(r): r for r in records}
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    issues: list[str] = []
    frozen_questions = []

    for q in QUESTIONS:
        gold_sources = []
        for g in q["golds"]:
            sid = g["source_record_id"]
            rec = rec_by_id.get(sid)
            if rec is None:
                issues.append(f"{q['id']}: missing source {sid}")
                continue
            bodies = []
            for cid in g["gold_chunk_ids"]:
                ch = chunk_by_id.get(cid)
                if ch is None:
                    issues.append(f"{q['id']}: missing chunk {cid}")
                    continue
                if ch["source_record_id"] != sid:
                    issues.append(f"{q['id']}: chunk {cid} belongs to {ch['source_record_id']} not {sid}")
                if ch["section"] != g["section"]:
                    issues.append(f"{q['id']}: chunk {cid} section {ch['section']} != {g['section']}")
                if ch["chapter"] != rec["chapter"] or ch["verse_start"] != rec["verse_start"] or ch["verse_end"] != rec["verse_end"]:
                    issues.append(f"{q['id']}: chunk {cid} verse range mismatch")
                bodies.append(ch["text"])
            joined = "\n".join(bodies)
            for phrase in g["must_contain"]:
                if norm(phrase) not in norm(joined):
                    issues.append(f"{q['id']}: gold chunks missing phrase {phrase!r}")
            gold_sources.append(
                {
                    "source_record_id": sid,
                    "chapter": rec["chapter"],
                    "verse_start": rec["verse_start"],
                    "verse_end": rec["verse_end"],
                    "text_label": rec["text_label"],
                    "section": g["section"],
                    "gold_chunk_ids": g["gold_chunk_ids"],
                }
            )
        # evidence from first gold chunk
        first_cid = q["golds"][0]["gold_chunk_ids"][0]
        first_chunk = chunk_by_id.get(first_cid, {})
        evidence = excerpt(first_chunk.get("text", ""), q["golds"][0]["must_contain"][0], 220)
        frozen_questions.append(
            {
                "id": q["id"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "gold_sources": gold_sources,
                "expected_answer_summary": q["expected_answer_summary"],
                "gold_evidence": evidence,
                "notes": q["notes"],
            }
        )

    sections = []
    for q in frozen_questions:
        secs = {g["section"] for g in q["gold_sources"]}
        if secs == {"translation"}:
            sections.append("translation_only")
        elif secs == {"purport"}:
            sections.append("purport_only")
        else:
            sections.append("both")

    chapters = sorted({g["chapter"] for q in frozen_questions for g in q["gold_sources"]})
    report = {
        "n_questions": len(frozen_questions),
        "by_category": dict(Counter(q["category"] for q in frozen_questions)),
        "by_difficulty": dict(Counter(q["difficulty"] for q in frozen_questions)),
        "by_gold_section": dict(Counter(sections)),
        "n_multiple_gold_source_objects": sum(1 for q in frozen_questions if len(q["gold_sources"]) > 1),
        "chapters_represented": chapters,
        "n_issues": len(issues),
        "issues": issues,
    }

    payload = {
        "status": "frozen",
        "purpose": "English retrieval evaluation for the Bhagavad Gita chunk corpus.",
        "how_to_use": (
            "The retrieval system must receive ONLY the 'question' field. "
            "Do not pass gold_sources, expected_answer_summary, or gold_evidence into the query. "
            "After retrieval, score hits against gold_chunk_ids / source_record_id."
        ),
        "do_not_edit_to_fit_retrieval_results": True,
        "n_questions": len(frozen_questions),
        "questions": frozen_questions,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(f"Validation failed with {len(issues)} issues.")
    print(f"Wrote frozen set to {OUT}")


if __name__ == "__main__":
    main()
