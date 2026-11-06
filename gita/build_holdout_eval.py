"""
Build and freeze a NEW 40-question held-out evaluation set.

Phase A only: gold labels come from gita_structured.json / gita_chunks.json.
This script does not run E5, BM25, RRF, or citation gating.

The original 30-question file is loaded solely to check duplication/overlap.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "gita_structured.json"
CHUNKS = ROOT / "gita_chunks.json"
OLD_EVAL = ROOT / "gita_eval_frozen.json"
OUT = ROOT / "gita_holdout_frozen.json"
REPORT = ROOT / "gita_holdout_validation.json"

CITATION_RE = re.compile(r"\b\d{1,2}\.\d{1,3}\b")
STOP = {
    "the", "a", "an", "of", "to", "and", "in", "on", "for", "with", "does", "what",
    "according", "gita", "krishna", "this", "that", "is", "are", "be", "or", "as",
    "how", "when", "who", "which", "from", "by", "about", "say", "says", "call",
}


def source_record_id(record: dict) -> str:
    if record["verse_start"] == record["verse_end"]:
        return f"{record['chapter']}.{record['verse_start']}"
    return f"{record['chapter']}.{record['verse_start']}-{record['verse_end']}"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def wordset(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


# Gold: source_record_id, section, gold_chunk_ids, must_contain
QUESTIONS = [
    # ----- citation_lookup (8): explicit dotted tokens, new verses -----
    {
        "id": "gita_holdout_001",
        "category": "citation_lookup",
        "difficulty": "easy",
        "question": "According to Bhagavad Gita 4.34, how should a person try to learn the truth?",
        "golds": [
            {"source_record_id": "4.34", "section": "translation", "gold_chunk_ids": ["4.34#translation"],
             "must_contain": ["approaching a spiritual master", "Inquire from him submissively"]},
        ],
        "expected_answer_summary": "By approaching a spiritual master, asking submissively, and serving him; self-realized souls can give knowledge because they have seen the truth.",
        "notes": "",
    },
    {
        "id": "gita_holdout_002",
        "category": "citation_lookup",
        "difficulty": "medium",
        "question": "In Bhagavad Gita 6.35, what does Krishna say about controlling a restless mind?",
        "golds": [
            {"source_record_id": "6.35", "section": "translation", "gold_chunk_ids": ["6.35#translation"],
             "must_contain": ["very difficult to curb the restless mind", "suitable practice and by detachment"]},
        ],
        "expected_answer_summary": "It is difficult to curb the restless mind, but it is possible by suitable practice and detachment.",
        "notes": "",
    },
    {
        "id": "gita_holdout_003",
        "category": "citation_lookup",
        "difficulty": "easy",
        "question": "What does Bhagavad Gita 7.7 say is superior to Krishna, and what image does He use for how everything rests on Him?",
        "golds": [
            {"source_record_id": "7.7", "section": "translation", "gold_chunk_ids": ["7.7#translation"],
             "must_contain": ["no truth superior to Me", "pearls are strung on a thread"]},
        ],
        "expected_answer_summary": "Nothing is superior to Krishna; everything rests on Him as pearls are strung on a thread.",
        "notes": "",
    },
    {
        "id": "gita_holdout_004",
        "category": "citation_lookup",
        "difficulty": "medium",
        "question": "According to Bhagavad Gita 9.22, what does Krishna say He does for those who worship Him with exclusive devotion?",
        "golds": [
            {"source_record_id": "9.22", "section": "translation", "gold_chunk_ids": ["9.22#translation"],
             "must_contain": ["I carry what they lack", "I preserve what they have"]},
        ],
        "expected_answer_summary": "He carries what they lack and preserves what they have.",
        "notes": "",
    },
    {
        "id": "gita_holdout_005",
        "category": "citation_lookup",
        "difficulty": "easy",
        "question": "What does Bhagavad Gita 10.8 say about Krishna as the source of the worlds?",
        "golds": [
            {"source_record_id": "10.8", "section": "translation", "gold_chunk_ids": ["10.8#translation"],
             "must_contain": ["source of all spiritual and material worlds", "Everything emanates from Me"]},
        ],
        "expected_answer_summary": "He is the source of all spiritual and material worlds; everything emanates from Him, and the wise who know this worship Him.",
        "notes": "",
    },
    {
        "id": "gita_holdout_006",
        "category": "citation_lookup",
        "difficulty": "easy",
        "question": "According to Bhagavad Gita 12.8, where should a person fix the mind and intelligence?",
        "golds": [
            {"source_record_id": "12.8", "section": "translation", "gold_chunk_ids": ["12.8#translation"],
             "must_contain": ["fix your mind upon Me", "engage all your intelligence in Me"]},
        ],
        "expected_answer_summary": "Fix the mind on Krishna and engage intelligence in Him; then one will live in Him.",
        "notes": "",
    },
    {
        "id": "gita_holdout_007",
        "category": "citation_lookup",
        "difficulty": "medium",
        "question": "What does Bhagavad Gita 17.15 describe as austerity of speech?",
        "golds": [
            {"source_record_id": "17.15", "section": "translation", "gold_chunk_ids": ["17.15#translation"],
             "must_contain": ["truthful, pleasing, beneficial", "reciting Vedic literature"]},
        ],
        "expected_answer_summary": "Speaking words that are truthful, pleasing, beneficial, and not agitating others, and regularly reciting Vedic literature.",
        "notes": "",
    },
    {
        "id": "gita_holdout_008",
        "category": "citation_lookup",
        "difficulty": "hard",
        "question": "In Bhagavad Gita 18.61, how does Krishna describe the Supreme Lord in relation to living beings wandering in the world?",
        "golds": [
            {"source_record_id": "18.61", "section": "translation", "gold_chunk_ids": ["18.61#translation"],
             "must_contain": ["situated in everyone’s heart", "seated as on a machine"]},
        ],
        "expected_answer_summary": "The Supreme Lord sits in everyone’s heart and directs the wanderings of living beings, who are seated as on a machine made of material energy.",
        "notes": "",
    },
    # ----- factual (8): no dotted citations -----
    {
        "id": "gita_holdout_009",
        "category": "factual",
        "difficulty": "easy",
        "question": "At the end of the first chapter, what does Sanjaya say Arjuna did after speaking on the battlefield?",
        "golds": [
            {"source_record_id": "1.46", "section": "translation", "gold_chunk_ids": ["1.46#translation"],
             "must_contain": ["cast aside his bow and arrows", "sat down on the chariot"]},
        ],
        "expected_answer_summary": "Arjuna threw down his bow and arrows and sat on the chariot, overwhelmed with grief.",
        "notes": "",
    },
    {
        "id": "gita_holdout_010",
        "category": "factual",
        "difficulty": "medium",
        "question": "What chain of inner reactions does the Gita say begins when a person keeps thinking about sense objects?",
        "golds": [
            {"source_record_id": "2.62", "section": "translation", "gold_chunk_ids": ["2.62#translation"],
             "must_contain": ["contemplating the objects of the senses", "from lust anger arises"]},
        ],
        "expected_answer_summary": "Contemplating sense objects produces attachment, attachment produces lust, and lust produces anger.",
        "notes": "",
    },
    {
        "id": "gita_holdout_011",
        "category": "factual",
        "difficulty": "medium",
        "question": "If someone leaves the body at the end of life while remembering Krishna alone, what does the Gita say happens?",
        "golds": [
            {"source_record_id": "8.5", "section": "translation", "gold_chunk_ids": ["8.5#translation"],
             "must_contain": ["remembering Me alone", "attains My nature"]},
        ],
        "expected_answer_summary": "That person at once attains Krishna’s nature; there is no doubt.",
        "notes": "",
    },
    {
        "id": "gita_holdout_012",
        "category": "factual",
        "difficulty": "medium",
        "question": "Where does Krishna say He sits as the Supersoul, and how does He describe His relation to the beginning, middle, and end of all beings?",
        "golds": [
            {"source_record_id": "10.20", "section": "translation", "gold_chunk_ids": ["10.20#translation"],
             "must_contain": ["seated in the hearts of all living entities", "beginning, the middle and the end"]},
        ],
        "expected_answer_summary": "He is the Supersoul in the hearts of all living beings, and He is their beginning, middle, and end.",
        "notes": "",
    },
    {
        "id": "gita_holdout_013",
        "category": "factual",
        "difficulty": "easy",
        "question": "After hearing Krishna describe Himself, what does Arjuna say he still wants to see about Krishna’s position in the universe?",
        "golds": [
            {"source_record_id": "11.3", "section": "translation", "gold_chunk_ids": ["11.3#translation"],
             "must_contain": ["how You have entered into this cosmic manifestation"]},
        ],
        "expected_answer_summary": "Arjuna wants to see how Krishna has entered the cosmic manifestation—that form of His.",
        "notes": "",
    },
    {
        "id": "gita_holdout_014",
        "category": "factual",
        "difficulty": "hard",
        "question": "When Krishna lists items in the process of knowledge, does He include approaching a genuine spiritual teacher along with qualities like humility and nonviolence?",
        "golds": [
            {"source_record_id": "13.8-12", "section": "translation", "gold_chunk_ids": ["13.8-12#translation"],
             "must_contain": ["approaching a bona fide spiritual master", "nonviolence"]},
        ],
        "expected_answer_summary": "Yes. The grouped verses list humility, pridelessness, nonviolence, and approaching a bona fide spiritual master among the process of knowledge.",
        "notes": "Grouped TEXTS 8–12; use existing source_record_id 13.8-12.",
    },
    {
        "id": "gita_holdout_015",
        "category": "factual",
        "difficulty": "medium",
        "question": "What does the Gita say happens to someone who serves Krishna in full devotion without fail, regarding the modes of nature?",
        "golds": [
            {"source_record_id": "14.26", "section": "translation", "gold_chunk_ids": ["14.26#translation"],
             "must_contain": ["transcends the modes of material nature", "comes to the level of Brahman"]},
        ],
        "expected_answer_summary": "That person immediately transcends the modes and comes to the level of Brahman.",
        "notes": "",
    },
    {
        "id": "gita_holdout_016",
        "category": "factual",
        "difficulty": "easy",
        "question": "What three things does the Gita call gates leading to hell, which a sane person should give up?",
        "golds": [
            {"source_record_id": "16.21", "section": "translation", "gold_chunk_ids": ["16.21#translation"],
             "must_contain": ["lust, anger and greed"]},
        ],
        "expected_answer_summary": "Lust, anger, and greed are three gates to hell and should be given up.",
        "notes": "",
    },
    # ----- conceptual (8) -----
    {
        "id": "gita_holdout_017",
        "category": "conceptual",
        "difficulty": "hard",
        "question": "Should a knowledgeable person try to make less informed workers quit their ordinary jobs, or is there a better way to help them?",
        "golds": [
            {"source_record_id": "3.26", "section": "translation", "gold_chunk_ids": ["3.26#translation"],
             "must_contain": ["should not induce them to stop work", "working in the spirit of devotion"]},
        ],
        "expected_answer_summary": "A learned person should not make ignorant workers stop; by working in devotion he should engage them in activities.",
        "notes": "",
    },
    {
        "id": "gita_holdout_018",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "Is it safer, according to the Gita, to do someone else’s job perfectly than to do one’s own duty even if one does it imperfectly?",
        "golds": [
            {"source_record_id": "3.35", "section": "translation", "gold_chunk_ids": ["3.35#translation"],
             "must_contain": ["far better to discharge one’s prescribed duties, even though faultily", "another’s path is dangerous"]},
        ],
        "expected_answer_summary": "No. One’s own duty done imperfectly is better; following another’s path is dangerous.",
        "notes": "Related later to 18.47, but this item golds the Chapter 3 statement.",
    },
    {
        "id": "gita_holdout_019",
        "category": "conceptual",
        "difficulty": "hard",
        "question": "What kind of seeing does the Gita praise when it talks about noticing stillness inside activity, and activity inside apparent stillness?",
        "golds": [
            {"source_record_id": "4.18", "section": "translation", "gold_chunk_ids": ["4.18#translation"],
             "must_contain": ["sees inaction in action, and action in inaction"]},
        ],
        "expected_answer_summary": "One who sees inaction in action and action in inaction is intelligent and transcendental even while active.",
        "notes": "",
    },
    {
        "id": "gita_holdout_020",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "How does true knowledge change the way a humble sage looks at a scholar, a cow, an elephant, a dog, and an outcaste?",
        "golds": [
            {"source_record_id": "5.18", "section": "translation", "gold_chunk_ids": ["5.18#translation"],
             "must_contain": ["see with equal vision", "a dog and a dog-eater"]},
        ],
        "expected_answer_summary": "By true knowledge they see those beings with equal vision.",
        "notes": "",
    },
    {
        "id": "gita_holdout_021",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "Why would a wise person refuse to chase pleasures that come from the senses, even if those pleasures feel good for a while?",
        "golds": [
            {"source_record_id": "5.22", "section": "translation", "gold_chunk_ids": ["5.22#translation"],
             "must_contain": ["sources of misery", "have a beginning and an end"]},
        ],
        "expected_answer_summary": "Sense pleasures are sources of misery, they begin and end, so the wise do not delight in them.",
        "notes": "",
    },
    {
        "id": "gita_holdout_022",
        "category": "conceptual",
        "difficulty": "hard",
        "question": "What makes someone a perfect yogi in how they relate to other people’s joy and pain, compared with their own?",
        "golds": [
            {"source_record_id": "6.32", "section": "translation", "gold_chunk_ids": ["6.32#translation"],
             "must_contain": ["true equality of all beings", "in both their happiness and their distress"]},
        ],
        "expected_answer_summary": "A perfect yogi sees others’ happiness and distress as equal to his own.",
        "notes": "",
    },
    {
        "id": "gita_holdout_023",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "The Gita mentions two ways of leaving this world, one bright and one dark. What is the difference in what happens afterward?",
        "golds": [
            {"source_record_id": "8.26", "section": "translation", "gold_chunk_ids": ["8.26#translation"],
             "must_contain": ["one in light and one in darkness", "when one passes in darkness, he returns"]},
        ],
        "expected_answer_summary": "Passing in light, one does not come back; passing in darkness, one returns.",
        "notes": "",
    },
    {
        "id": "gita_holdout_024",
        "category": "conceptual",
        "difficulty": "easy",
        "question": "Can everyday acts like eating, giving, or performing austerity be turned into worship, according to the Gita?",
        "golds": [
            {"source_record_id": "9.27", "section": "translation", "gold_chunk_ids": ["9.27#translation"],
             "must_contain": ["Whatever you do, whatever you eat", "as an offering to Me"]},
        ],
        "expected_answer_summary": "Yes. Whatever one does, eats, offers, gives, or performs as austerity should be done as an offering to Krishna.",
        "notes": "",
    },
    # ----- purport (8): evidence in commentary, no dotted tokens -----
    {
        "id": "gita_holdout_025",
        "category": "purport",
        "difficulty": "medium",
        "question": "In the commentary about how ordinary people copy a great person’s example, why would a leader who smokes be a poor teacher of giving up smoking, and what title is given to a teacher who first behaves properly?",
        "golds": [
            {"source_record_id": "3.21", "section": "purport", "gold_chunk_ids": ["3.21#purport.1"],
             "must_contain": ["stop smoking if he himself smokes", "called äcärya"]},
        ],
        "expected_answer_summary": "A leader cannot teach the public to stop smoking if he smokes; one who behaves properly before teaching is called acarya.",
        "notes": "Smoking example and acarya label are purport-only.",
    },
    {
        "id": "gita_holdout_026",
        "category": "purport",
        "difficulty": "medium",
        "question": "When the yoga commentary discusses regulating eating, what kind of food does it say a practitioner should become used to accepting?",
        "golds": [
            {"source_record_id": "6.17", "section": "purport", "gold_chunk_ids": ["6.17#purport.1"],
             "must_contain": ["accept prasädam"]},
        ],
        "expected_answer_summary": "Eating can be regulated when one practices accepting prasadam, sanctified food.",
        "notes": "",
    },
    {
        "id": "gita_holdout_027",
        "category": "purport",
        "difficulty": "hard",
        "question": "The commentary on the Lord’s covering energy says a bound person cannot free another bound person. Who does it say can actually release a conditioned soul?",
        "golds": [
            {"source_record_id": "7.14", "section": "purport", "gold_chunk_ids": ["7.14#purport.2"],
             "must_contain": ["bound cannot help", "only Lord Kåñëa"]},
        ],
        "expected_answer_summary": "Only Krishna or His bona fide representative the spiritual master can release the conditioned soul.",
        "notes": "Evidence is in purport part 2, not the short translation.",
    },
    {
        "id": "gita_holdout_028",
        "category": "purport",
        "difficulty": "hard",
        "question": "What example does the commentary give of a king whose last thought was an animal, and what body did he get next?",
        "golds": [
            {"source_record_id": "8.6", "section": "purport", "gold_chunk_ids": ["8.6#purport.1"],
             "must_contain": ["Mahäräja Bharata", "body of a deer"]},
        ],
        "expected_answer_summary": "Maharaja Bharata thought of a deer at death and was transferred into a deer’s body.",
        "notes": "Bharata story is purport-only.",
    },
    {
        "id": "gita_holdout_029",
        "category": "purport",
        "difficulty": "medium",
        "question": "When Krishna gives Arjuna special vision, does the commentary say a pure devotee prefers to see the two-handed form, or that the universal display is what devotees most want?",
        "golds": [
            {"source_record_id": "11.8", "section": "purport", "gold_chunk_ids": ["11.8#purport.1"],
             "must_contain": ["form with two hands", "universal form of Kåñëa is not very important"]},
        ],
        "expected_answer_summary": "A pure devotee prefers Krishna’s two-handed form; the universal form is not very important to such devotees.",
        "notes": "Part 1 of the 11.8 purport.",
    },
    {
        "id": "gita_holdout_030",
        "category": "purport",
        "difficulty": "hard",
        "question": "In the commentary that compares this world to an upside-down banyan, what is said to be the purpose of studying the Vedas?",
        "golds": [
            {"source_record_id": "15.1", "section": "purport", "gold_chunk_ids": ["15.1#purport.1"],
             "must_contain": ["purpose of Vedic study is to understand Kåñëa"]},
        ],
        "expected_answer_summary": "The purpose of Vedic study is to understand Krishna; one in Krishna consciousness already knows the Vedas.",
        "notes": "Part 1 of the 15.1 purport.",
    },
    {
        "id": "gita_holdout_031",
        "category": "purport",
        "difficulty": "medium",
        "question": "Does the commentary on charity in the mode of goodness recommend giving to anyone at random, including every poor person, as a spiritual rule?",
        "golds": [
            {"source_record_id": "17.20", "section": "purport", "gold_chunk_ids": ["17.20#purport.1"],
             "must_contain": ["indiscriminate charity is not recommended"]},
        ],
        "expected_answer_summary": "No. Vedic literature does not recommend indiscriminate charity; worthiness and spiritual consideration matter.",
        "notes": "",
    },
    {
        "id": "gita_holdout_032",
        "category": "purport",
        "difficulty": "hard",
        "question": "In the long commentary on the process of knowledge, what does it say that process finally aims at, beyond listing qualities like humility?",
        "golds": [
            {"source_record_id": "13.8-12", "section": "purport", "gold_chunk_ids": ["13.8-12#purport.1"],
             "must_contain": ["unalloyed devotional service"]},
        ],
        "expected_answer_summary": "The process of knowledge terminates in unalloyed devotional service to the Lord.",
        "notes": "Gold is purport part 1, not the whole seven-part split.",
    },
    # ----- terminology (8) -----
    {
        "id": "gita_holdout_033",
        "category": "terminology",
        "difficulty": "easy",
        "question": "How does this Gita distinguish sannyasa, the renounced order, from tyaga, renunciation?",
        "golds": [
            {"source_record_id": "18.2", "section": "translation", "gold_chunk_ids": ["18.2#translation"],
             "must_contain": ["renounced order of life [sannyäsa]", "renunciation [tyäga]"]},
        ],
        "expected_answer_summary": "Sannyasa is giving up activities based on material desire; tyaga is giving up the results of all activities.",
        "notes": "",
    },
    {
        "id": "gita_holdout_034",
        "category": "terminology",
        "difficulty": "easy",
        "question": "What three words does this book say have been used from the beginning of creation to indicate the Supreme Absolute Truth?",
        "golds": [
            {"source_record_id": "17.23", "section": "translation", "gold_chunk_ids": ["17.23#translation"],
             "must_contain": ["oà tat sat"]},
        ],
        "expected_answer_summary": "Om tat sat.",
        "notes": "",
    },
    {
        "id": "gita_holdout_035",
        "category": "terminology",
        "difficulty": "medium",
        "question": "What does this Gita mean by samadhi when it describes yoga reaching a stage of trance?",
        "golds": [
            {"source_record_id": "6.20-23", "section": "translation", "gold_chunk_ids": ["6.20-23#translation"],
             "must_contain": ["trance, or samädhi", "mind is completely restrained"]},
        ],
        "expected_answer_summary": "Samadhi is the yogic trance in which the mind is fully restrained from material mental activities and one sees and rejoices in the self.",
        "notes": "Grouped TEXTS 20–23.",
    },
    {
        "id": "gita_holdout_036",
        "category": "terminology",
        "difficulty": "medium",
        "question": "When the Gita talks about work becoming a sacrifice for Vishnu, what does it say happens if work is not done that way?",
        "golds": [
            {"source_record_id": "3.9", "section": "translation", "gold_chunk_ids": ["3.9#translation"],
             "must_contain": ["sacrifice for Viñëu", "work causes bondage"]},
        ],
        "expected_answer_summary": "Work done as a sacrifice for Vishnu frees one; otherwise work causes bondage.",
        "notes": "",
    },
    {
        "id": "gita_holdout_037",
        "category": "terminology",
        "difficulty": "hard",
        "question": "What does the commentary mean by maya when it describes the Lord’s covering energy as illusion that ties a person like ropes?",
        "golds": [
            {"source_record_id": "7.14", "section": "purport", "gold_chunk_ids": ["7.14#purport.2"],
             "must_contain": ["mäyä [illusion]", "ropes of illusion"]},
        ],
        "expected_answer_summary": "Maya is illusion, the Lord’s energy that tightly binds the conditioned soul; its background is the Lord as supreme magician.",
        "notes": "The English/Sanskrit maya explanation is in purport part 2.",
    },
    {
        "id": "gita_holdout_038",
        "category": "terminology",
        "difficulty": "hard",
        "question": "In the commentary on seeing action and inaction, what does akarma mean?",
        "golds": [
            {"source_record_id": "4.18", "section": "purport", "gold_chunk_ids": ["4.18#purport.1"],
             "must_contain": ["Akarma means without reaction to work"]},
        ],
        "expected_answer_summary": "Akarma means without reaction to work; Krishna-conscious activity does not bind.",
        "notes": "",
    },
    {
        "id": "gita_holdout_039",
        "category": "terminology",
        "difficulty": "medium",
        "question": "What contrast does Arjuna draw between people engaged in personal devotion and those who worship the unmanifested impersonal Brahman?",
        "golds": [
            {"source_record_id": "12.1", "section": "translation", "gold_chunk_ids": ["12.1#translation"],
             "must_contain": ["devotional service", "impersonal Brahman, the unmanifested"]},
        ],
        "expected_answer_summary": "He asks which are more perfect: those in Krishna’s devotional service, or those who worship the unmanifested impersonal Brahman.",
        "notes": "",
    },
    {
        "id": "gita_holdout_040",
        "category": "terminology",
        "difficulty": "hard",
        "question": "According to the commentary on entering the kingdom of God at death, what is brahma-nirvana, and how does it differ from a Buddhist idea of void?",
        "golds": [
            {"source_record_id": "2.72", "section": "purport", "gold_chunk_ids": ["2.72#purport.1"],
             "must_contain": ["brahma-nirväëa", "Buddhist philosophy"]},
        ],
        "expected_answer_summary": "Brahma-nirvana is the divine Krishna-conscious state; unlike Buddhist void after material life, the Gita teaches actual life begins after materialistic life.",
        "notes": "The term brahma-nirvana is in the purport, not the short translation.",
    },
]


def excerpt(text: str, needle: str, radius: int = 220) -> str:
    raw_idx = text.lower().find(needle.lower()[:24]) if needle else 0
    if raw_idx < 0:
        raw_idx = 0
    start = max(0, raw_idx - 40)
    end = min(len(text), raw_idx + radius)
    return text[start:end].replace("\n", " ").strip()


def main() -> None:
    records = json.loads(CORPUS.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    old = json.loads(OLD_EVAL.read_text(encoding="utf-8"))
    rec_by_id = {source_record_id(r): r for r in records}
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    issues: list[str] = []
    frozen_questions = []
    overlap_notes: list[str] = []

    ids = [q["id"] for q in QUESTIONS]
    if len(ids) != 40:
        issues.append(f"expected 40 questions, found {len(ids)}")
    if len(ids) != len(set(ids)):
        issues.append("duplicate holdout IDs")
    if any(not i.startswith("gita_holdout_") for i in ids):
        issues.append("IDs must use gita_holdout_ prefix")

    cats = Counter(q["category"] for q in QUESTIONS)
    expected_cats = {
        "citation_lookup": 8,
        "factual": 8,
        "conceptual": 8,
        "purport": 8,
        "terminology": 8,
    }
    if dict(cats) != expected_cats:
        issues.append(f"category counts {dict(cats)} != {expected_cats}")

    old_questions = old["questions"]
    old_qtexts = {norm(q["question"]): q["id"] for q in old_questions}
    old_golds = {
        g["source_record_id"]
        for q in old_questions
        for g in q["gold_sources"]
    }

    citation_ids = []
    for q in QUESTIONS:
        tokens = CITATION_RE.findall(q["question"])
        if q["category"] == "citation_lookup":
            if not tokens:
                issues.append(f"{q['id']} citation question has no dotted token")
            citation_ids.append(q["id"])
        elif tokens:
            issues.append(f"{q['id']} non-citation question contains {tokens}")

        nq = norm(q["question"])
        if nq in old_qtexts:
            issues.append(f"{q['id']} exact duplicate of {old_qtexts[nq]}")
        ws = wordset(q["question"])
        for oq in old_questions:
            ows = wordset(oq["question"])
            if not ws or not ows:
                continue
            jac = len(ws & ows) / len(ws | ows)
            if jac >= 0.55:
                issues.append(
                    f"{q['id']} near-duplicate wording of {oq['id']} (jaccard={jac:.2f})"
                )

        gold_sources = []
        for g in q["golds"]:
            sid = g["source_record_id"]
            rec = rec_by_id.get(sid)
            if rec is None:
                issues.append(f"{q['id']}: missing source {sid}")
                continue
            if sid in old_golds:
                overlap_notes.append(f"{q['id']} reuses old gold source {sid}")
            bodies = []
            for cid in g["gold_chunk_ids"]:
                ch = chunk_by_id.get(cid)
                if ch is None:
                    issues.append(f"{q['id']}: missing chunk {cid}")
                    continue
                if ch["source_record_id"] != sid:
                    issues.append(f"{q['id']}: chunk {cid} belongs to {ch['source_record_id']}")
                if ch["section"] != g["section"]:
                    issues.append(f"{q['id']}: chunk {cid} section mismatch")
                if (
                    ch["chapter"] != rec["chapter"]
                    or ch["verse_start"] != rec["verse_start"]
                    or ch["verse_end"] != rec["verse_end"]
                ):
                    issues.append(f"{q['id']}: chunk {cid} verse range mismatch")
                bodies.append(ch["text"])
            joined = "\n".join(bodies)
            if not joined.strip():
                issues.append(f"{q['id']}: empty gold text for {sid}")
            for phrase in g["must_contain"]:
                if norm(phrase) not in norm(joined):
                    issues.append(f"{q['id']}: missing phrase {phrase!r}")
            gold_sources.append(
                {
                    "source_record_id": sid,
                    "chapter": rec["chapter"],
                    "verse_start": rec["verse_start"],
                    "verse_end": rec["verse_end"],
                    "text_label": rec["text_label"],
                    "section": g["section"],
                    "gold_chunk_ids": list(g["gold_chunk_ids"]),
                }
            )
        if not gold_sources:
            issues.append(f"{q['id']}: empty gold labels")
        first_cid = q["golds"][0]["gold_chunk_ids"][0]
        first_chunk = chunk_by_id.get(first_cid, {})
        evidence = excerpt(first_chunk.get("text", ""), q["golds"][0]["must_contain"][0])
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

    chapters = sorted({g["chapter"] for q in frozen_questions for g in q["gold_sources"]})
    chapter_counts = Counter(g["chapter"] for q in frozen_questions for g in q["gold_sources"])
    if set(range(1, 19)) - set(chapters):
        issues.append(f"missing chapters {sorted(set(range(1, 19)) - set(chapters))}")

    sections = []
    for q in frozen_questions:
        secs = {g["section"] for g in q["gold_sources"]}
        if secs == {"translation"}:
            sections.append("translation_only")
        elif secs == {"purport"}:
            sections.append("purport_only")
        else:
            sections.append("both")

    report = {
        "n_questions": len(frozen_questions),
        "by_category": dict(Counter(q["category"] for q in frozen_questions)),
        "by_difficulty": dict(Counter(q["difficulty"] for q in frozen_questions)),
        "by_gold_section": dict(Counter(sections)),
        "n_multiple_gold_source_objects": sum(1 for q in frozen_questions if len(q["gold_sources"]) > 1),
        "citation_question_ids": citation_ids,
        "n_citation_questions": len(citation_ids),
        "chapters_represented": chapters,
        "questions_or_gold_objects_per_chapter": {str(k): v for k, v in sorted(chapter_counts.items())},
        "old_gold_source_reuse": overlap_notes,
        "n_issues": len(issues),
        "issues": issues,
        "frozen_before_retrieval": True,
    }

    payload = {
        "status": "frozen",
        "set_type": "held_out",
        "purpose": (
            "Held-out English retrieval evaluation. Created after Steps 7–10 used the original "
            "30 questions for architecture decisions. Do not edit to fit retrieval results."
        ),
        "how_to_use": (
            "The retrieval system must receive ONLY the 'question' field. "
            "Do not pass gold_sources, expected_answer_summary, or gold_evidence into the query."
        ),
        "do_not_edit_to_fit_retrieval_results": True,
        "n_questions": len(frozen_questions),
        "questions": frozen_questions,
    }

    if issues:
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit(f"Validation failed with {len(issues)} issues. Held-out set NOT frozen.")

    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("THE HELD-OUT SET IS NOW FROZEN.")
    print(f"Wrote {OUT}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
