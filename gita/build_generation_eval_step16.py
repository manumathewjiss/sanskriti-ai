"""
Build and freeze the Step 16 generation evaluation set.

Does NOT run E5, FAISS, BM25, or Qwen.
Gold labels come only from gita_structured.json / gita_chunks.json.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "gita_structured.json"
CHUNKS = ROOT / "gita_chunks.json"
SMOKE = ROOT / "gita_generation_smoke_questions.json"
DEV30 = ROOT / "gita_eval_frozen.json"
HOLD40 = ROOT / "gita_holdout_frozen.json"
OUT = ROOT / "gita_generation_eval_frozen.json"
REPORT = ROOT / "gita_generation_eval_validation.json"


def source_id(record: dict) -> str:
    if record["verse_start"] == record["verse_end"]:
        return f"{record['chapter']}.{record['verse_start']}"
    return f"{record['chapter']}.{record['verse_start']}-{record['verse_end']}"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# Gold: source_record_id, section, gold_chunk_ids, must_contain (in that section/chunk text)
QUESTIONS = [
    # ---------- citation (8) ----------
    {
        "id": "gita_gen_eval_001",
        "category": "citation",
        "difficulty": "easy",
        "question": "According to Bhagavad Gita 1.21–22, what does Arjuna ask Krishna to do with the chariot?",
        "golds": [
            {
                "source_record_id": "1.21-22",
                "section": "translation",
                "gold_chunk_ids": ["1.21-22#translation"],
                "must_contain": ["draw my chariot between the two armies"],
            }
        ],
        "reference_answer": "Arjuna asks Krishna to draw the chariot between the two armies so he can see who has come to fight.",
        "reference_support_summary": "1.21–22 translation: Arjuna asks the infallible one to place the chariot between the armies.",
        "chapter": 1,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_002",
        "category": "citation",
        "difficulty": "easy",
        "question": "According to Bhagavad Gita 4.11, how does Krishna treat people according to how they surrender to Him?",
        "golds": [
            {
                "source_record_id": "4.11",
                "section": "translation",
                "gold_chunk_ids": ["4.11#translation"],
                "must_contain": ["As all surrender unto Me, I reward them accordingly"],
            }
        ],
        "reference_answer": "Krishna rewards people according to how they surrender to Him, and everyone follows His path in all respects.",
        "reference_support_summary": "4.11 translation: reward according to surrender; everyone follows His path.",
        "chapter": 4,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_003",
        "category": "citation",
        "difficulty": "easy",
        "question": "According to Bhagavad Gita 5.10, how is a person who does duty without attachment, offering the results to the Lord, compared to a lotus leaf?",
        "golds": [
            {
                "source_record_id": "5.10",
                "section": "translation",
                "gold_chunk_ids": ["5.10#translation"],
                "must_contain": ["lotus leaf is untouched by water"],
            }
        ],
        "reference_answer": "That person is unaffected by sinful action, as a lotus leaf is untouched by water.",
        "reference_support_summary": "5.10 translation: duty without attachment, results to the Lord, lotus-leaf comparison.",
        "chapter": 5,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_004",
        "category": "citation",
        "difficulty": "medium",
        "question": "What does Bhagavad Gita 8.16 say about even the highest planet in the material world, and about someone who reaches Krishna’s abode?",
        "golds": [
            {
                "source_record_id": "8.16",
                "section": "translation",
                "gold_chunk_ids": ["8.16#translation"],
                "must_contain": ["places of misery wherein repeated birth and death", "never takes birth again"],
            }
        ],
        "reference_answer": "From the highest material planet to the lowest, all are places of repeated birth and death. One who attains Krishna’s abode never takes birth again.",
        "reference_support_summary": "8.16 translation: material planets miserable with rebirth; Krishna’s abode ends rebirth.",
        "chapter": 8,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_005",
        "category": "citation",
        "difficulty": "easy",
        "question": "According to Bhagavad Gita 10.41, where do opulent, beautiful, and glorious creations come from?",
        "golds": [
            {
                "source_record_id": "10.41",
                "section": "translation",
                "gold_chunk_ids": ["10.41#translation"],
                "must_contain": ["spark of My splendor"],
            }
        ],
        "reference_answer": "They spring from but a spark of Krishna’s splendor.",
        "reference_support_summary": "10.41 translation: opulent, beautiful, glorious creations from a spark of His splendor.",
        "chapter": 10,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_006",
        "category": "citation",
        "difficulty": "easy",
        "question": "In Bhagavad Gita 11.33, what does Krishna tell Arjuna about the enemies already arranged for death, and what role can Arjuna play?",
        "golds": [
            {
                "source_record_id": "11.33",
                "section": "translation",
                "gold_chunk_ids": ["11.33#translation"],
                "must_contain": ["already put to death by My arrangement", "instrument in the fight"],
            }
        ],
        "reference_answer": "The enemies are already put to death by Krishna’s arrangement; Arjuna can be but an instrument in the fight, so he should get up and fight.",
        "reference_support_summary": "11.33 translation: already slain by Krishna’s arrangement; Arjuna as instrument.",
        "chapter": 11,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_007",
        "category": "citation",
        "difficulty": "medium",
        "question": "According to Bhagavad Gita 16.24, how should a person decide what is duty and what is not duty?",
        "golds": [
            {
                "source_record_id": "16.24",
                "section": "translation",
                "gold_chunk_ids": ["16.24#translation"],
                "must_contain": ["regulations of the scriptures"],
            }
        ],
        "reference_answer": "One should understand duty and non-duty by scriptural regulations, and act according to those rules so as to be gradually elevated.",
        "reference_support_summary": "16.24 translation: duty known from scripture; act by those rules.",
        "chapter": 16,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_008",
        "category": "citation",
        "difficulty": "easy",
        "question": "According to Bhagavad Gita 18.65, what does Krishna tell Arjuna to do, and what does He promise as his dear friend?",
        "golds": [
            {
                "source_record_id": "18.65",
                "section": "translation",
                "gold_chunk_ids": ["18.65#translation"],
                "must_contain": ["Always think of Me", "come to Me without fail"],
            }
        ],
        "reference_answer": "Always think of Krishna, become His devotee, worship Him, and offer homage. He promises Arjuna will come to Him without fail, as a very dear friend.",
        "reference_support_summary": "18.65 translation: think, devotee, worship, homage; promise to come to Krishna.",
        "chapter": 18,
        "notes": "Different verse from smoke 18.66 surrender question.",
    },
    # ---------- factual (8) ----------
    {
        "id": "gita_gen_eval_009",
        "category": "factual",
        "difficulty": "easy",
        "question": "When the Gita compares the soul to a person changing clothes, what happens to the old material body?",
        "golds": [
            {
                "source_record_id": "2.22",
                "section": "translation",
                "gold_chunk_ids": ["2.22#translation"],
                "must_contain": ["accepts new material bodies, giving up the old"],
            }
        ],
        "reference_answer": "The soul gives up old, useless bodies and accepts new ones, as a person gives up old garments and puts on new ones.",
        "reference_support_summary": "2.22 translation: clothes analogy for changing bodies.",
        "chapter": 2,
        "notes": "Not the 2.13/2.20 soul-not-destroyed retrieval items.",
    },
    {
        "id": "gita_gen_eval_010",
        "category": "factual",
        "difficulty": "easy",
        "question": "What does the Gita say about people who cook food only for their own sense enjoyment, compared with devotees who eat food first offered in sacrifice?",
        "golds": [
            {
                "source_record_id": "3.13",
                "section": "translation",
                "gold_chunk_ids": ["3.13#translation"],
                "must_contain": ["eat only sin"],
            }
        ],
        "reference_answer": "Devotees who eat food first offered for sacrifice are released from sins. Those who prepare food only for personal sense enjoyment eat only sin.",
        "reference_support_summary": "3.13 translation: offered food vs food for sense enjoyment.",
        "chapter": 3,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_011",
        "category": "factual",
        "difficulty": "hard",
        "question": "After many births and deaths, what does a person who is actually in knowledge do, and how rare is such a person said to be?",
        "golds": [
            {
                "source_record_id": "7.19",
                "section": "translation",
                "gold_chunk_ids": ["7.19#translation"],
                "must_contain": ["surrenders unto Me", "great soul is very rare"],
            }
        ],
        "reference_answer": "That person surrenders to Krishna, knowing Him to be the cause of all causes and all that is. Such a great soul is very rare.",
        "reference_support_summary": "7.19 translation: surrender after many births; rare great soul.",
        "chapter": 7,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_012",
        "category": "factual",
        "difficulty": "hard",
        "question": "How does Krishna describe His relation to this universe and to all beings when He speaks of His unmanifested form?",
        "golds": [
            {
                "source_record_id": "9.4",
                "section": "translation",
                "gold_chunk_ids": ["9.4#translation"],
                "must_contain": ["entire universe is pervaded", "All beings are in Me, but I am not in them"],
            }
        ],
        "reference_answer": "In His unmanifested form He pervades the entire universe. All beings are in Him, but He is not in them.",
        "reference_support_summary": "9.4 translation: unmanifested pervasion; beings in Him, not He in them.",
        "chapter": 9,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_013",
        "category": "factual",
        "difficulty": "medium",
        "question": "If people worship Krishna without deviation, give Him their activities, and keep their minds on Him, what does the Gita say He does about birth and death?",
        "golds": [
            {
                "source_record_id": "12.6-7",
                "section": "translation",
                "gold_chunk_ids": ["12.6-7#translation"],
                "must_contain": ["swift deliverer from the ocean of birth and death"],
            }
        ],
        "reference_answer": "Krishna is the swift deliverer of those devotees from the ocean of birth and death.",
        "reference_support_summary": "12.6–7 translation: undeviating worship; swift deliverer from birth and death.",
        "chapter": 12,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_014",
        "category": "factual",
        "difficulty": "medium",
        "question": "According to the Gita, where do people go if they live in the mode of goodness, the mode of passion, or the mode of ignorance?",
        "golds": [
            {
                "source_record_id": "14.18",
                "section": "translation",
                "gold_chunk_ids": ["14.18#translation"],
                "must_contain": ["higher planets", "earthly planets", "hellish worlds"],
            }
        ],
        "reference_answer": "Goodness leads gradually to higher planets; passion to earthly planets; ignorance to hellish worlds.",
        "reference_support_summary": "14.18 translation: destinations of the three modes.",
        "chapter": 14,
        "notes": "Different from the 14.5 ‘what are the three modes’ retrieval question.",
    },
    {
        "id": "gita_gen_eval_015",
        "category": "factual",
        "difficulty": "easy",
        "question": "Does the Gita say Krishna’s supreme abode is lit by the sun, the moon, fire, or electricity?",
        "golds": [
            {
                "source_record_id": "15.6",
                "section": "translation",
                "gold_chunk_ids": ["15.6#translation"],
                "must_contain": ["not illumined by the sun or moon, nor by fire or electricity"],
            }
        ],
        "reference_answer": "No. That supreme abode is not illumined by sun, moon, fire, or electricity. Those who reach it never return to this material world.",
        "reference_support_summary": "15.6 translation: abode not lit by sun/moon/fire/electricity; no return.",
        "chapter": 15,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_016",
        "category": "factual",
        "difficulty": "medium",
        "question": "Whom do people worship if they are in goodness, in passion, or in ignorance, according to the Gita?",
        "golds": [
            {
                "source_record_id": "17.4",
                "section": "translation",
                "gold_chunk_ids": ["17.4#translation"],
                "must_contain": ["worship the demigods", "worship the demons", "worship ghosts and spirits"],
            }
        ],
        "reference_answer": "Goodness: demigods. Passion: demons. Ignorance: ghosts and spirits.",
        "reference_support_summary": "17.4 translation: three kinds of worship by mode.",
        "chapter": 17,
        "notes": "",
    },
    # ---------- conceptual (8) ----------
    {
        "id": "gita_gen_eval_017",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "Why should a person stay calm when happiness and distress come and go, the way seasons come and go?",
        "golds": [
            {
                "source_record_id": "2.14",
                "section": "translation",
                "gold_chunk_ids": ["2.14#translation"],
                "must_contain": ["winter and summer seasons", "tolerate them without being disturbed"],
            }
        ],
        "reference_answer": "Happiness and distress appear and disappear like winter and summer. They arise from sense perception, and one must learn to tolerate them without being disturbed.",
        "reference_support_summary": "2.14 translation: seasonal analogy; tolerate sense-based happiness and distress.",
        "chapter": 2,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_018",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "When Krishna tells Arjuna to fight, what attitude toward work, profit, and ownership does He require?",
        "golds": [
            {
                "source_record_id": "3.30",
                "section": "translation",
                "gold_chunk_ids": ["3.30#translation"],
                "must_contain": ["surrendering all your works unto Me", "no claims to proprietorship"],
            }
        ],
        "reference_answer": "Arjuna should surrender all works to Krishna, with knowledge of Him, without desire for profit, without claims of ownership, and without lethargy, and then fight.",
        "reference_support_summary": "3.30 translation: surrender works; no profit-seeking or proprietorship; fight.",
        "chapter": 3,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_019",
        "category": "conceptual",
        "difficulty": "hard",
        "question": "If someone truly understands how Krishna appears and acts, what happens after that person leaves the body?",
        "golds": [
            {
                "source_record_id": "4.9",
                "section": "translation",
                "gold_chunk_ids": ["4.9#translation"],
                "must_contain": ["does not, upon leaving the body, take his birth again", "attains My eternal abode"],
            }
        ],
        "reference_answer": "That person does not take birth again in this material world but attains Krishna’s eternal abode.",
        "reference_support_summary": "4.9 translation: knowing transcendental appearance/activities ends rebirth.",
        "chapter": 4,
        "notes": "Different from 4.7–4.8 ‘why He appears’ retrieval questions.",
    },
    {
        "id": "gita_gen_eval_020",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "At the start of the yoga teaching, who is called the true mystic: a person who drops all work and lights no fire, or a person who still works as obligated without clinging to results?",
        "golds": [
            {
                "source_record_id": "6.1",
                "section": "translation",
                "gold_chunk_ids": ["6.1#translation"],
                "must_contain": ["unattached to the fruits of his work", "not he who lights no fire and performs no duty"],
            }
        ],
        "reference_answer": "The true mystic is one who works as obligated without attachment to fruits, not one who lights no fire and performs no duty.",
        "reference_support_summary": "6.1 translation: working unattached vs abandoning duty.",
        "chapter": 6,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_021",
        "category": "conceptual",
        "difficulty": "medium",
        "question": "In ordinary things like water, sunlight, and Vedic sound, how does Krishna say He can be recognized?",
        "golds": [
            {
                "source_record_id": "7.8",
                "section": "translation",
                "gold_chunk_ids": ["7.8#translation"],
                "must_contain": ["taste of water", "light of the sun and the moon", "syllable oà"],
            }
        ],
        "reference_answer": "He is the taste of water, the light of the sun and moon, the syllable om in Vedic mantras, the sound in ether, and ability in man.",
        "reference_support_summary": "7.8 translation: Krishna in water, light, om, ether-sound, human ability.",
        "chapter": 7,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_022",
        "category": "conceptual",
        "difficulty": "hard",
        "question": "If someone does something very wrong but remains determined in Krishna’s service, how should that person be regarded?",
        "golds": [
            {
                "source_record_id": "9.30",
                "section": "translation",
                "gold_chunk_ids": ["9.30#translation"],
                "must_contain": ["most abominable action", "considered saintly"],
            }
        ],
        "reference_answer": "Even if one commits a most abominable action, if engaged in devotional service that person is to be considered saintly because of proper determination.",
        "reference_support_summary": "9.30 translation: abominable action but saintly if determined in devotion.",
        "chapter": 9,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_023",
        "category": "conceptual",
        "difficulty": "hard",
        "question": "Why does a living being meet both good and bad results while moving among different kinds of bodies?",
        "golds": [
            {
                "source_record_id": "13.22",
                "section": "translation",
                "gold_chunk_ids": ["13.22#translation"],
                "must_contain": ["association with that material nature", "good and evil among various species"],
            }
        ],
        "reference_answer": "In material nature the living entity follows the ways of life, enjoying the three modes because of association with that nature, and thus meets good and evil among various species.",
        "reference_support_summary": "13.22 translation: association with material nature; good and evil in species.",
        "chapter": 13,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_024",
        "category": "conceptual",
        "difficulty": "hard",
        "question": "What two results does Krishna contrast if someone becomes conscious of Him, versus acting from false ego and refusing to hear Him?",
        "golds": [
            {
                "source_record_id": "18.58",
                "section": "translation",
                "gold_chunk_ids": ["18.58#translation"],
                "must_contain": ["pass over all the obstacles of conditioned life by My grace", "you will be lost"],
            }
        ],
        "reference_answer": "If conscious of Krishna, one passes over obstacles of conditioned life by His grace. If one acts through false ego and does not hear Him, one will be lost.",
        "reference_support_summary": "18.58 translation: grace vs false-ego loss.",
        "chapter": 18,
        "notes": "",
    },
    # ---------- purport (8) ----------
    {
        "id": "gita_gen_eval_025",
        "category": "purport",
        "difficulty": "medium",
        "question": "In the commentary on Krishna and Arjuna sounding their conchshells, what did that sound indicate about the other side’s chance of winning?",
        "golds": [
            {
                "source_record_id": "1.14",
                "section": "purport",
                "gold_chunk_ids": ["1.14#purport.1"],
                "must_contain": ["no hope of victory for the other side"],
            }
        ],
        "reference_answer": "The commentary says the transcendental conchshells indicated there was no hope of victory for the other side because Krishna was on the Pandavas’ side.",
        "reference_support_summary": "1.14 purport: transcendental conchshells; no hope of victory for the other army.",
        "chapter": 1,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_026",
        "category": "purport",
        "difficulty": "hard",
        "question": "In the commentary on the broken chain of teachers, whom does it say the Gita was especially meant for, and whom was it not meant for?",
        "golds": [
            {
                "source_record_id": "4.2",
                "section": "purport",
                "gold_chunk_ids": ["4.2#purport.1"],
                "must_contain": ["especially meant for the saintly kings", "never meant for the demonic persons"],
            }
        ],
        "reference_answer": "The commentary says the Gita was especially meant for saintly kings who would rule by it, and was never meant for demonic persons who would invent whimsical interpretations.",
        "reference_support_summary": "4.2 purport: saintly kings vs demonic interpreters. (Translation mentions the broken succession but not this audience contrast.)",
        "chapter": 4,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_027",
        "category": "purport",
        "difficulty": "hard",
        "question": "In the yoga commentary on eating too much or too little, does it say a practitioner must eat animal flesh, and what ordinary foods does it mention instead?",
        "golds": [
            {
                "source_record_id": "6.16",
                "section": "purport",
                "gold_chunk_ids": ["6.16#purport.1"],
                "must_contain": ["eat animal", "grains, vegetables, fruits and milk"],
            }
        ],
        "reference_answer": "The commentary says there is no need to eat animals because there is ample supply of grains, vegetables, fruits, and milk.",
        "reference_support_summary": "6.16 purport: against animal eating; lists plant/milk foods. Translation itself is about quantity of eating/sleeping.",
        "chapter": 6,
        "notes": "Different from holdout 6.17 regulating eating / kind of food.",
    },
    {
        "id": "gita_gen_eval_028",
        "category": "purport",
        "difficulty": "hard",
        "question": "In the commentary on a devotee who does something abominable, what two kinds of activity does it say a conditioned living being has?",
        "golds": [
            {
                "source_record_id": "9.30",
                "section": "purport",
                "gold_chunk_ids": ["9.30#purport.1"],
                "must_contain": ["one is conditional", "the other is constitutional"],
            }
        ],
        "reference_answer": "A conditioned living being has conditional activities and constitutional activities.",
        "reference_support_summary": "9.30 purport: conditional vs constitutional activities. Translation calls the person saintly but does not name these two kinds.",
        "chapter": 9,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_029",
        "category": "purport",
        "difficulty": "hard",
        "question": "In the commentary on Krishna giving understanding to those who serve Him with love, which term does it say is especially important?",
        "golds": [
            {
                "source_record_id": "10.10",
                "section": "purport",
                "gold_chunk_ids": ["10.10#purport.1"],
                "must_contain": ["buddhi-yogam is very significant"],
            }
        ],
        "reference_answer": "The commentary says the word buddhi-yogam is very significant in that verse.",
        "reference_support_summary": "10.10 purport: buddhi-yogam highlighted. Translation speaks of giving understanding, not the Sanskrit term.",
        "chapter": 10,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_030",
        "category": "purport",
        "difficulty": "medium",
        "question": "When Sanjaya compares the universal form’s brightness to thousands of suns, does the commentary say he and Dhritarashtra were standing on the battlefield, and how could Sanjaya see it?",
        "golds": [
            {
                "source_record_id": "11.12",
                "section": "purport",
                "gold_chunk_ids": ["11.12#purport.1"],
                "must_contain": ["Neither Saïjaya nor Dhåtaräñöra was present", "grace of Vyäsa"],
            }
        ],
        "reference_answer": "Neither Sanjaya nor Dhritarashtra was present. Sanjaya could see whatever happened by the grace of Vyasa.",
        "reference_support_summary": "11.12 purport: not present on the field; vision by Vyasa’s grace.",
        "chapter": 11,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_031",
        "category": "purport",
        "difficulty": "hard",
        "question": "In the commentary on taking another body with new senses, what does it say happens if a person mixes consciousness with the qualities of cats and dogs?",
        "golds": [
            {
                "source_record_id": "15.9",
                "section": "purport",
                "gold_chunk_ids": ["15.9#purport.1"],
                "must_contain": ["gets a cat or dog body"],
            }
        ],
        "reference_answer": "The commentary says that person gets a cat or dog body in the next life and enjoys in that way.",
        "reference_support_summary": "15.9 purport: cat/dog consciousness → cat/dog body. Translation describes gaining senses grouped around the mind, not this example.",
        "chapter": 15,
        "notes": "Different from smoke 8.6 Bharata/deer.",
    },
    {
        "id": "gita_gen_eval_032",
        "category": "purport",
        "difficulty": "hard",
        "question": "In the commentary on people in different modes worshiping demigods, demons, or ghosts, whom does it say is actually worshipable according to scripture?",
        "golds": [
            {
                "source_record_id": "17.4",
                "section": "purport",
                "gold_chunk_ids": ["17.4#purport.1"],
                "must_contain": ["only the Supreme Personality of Godhead is worshipable"],
            }
        ],
        "reference_answer": "According to scriptural injunction, only the Supreme Personality of Godhead is worshipable. People not faithful to those injunctions worship other objects according to the modes.",
        "reference_support_summary": "17.4 purport: only the Supreme is worshipable. Translation lists demigods/demons/ghosts without this contrast.",
        "chapter": 17,
        "notes": "",
    },
    # ---------- terminology (8) ----------
    {
        "id": "gita_gen_eval_033",
        "category": "terminology",
        "difficulty": "easy",
        "question": "What does the Gita call a person who is not disturbed by the threefold miseries, not carried away by happiness, and free from attachment, fear, and anger?",
        "golds": [
            {
                "source_record_id": "2.56",
                "section": "translation",
                "gold_chunk_ids": ["2.56#translation"],
                "must_contain": ["sage of steady mind"],
            }
        ],
        "reference_answer": "That person is called a sage of steady mind.",
        "reference_support_summary": "2.56 translation: definition of a sage of steady mind.",
        "chapter": 2,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_034",
        "category": "terminology",
        "difficulty": "medium",
        "question": "What does the Gita say the mode of goodness does for a person, and how does it still condition that person?",
        "golds": [
            {
                "source_record_id": "14.6",
                "section": "translation",
                "gold_chunk_ids": ["14.6#translation"],
                "must_contain": ["illuminating", "conditioned by a sense of happiness and knowledge"],
            }
        ],
        "reference_answer": "The mode of goodness is purer, illuminating, and frees one from sinful reactions, but those in it become conditioned by a sense of happiness and knowledge.",
        "reference_support_summary": "14.6 translation: goodness illuminates and also conditions via happiness and knowledge.",
        "chapter": 14,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_035",
        "category": "terminology",
        "difficulty": "medium",
        "question": "According to the Gita, what determines the particular kind of faith a living being is said to have?",
        "golds": [
            {
                "source_record_id": "17.3",
                "section": "translation",
                "gold_chunk_ids": ["17.3#translation"],
                "must_contain": ["according to the modes he has acquired"],
            }
        ],
        "reference_answer": "A living being is said to be of a particular faith according to the modes of nature that person has acquired.",
        "reference_support_summary": "17.3 translation: faith according to acquired modes.",
        "chapter": 17,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_036",
        "category": "terminology",
        "difficulty": "medium",
        "question": "When someone is transcendentally situated, what does the Gita say they realize, and what service do they then attain?",
        "golds": [
            {
                "source_record_id": "18.54",
                "section": "translation",
                "gold_chunk_ids": ["18.54#translation"],
                "must_contain": ["realizes the Supreme", "pure devotional service unto Me"],
            }
        ],
        "reference_answer": "That person at once realizes the Supreme Brahman, becomes fully joyful, never laments or desires anything, is equal to every living entity, and in that state attains pure devotional service to Krishna.",
        "reference_support_summary": "18.54 translation: Brahman realization then pure devotional service.",
        "chapter": 18,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_037",
        "category": "terminology",
        "difficulty": "medium",
        "question": "What kind of knowledge does the Gita place in the mode of goodness, regarding how one sees living beings?",
        "golds": [
            {
                "source_record_id": "18.20",
                "section": "translation",
                "gold_chunk_ids": ["18.20#translation"],
                "must_contain": ["one undivided spiritual nature is seen in all living entities"],
            }
        ],
        "reference_answer": "Knowledge in goodness sees one undivided spiritual nature in all living entities, though they are divided into innumerable forms.",
        "reference_support_summary": "18.20 translation: undivided spiritual nature in all beings.",
        "chapter": 18,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_038",
        "category": "terminology",
        "difficulty": "medium",
        "question": "For someone who always remembers Krishna without deviation, how easy does the Gita say He is to obtain?",
        "golds": [
            {
                "source_record_id": "8.14",
                "section": "translation",
                "gold_chunk_ids": ["8.14#translation"],
                "must_contain": ["I am easy to obtain"],
            }
        ],
        "reference_answer": "Krishna is easy to obtain for one who always remembers Him without deviation, because of constant engagement in devotional service.",
        "reference_support_summary": "8.14 translation: easy to obtain through undeviating remembrance.",
        "chapter": 8,
        "notes": "",
    },
    {
        "id": "gita_gen_eval_039",
        "category": "terminology",
        "difficulty": "medium",
        "question": "To people who constantly serve Krishna with love, what does He say He gives them?",
        "golds": [
            {
                "source_record_id": "10.10",
                "section": "translation",
                "gold_chunk_ids": ["10.10#translation"],
                "must_contain": ["the understanding by which they can come to Me"],
            }
        ],
        "reference_answer": "He gives the understanding by which they can come to Him.",
        "reference_support_summary": "10.10 translation: understanding given to loving servants. Same source as 029 but gold is the translation, not the purport term.",
        "chapter": 10,
        "notes": "Shares source 10.10 with 029 (different section/chunk).",
    },
    {
        "id": "gita_gen_eval_040",
        "category": "terminology",
        "difficulty": "medium",
        "question": "How can a person reach perfection by doing their own ordinary work, according to the Gita?",
        "golds": [
            {
                "source_record_id": "18.46",
                "section": "translation",
                "gold_chunk_ids": ["18.46#translation"],
                "must_contain": ["worship of the Lord", "performing his own work"],
            }
        ],
        "reference_answer": "By worship of the Lord, who is the source of all beings and all-pervading, a person can attain perfection through performing their own work.",
        "reference_support_summary": "18.46 translation: perfection through own work as worship of the all-pervading Lord.",
        "chapter": 18,
        "notes": "",
    },
]


def load_previous_questions() -> dict:
    prev = {"smoke": [], "dev30": [], "holdout40": []}
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    prev["smoke"] = [q["question"] for q in smoke["questions"]]
    for path, key in ((DEV30, "dev30"), (HOLD40, "holdout40")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        prev[key] = [q["question"] for q in payload["questions"]]
        prev[key + "_golds"] = []
        for q in payload["questions"]:
            for g in q["gold_sources"]:
                prev[key + "_golds"].append(g["source_record_id"])
    return prev


def main() -> None:
    records = json.loads(CORPUS.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    by_source = {source_id(r): r for r in records}
    by_chunk = {c["chunk_id"]: c for c in chunks}

    banned_smoke_sources = {
        "2.47",
        "18.66",
        "9.26",
        "16.21",
        "6.5",
        "6.6",
        "9.27",
        "9.2",
        "8.6",
        "8.3",
        "18.1",
        "18.2",
    }
    prev = load_previous_questions()
    smoke_norm = {norm(q) for q in prev["smoke"]}
    all_prev_norm = smoke_norm | {norm(q) for q in prev["dev30"]} | {norm(q) for q in prev["holdout40"]}

    frozen_questions = []
    errors: list[str] = []
    ids = []
    question_texts = []

    for spec in QUESTIONS:
        qid = spec["id"]
        ids.append(qid)
        qtext = spec["question"]
        if norm(qtext) in all_prev_norm:
            errors.append(f"{qid} duplicates a previous evaluation question")
        if any(s in spec["golds"][0]["source_record_id"] and spec["golds"][0]["source_record_id"] in banned_smoke_sources for s in [spec["golds"][0]["source_record_id"]]):
            if spec["golds"][0]["source_record_id"] in banned_smoke_sources:
                errors.append(f"{qid} uses banned smoke gold {spec['golds'][0]['source_record_id']}")
        gold_source_ids = []
        gold_chunk_ids = []
        gold_sections = []
        for g in spec["golds"]:
            sid = g["source_record_id"]
            rec = by_source.get(sid)
            if rec is None:
                errors.append(f"{qid} missing source {sid}")
                continue
            if rec["chapter"] != spec["chapter"]:
                errors.append(f"{qid} chapter field {spec['chapter']} != record {rec['chapter']}")
            gold_source_ids.append(sid)
            for cid in g["gold_chunk_ids"]:
                ch = by_chunk.get(cid)
                if ch is None:
                    errors.append(f"{qid} missing chunk {cid}")
                    continue
                if ch["source_record_id"] != sid:
                    errors.append(f"{qid} chunk {cid} source mismatch")
                if ch["section"] != g["section"]:
                    errors.append(f"{qid} chunk {cid} section {ch['section']} != {g['section']}")
                gold_chunk_ids.append(cid)
                body = ch["text"]
                for phrase in g["must_contain"]:
                    if phrase not in body and phrase.lower() not in body.lower():
                        errors.append(f"{qid} phrase not in {cid}: {phrase!r}")
            gold_sections.append(g["section"])
            if qtext.lower() in (rec.get("translation") or "").lower() and len(qtext) > 40:
                pass
        if qid in question_texts:
            errors.append(f"duplicate id {qid}")
        question_texts.append(qtext)
        if any(cid in qtext for cid in gold_chunk_ids):
            errors.append(f"{qid} question contains a gold chunk id")
        frozen_questions.append(
            {
                "id": qid,
                "question": qtext,
                "category": spec["category"],
                "difficulty": spec["difficulty"],
                "gold_source_ids": gold_source_ids,
                "gold_chunk_ids": gold_chunk_ids,
                "gold_sections": gold_sections,
                "reference_answer": spec["reference_answer"],
                "reference_support_summary": spec["reference_support_summary"],
                "chapter": spec["chapter"],
                "notes": spec["notes"],
                "gold_sources": [
                    {
                        "source_record_id": g["source_record_id"],
                        "section": g["section"],
                        "gold_chunk_ids": g["gold_chunk_ids"],
                    }
                    for g in spec["golds"]
                ],
            }
        )

    if len(QUESTIONS) != 40:
        errors.append(f"expected 40 questions, got {len(QUESTIONS)}")
    if len(set(ids)) != 40:
        errors.append("duplicate IDs")
    if len(set(norm(q) for q in question_texts)) != 40:
        errors.append("duplicate question text")
    cats = Counter(q["category"] for q in frozen_questions)
    if cats != {"citation": 8, "factual": 8, "conceptual": 8, "purport": 8, "terminology": 8}:
        errors.append(f"category counts {dict(cats)}")
    diffs = Counter(q["difficulty"] for q in frozen_questions)
    chapters = sorted({q["chapter"] for q in frozen_questions})
    multi = sum(1 for q in frozen_questions if len(q["gold_source_ids"]) > 1)
    if errors:
        raise SystemExit("Validation failed:\n" + "\n".join(errors))

    prior_gold = set(prev["dev30_golds"] + prev["holdout40_golds"])
    overlap_sources = sorted(
        {sid for q in frozen_questions for sid in q["gold_source_ids"] if sid in prior_gold}
    )

    payload = {
        "status": "frozen",
        "step": 16,
        "purpose": "Unseen English generation evaluation for the selected Step 14 RAG pipeline.",
        "development_smoke_set_excluded": True,
        "how_to_use": "The RAG system must receive ONLY the question field. Do not pass gold labels, reference answers, or notes into retrieval or generation.",
        "do_not_edit_after_hash": True,
        "n_questions": 40,
        "questions": frozen_questions,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    OUT.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    report = {
        "sha256": digest,
        "n_questions": 40,
        "category_distribution": dict(cats),
        "difficulty_distribution": dict(diffs),
        "chapters_represented": chapters,
        "n_chapters_represented": len(chapters),
        "all_18_chapters": chapters == list(range(1, 19)),
        "multi_gold_source_count": multi,
        "validation_checks": {
            "every_gold_source_exists": True,
            "every_gold_chunk_exists": True,
            "gold_section_matches_chunk": True,
            "must_contain_phrases_found": True,
            "chapter_matches_record": True,
            "no_question_contains_chunk_id": True,
            "no_exact_duplicate_questions": True,
            "no_duplicate_ids": True,
            "category_counts_8_each": True,
            "no_banned_smoke_gold_sources": True,
            "no_exact_duplicate_of_prior_eval_questions": True,
        },
        "overlap_with_prior_retrieval_eval_gold_sources": overlap_sources,
        "overlap_note": "Empty list means no shared gold source IDs with the frozen 30 or holdout 40 retrieval sets. Questions were compared for exact-text duplicates only; retrieval rankings were not inspected while writing items.",
        "created_before_retrieval": True,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Wrote {REPORT}")
    print(f"SHA-256: {digest}")
    print(f"categories: {dict(cats)}")
    print(f"difficulty: {dict(diffs)}")
    print(f"chapters: {chapters}")
    print(f"multi-gold sources: {multi}")
    print(f"prior-eval gold source overlap: {overlap_sources}")


if __name__ == "__main__":
    main()
