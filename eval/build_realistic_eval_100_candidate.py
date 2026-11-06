#!/usr/bin/env python3
"""Build eval/data/realistic_eval_100_candidate.json from the approved 10-question
pilot plus 90 new realistic questions. Dataset creation only. Does not run FAISS.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = ROOT / "eval/data/realistic_pilot_10.json"
SAMPLE_PATH = ROOT / "eval/data/faiss_baseline_sample.json"
CSV_PATH = ROOT / "res/combined.csv"
OUT_PATH = ROOT / "eval/data/realistic_eval_100_candidate.json"
REPORT_PATH = ROOT / "eval/data/realistic_eval_100_candidate_review.json"

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "at", "as", "is", "was", "were", "be", "been", "are", "this", "that",
    "these", "those", "he", "she", "it", "they", "his", "her", "their", "him",
    "them", "who", "what", "when", "where", "why", "how", "did", "does", "do",
    "said", "say", "after", "before", "about", "into", "than", "then", "also",
    "not", "no", "nor", "but", "if", "which", "whom", "whose", "you", "your",
    "o", "oh", "had", "have", "has", "will", "would", "could", "should", "may",
    "according", "being", "been",
}

# New questions 11-100. Primaries are unused frozen-500 row_index values.
# Extra gold is added only when a second verse independently answers the question.
NEW_QUESTIONS = [
    # --- Ramayana (18) ---
    dict(
        primary=640, extra=[], category="character_action",
        question="Which rakshasi did Vishvamitra warn Rama was destroying the lands of Malada and Karusha?",
        why="640 names Tataka as the one who daily ruins Malada and Karusha.",
        flags=["possible_english_leakage"],
        flag_notes="Question reuses the place-names Malada and Karusha from the verse; review whether that is too close to the Dutt wording.",
    ),
    dict(
        primary=2326, extra=[], category="event",
        question="While Vasistha stood with the Brahmanas before Dasaratha, what immediate order did he ask the king to give concerning Rama?",
        why="2326 is Vasistha telling the king to order Rama's installation at once.",
        flags=[],
    ),
    dict(
        primary=2410, extra=[], category="event",
        question="When Rama entered to see Dasaratha about the coronation, whom did he find sitting with his father, and how did the king look?",
        why="2410 is Rama seeing Dasaratha seated with Kaikeyi, looking wretched and dried-up.",
        flags=[],
    ),
    dict(
        primary=2481, extra=[], category="factual",
        question="On the day people had dressed for Rama's installation, what did those present notice about Rama's face?",
        why="2481 states that nobody saw any sign of mental agony on Rama's face.",
        flags=[],
    ),
    dict(
        primary=3628, extra=[], category="event",
        question="When Rama was getting Sita onto the river raft, what belongings did he place beside her?",
        why="3628 says Rama seated Sita on the raft and laid her clothes, ornaments, hoe, and basket beside her.",
        flags=[],
    ),
    dict(
        primary=4557, extra=[], category="character_action",
        question="After leaving his counselors behind, whom did Bharata go to visit, following the family priest?",
        why="4557 says Bharata left the ministers and went after the priest in order to see Bharadvaja.",
        flags=[],
    ),
    dict(
        primary=7394, extra=[], category="location",
        question="After Sita was abducted, at which lake did Rama begin to lament while walking with Lakshmana?",
        why="7394 names lake Pampa as the place Rama reached with Lakshmana and began to lament.",
        flags=[],
    ),
    dict(
        primary=8823, extra=[], category="event",
        question="In Sugriva's search orders, where were the monkeys told to look together for Rama's wife?",
        why="8823 instructs them to search mountain fastnesses, cascades, and woods for Sita. Neighboring verses name different rivers or islands, so they are not gold for this wording.",
        flags=[],
    ),
    dict(
        primary=9048, extra=[], category="advice_teaching",
        question="During the search for Sita, what does this counsel say happens to people who keep acting, compared with people who give in to grief?",
        why="9048 contrasts acting (fruit of effort appears) with giving way to grief (vigour cannot return).",
        flags=[],
    ),
    dict(
        primary=1554, extra=[], category="factual",
        question="What object, stored in its case, did Janaka's counselors bring when they approached the king?",
        why="1554 says the counselors brought the iron-plated bow in its case to Janaka.",
        flags=[],
    ),
    dict(
        primary=18189, extra=[], category="event",
        question="When Sita was heard weeping in the forest, whom did the young rishis go to tell?",
        why="18189 says the rishis' sons ran to Valmiki and reported Sita's wailing.",
        flags=[],
    ),
    dict(
        primary=5017, extra=[], category="philosophical",
        question="In this teaching spoken to Rama, whose life is called happy, and whose life unhappy?",
        why="5017: happy is the person others depend on; unhappy is the person who depends on others.",
        flags=[],
    ),
    dict(
        primary=7761, extra=[], category="event",
        question="What two things did Sugriva promise Rama he would regain after seeing Rama's grief?",
        why="7761 is Sugriva promising to deliver Rama so that he will recover both his wife and the kingdom.",
        flags=[],
    ),
    dict(
        primary=1604, extra=[], category="location",
        question="Where did Janaka say his brother Kusadhvaja ruled, and what river washed that city?",
        why="1604 names Kusadhvaja's city as Sankasya, washed by the river Ikshumati.",
        flags=[],
    ),
    dict(
        primary=8701, extra=[], category="reason",
        question="According to Sugriva, why was extra help unnecessary for Rama in battle?",
        why="8701: the mere sound of Rama stretching his bow made the earth and mountains tremble, so helpers are pointless.",
        flags=[],
    ),
    dict(
        primary=9401, extra=[], category="event",
        question="When the monkeys were discouraged at the ocean, what feat were they urging Hanuman to attempt, and to whose three strides did they compare it?",
        why="9401 urges Hanuman to put forth his power like Vishnu covering the three worlds in three steps.",
        flags=["possible_english_leakage"],
        flag_notes="Question mentions Vishnu's three steps, which is also in the source English.",
    ),
    dict(
        primary=1499, extra=[], category="location",
        question="After leaving the northern direction, toward which quarter did the great muni go to perform severe austerities?",
        why="1499 says the muni left the north and undertook dreadful austerities in the east.",
        flags=[],
    ),
    dict(
        primary=5099, extra=[], category="philosophical",
        question="How does this teaching spoken to Rama compare family, home, and property to a traveler's overnight stop?",
        why="5099 compares parents, house, and wealth to a traveler's overnight stop that is left the next day.",
        flags=["possible_english_leakage"],
        flag_notes="Question keeps the triad parents/home/wealth and the traveler-lodging analogy, which is the verse's distinctive wording.",
    ),
    # --- Mahabharata (72) ---
    dict(
        primary=22706, extra=[], category="factual",
        question="Which of Shantanu and Satyavati's sons inherited the throne after the other was slain by Gandharvas before adulthood?",
        why="22706 states that Chitrangada was killed before youth and therefore Vichitravirya became king.",
        flags=[],
    ),
    dict(
        primary=25741, extra=[], category="character_action",
        question="How did Duryodhana propose to buy Drupada's court away from Yudhishthira's side?",
        why="25741 proposes tempting Drupada, his sons, and ministers with large gifts so they abandon Yudhishthira.",
        flags=["other"],
        flag_notes="Same Duryodhana-Vidura plotting conversation as the Dhritarashtra/Vidura-concealment question (row 25737).",
    ),
    dict(
        primary=44857, extra=[], category="advice_teaching",
        question="After Krishna had spoken as a friend seeking peace, what did Bhishma tell Duryodhana not to follow instead?",
        why="44857: Bhishma tells his son to listen to Krishna's peace-words and not follow vindictiveness.",
        flags=[],
    ),
    dict(
        primary=45154, extra=[], category="advice_teaching",
        question="According to Kunti, what livelihood belongs to each of the four social orders?",
        why="45154 assigns alms, protecting subjects, acquiring wealth, and serving the other orders.",
        flags=[],
    ),
    dict(
        primary=34870, extra=[], category="philosophical",
        question="According to the snake who spoke with Bhima, what three kinds of rebirth can a person's own acts produce?",
        why="34870 lists human birth, heavenly existence, and birth among animals as the three results of one's acts.",
        flags=[],
    ),
    dict(
        primary=42527, extra=[], category="philosophical",
        question="In Sanatsujata's teaching, what is named as one kind of death, what as another, and what is identified with immortality?",
        why="42527: Yama is one death; wearing-out of the soul another; pursuit of Brahman/self-knowledge is immortality.",
        flags=[],
    ),
    dict(
        primary=48051, extra=[], category="philosophical",
        question="In Krishna's battlefield teaching to Arjuna, what kind of person is said to attain peace?",
        why="48051: the person who casts off desires and is free from attachment, craving, and egoism attains peace.",
        flags=["other"],
        flag_notes="Related Gita cluster with the egoism/killing teaching (row 48559).",
    ),
    dict(
        primary=51530, extra=[51531], category="factual",
        question="When the Pandavas consulted after the ninth day's fight, what did they recall Bhishma had promised about counsel versus fighting?",
        why="51530 is Bhishma's prior pledge: he will give counsel but will not fight for the Pandavas' interests.",
        extra_why="51531 continues the same pledge independently: Bhishma will fight for Duryodhana's interests. Together they answer what he would and would not do.",
        flags=["uncertain_gold"],
        flag_notes="51531 is a continuation of the same quoted pledge. Review whether it independently answers or is only the next clause.",
    ),
    dict(
        primary=52268, extra=[], category="event",
        question="After Bhishma had spoken to Duryodhana from the bed of arrows, what did he then do despite his wounds?",
        why="52268: after speaking, Ganga's son remained silent and kept himself tranquil though his vitals were burning with arrows.",
        flags=[],
    ),
    dict(
        primary=65605, extra=[], category="reason",
        question="After Bhishma, Drona, Karna, Jayadratha, Duryodhana's brothers, and Lakshmana had fallen, what question did Kripa raise about continuing?",
        why="65605 lists those deaths and asks what remains for them to do now.",
        flags=[],
    ),
    dict(
        primary=28889, extra=[], category="event",
        question="When Yudhishthira was leaving for the forest after the dice game, which elders did he formally bid farewell to first?",
        why="28889: he takes leave of the Bharatas, aged grandsire Bhishma, Somadatta, and king Vahlika.",
        flags=[],
    ),
    dict(
        primary=29369, extra=[], category="character_action",
        question="What long-sought purpose did Kirmira say had brought him roaming armed across the world?",
        why="29369: Kirmira says he had long been roaming in order to kill Bhima and had not found him.",
        flags=[],
    ),
    dict(
        primary=29736, extra=[], category="event",
        question="During the fight at Saubha, what did Krishna say he started doing with his arrows to the enemies of the gods?",
        why="29736: Krishna took his bow and began cutting off their heads from the Saubha car.",
        flags=["other"],
        flag_notes="Same Saubha episode as the question about where Shalva stood (row 29690).",
    ),
    dict(
        primary=36581, extra=[], category="factual",
        question="What gift from Airavata's pair of bells did Indra present to Skanda?",
        why="36581: Indra brought Airavata's two Vaijayanti bells and gave them to Guha.",
        flags=[],
    ),
    dict(
        primary=38758, extra=[], category="event",
        question="When Kunti invoked Surya, what did she keep gazing at without growing tired of its beauty?",
        why="38758: her mind and eyes stayed fixed on the morning sun and she felt no satiety.",
        flags=[],
    ),
    dict(
        primary=39535, extra=[], category="event",
        question="In Virata's court, what physical insult did Draupadi say the suta's son had given her?",
        why="39535: Draupadi says the suta's son kicked her.",
        flags=[],
    ),
    dict(
        primary=39839, extra=[], category="location",
        question="Where in Virata's palace did Draupadi find Bhima waiting?",
        why="39839: she saw Bhima standing at the kitchen door.",
        flags=[],
    ),
    dict(
        primary=41154, extra=[], category="event",
        question="Just before Kurukshetra, why did the earth become crowded with kings?",
        why="41154: kings had gathered either for the Kauravas or for the Pandavas.",
        flags=[],
    ),
    dict(
        primary=44152, extra=[], category="advice_teaching",
        question="Instead of fighting, what did Krishna propose the visiting kings do together before returning to their own countries?",
        why="44152: dine and drink together, then return to their own homes.",
        flags=[],
    ),
    dict(
        primary=47715, extra=[], category="reason",
        question="What amazed Dhritarashtra about his own heart after he heard that Bhishma had been slain?",
        why="47715: he notes that his heart does not burst even on hearing Bhishma's death.",
        flags=[],
    ),
    dict(
        primary=49930, extra=[], category="event",
        question="On that morning of battle, which Kaurava leaders did Shikhandin attack with Virata, and which leaders did Arjuna attack?",
        why="49930: Shikhandin with Virata attacked Bhishma; Arjuna attacked Drona, Kripa, Vikarna, and other kings.",
        flags=["other"],
        flag_notes="Battle-cluster overlap with other Kurukshetra action questions.",
    ),
    dict(
        primary=53471, extra=[], category="event",
        question="After Arjuna slew Bhagadatta, what did he then do to the rest of the Kaurava warriors?",
        why="53471: after slaying that king, Arjuna began crushing the remaining warriors like a gale crushing trees.",
        flags=[],
    ),
    dict(
        primary=61605, extra=[], category="event",
        question="After Arjuna challenged Ashvatthaman, what did Krishna do with Arjuna?",
        why="61605: Krishna drove Arjuna to Drona's son's presence, as wind carries Indra to a sacrifice.",
        flags=[],
    ),
    dict(
        primary=68145, extra=[], category="reason",
        question="After Duryodhana had fallen, what mistake did the speaker say Yudhishthira had made about deciding the war?",
        why="68145: Yudhishthira made the whole result depend on the victory or defeat of only one warrior.",
        flags=[],
    ),
    dict(
        primary=69345, extra=[], category="advice_teaching",
        question="Why was Dhritarashtra told not to grieve for the slain Kuru heroes?",
        why="69345: their destruction was inevitable, and those heroes had already attained the highest end.",
        flags=["other"],
        flag_notes="Related 'why grieve' cluster with Arjuna's later speeches to Yudhishthira (rows 70460, 70066).",
    ),
    dict(
        primary=70460, extra=[], category="advice_teaching",
        question="After Yudhishthira had won the kingdom by Kshatriya duty, why did Arjuna ask him not to keep grieving?",
        why="70460: he has acquired hard-won sovereignty and defeated his foes, so grief is out of place.",
        flags=["other"],
        flag_notes="Same Rajadharmanushasana consolation cluster as row 70066.",
    ),
    dict(
        primary=74469, extra=[], category="advice_teaching",
        question="According to this teaching on enmity, why is it unsafe to rely on an old enemy's friendly words?",
        why="74469: enmity from mutual injury never dies; people perish from misplaced confidence, so they should not meet again.",
        flags=[],
    ),
    dict(
        primary=87154, extra=[], category="advice_teaching",
        question="According to this teaching, why do ascetics regarded as wise refuse meat as food?",
        why="87154 says they refrain because of the faults attached to eating meat, which the speaker then offers to explain.",
        flags=["uncertain_gold"],
        flag_notes="The verse announces the faults but does not list them. Review whether announcing the reason to refrain is enough of an answer.",
    ),
    dict(
        primary=89988, extra=[], category="factual",
        question="In this enumeration of foremost beings, who rules the waters, the storm-gods, hot bodies, and luminous bodies?",
        why="89988: Varuna of waters, Indra of Maruts, Arka of hot bodies, Indra of luminous bodies.",
        flags=["ambiguous_answer"],
        flag_notes="Indra appears twice (Maruts and luminous bodies). Review whether the last item is Indra or Indu/moon; Dutt prints 'Indra of all luminous bodies' while the Sanskrit has indu.",
    ),
    dict(
        primary=92302, extra=[], category="event",
        question="After Vyasa spoke to him, what did Dhritarashtra tell Yudhishthira about his own grief?",
        why="92302: Dhritarashtra says that through Yudhishthira's grace, grief no longer stands in his way.",
        flags=[],
    ),
    dict(
        primary=21361, extra=[], category="event",
        question="When Vyasa arrived at the snake sacrifice, what kind of seat did the king offer him?",
        why="21361: with the sadasyas' approval, the king offered the rishi a golden seat, as Indra did for Brihaspati.",
        flags=[],
    ),
    dict(
        primary=27978, extra=[], category="event",
        question="In the Shishupala episode, how did some of the assembled warriors describe Bhishma after he had spoken?",
        why="27978: some called Bhishma a sinful, boastful old man who deserved no pardon.",
        flags=[],
    ),
    dict(
        primary=28801, extra=[], category="reason",
        question="After the dice-game injuries, why did the speaker say the Pandavas would not forgive the Kauravas?",
        why="28801: the persecuted Pandavas will not forgive those injuries, and none of them can forgive the insult to Draupadi.",
        flags=["other"],
        flag_notes="Dice-game aftermath cluster with the elders' 'Fie' reaction (pilot_03) and Dhrishtadyumna's killer list (pilot_06). Different questions, same broader episode.",
    ),
    dict(
        primary=31560, extra=[], category="event",
        question="If Nala did not want to play dice, what alternative contest did Pushkara propose to settle their quarrel?",
        why="31560 proposes a duel with weapons instead of dice so that peace would come by single combat.",
        flags=[],
    ),
    dict(
        primary=33490, extra=[], category="factual",
        question="In this description, to which two Vrishni heroes is Arjuna said to be equal?",
        why="33490 equates Arjuna with invincible Sankarshana (Balarama) and Vasudeva (Krishna).",
        flags=[],
    ),
    dict(
        primary=35635, extra=[], category="event",
        question="Besides restoring Indradyumna to heaven, whom did Markandeya say Krishna had rescued from hell?",
        why="35635: Krishna rescued the royal sage Nriga from hell and caused him to regain heaven.",
        flags=[],
    ),
    dict(
        primary=35783, extra=[], category="event",
        question="Whom did Brihadashva install on the throne when the time came?",
        why="35783: Brihadashva installed his son Kuvalashva as king.",
        flags=[],
    ),
    dict(
        primary=39106, extra=[], category="event",
        question="After Dharma spoke at the lake, what happened to him, and what happened to the Pandava brothers?",
        why="39106: Dharma disappeared, and the Pandavas were reunited after an agreeable sleep.",
        flags=[],
    ),
    dict(
        primary=41391, extra=[], category="factual",
        question="In the story of Indra's defeat, whom was the lady told to choose as husband instead of Indra, and why?",
        why="41391: Nahusha, king of the gods, is said to be superior to Indra, so she should choose him.",
        flags=[],
    ),
    dict(
        primary=21457, extra=[], category="philosophical",
        question="What spiritual result is promised to people who tell Vyasa's history to others, or who listen to it?",
        why="21457: reciters and hearers cannot be affected by the fruits of good or bad deeds.",
        flags=["other"],
        flag_notes="Meta-textual claim about the Mahabharata itself rather than a story event. Review whether this is an appropriate realistic user question.",
    ),
    dict(
        primary=45975, extra=[], category="event",
        question="With which companions did Balarama enter the Pandava camp?",
        why="45975: Halayudha entered the Pandava camp with Akrura, Gada, Samba, Uddhava, and others.",
        flags=[],
    ),
    dict(
        primary=48559, extra=[], category="philosophical",
        question="According to this Gita teaching, what is said of a person without egoism whose mind is unsullied, even if he kills many people?",
        why="48559: such a person does not kill and is not bound by action, even if he kills all these people.",
        flags=["other"],
        flag_notes="Related Gita cluster with the peace-through-renouncing-desire teaching (row 48051).",
    ),
    dict(
        primary=49872, extra=[], category="reason",
        question="According to Sanjaya, what was Bhishma trying to protect by fighting so hard that morning?",
        why="49872: Bhishma fought in order to save Dhritarashtra's sons from their fear of Bhimasena.",
        flags=["other"],
        flag_notes="Kurukshetra battle cluster with other Bhishma-fight questions.",
    ),
    dict(
        primary=58601, extra=[], category="event",
        question="What reminder is given here about Karna's earlier defeat, along with the Kaurava army, during the year the Pandavas spent in disguise?",
        why="58601: Partha defeated all the assembled Kauravas, including Karna and his younger brother.",
        flags=[],
    ),
    dict(
        primary=24006, extra=[], category="event",
        question="At the trial of arms, how did the spectators react to Duryodhana's speech, and what happened in the sky afterward?",
        why="24006: a loud approving cheer arose, and at that time the sun went down.",
        flags=["other"],
        flag_notes="Thin unique fact (cheers plus sunset). Review whether this is too weak for the benchmark.",
    ),
    dict(
        primary=27366, extra=[], category="event",
        question="How did Jarasandha say he was willing to fight Krishna, Bhima, and Arjuna?",
        why="27366: army against army, or one against one, two, or three, together or separately.",
        flags=[],
    ),
    dict(
        primary=49255, extra=[], category="event",
        question="What did Abhimanyu do when he saw Dhrishtadyumna engaged with Ashvatthaman?",
        why="49255: Subhadra's son ran up, showering sharp arrows.",
        flags=["other"],
        flag_notes="Generic battle-action verse; review distinctiveness.",
    ),
    dict(
        primary=55011, extra=[], category="factual",
        question="What sounds from Drona's pavilion did Dhritarashtra say he used to hear constantly but could no longer hear?",
        why="55011 lists bow-twang, Vedic recitation, shafts, swords, and chariot-rattle as sounds of Drona's pavilion that are now gone.",
        flags=["uncertain_gold", "other"],
        flag_notes="English is garbled ('were the be heard'). Neighboring verses also say other camps' sounds are gone; gold is limited to Drona's pavilion only.",
    ),
    dict(
        primary=63015, extra=[], category="event",
        question="When Arjuna saw the Samsaptakas formed against him, how did he respond?",
        why="63015: angry Arjuna rushed at them, drawing Gandiva.",
        flags=["other"],
        flag_notes="Generic battle-action verse.",
    ),
    dict(
        primary=66277, extra=[], category="event",
        question="After Yudhishthira killed Shalya, to what old victory was the army's praise of him compared?",
        why="66277: they praised him as the gods once praised Indra after the destruction of Vritra.",
        flags=[],
    ),
    dict(
        primary=68327, extra=[], category="factual",
        question="Who did Yudhishthira say could endure the Brahmastras of Drona and Karna, when even Indra could not?",
        why="68327: no one else, not even thunder-wielding Purandara (Indra), except Krishna.",
        flags=[],
    ),
    dict(
        primary=70066, extra=[], category="advice_teaching",
        question="After conquering the earth, what did Arjuna say it was foolish for a royal-born king like Yudhishthira to do?",
        why="70066: abandoning virtue and profit to go live in the forest.",
        flags=["other"],
        flag_notes="Same consolation cluster as row 70460 (Arjuna urging Yudhishthira not to renounce the kingdom).",
    ),
    dict(
        primary=91520, extra=[], category="factual",
        question="Which members of the Pandava household did not know about Bhima's harsh treatment of Dhritarashtra in the palace?",
        why="91520: Yudhishthira, Arjuna, Kunti, Draupadi, and the twins did not know it.",
        flags=["uncertain_gold", "other"],
        flag_notes="Source English is truncated ('was to'). The referent of 'it' depends on neighboring verses about Bhima's verbal cruelty.",
    ),
    dict(
        primary=36753, extra=[], category="reason",
        question="Why, in Dhritarashtra's lament, would a man who had already reached heaven in his own body come back to earth?",
        why="36753: a man who has gone to heaven in his own body would not wish to return unless he saw the Kurus at the point of death by Time.",
        flags=["ambiguous_answer"],
        flag_notes="The verse never names the returned man. Neighbor 36752 identifies him as Arjuna. Review whether leaving him unnamed makes the gold too hard, or whether naming Arjuna would over-specify.",
    ),
    dict(
        primary=35523, extra=[], category="philosophical",
        question="What three hardships did Baka say even the deathless still have to endure?",
        why="35523: living with the disagreeable, separation from the beloved, and association with the wicked.",
        flags=[],
    ),
    dict(
        primary=22368, extra=[], category="character_action",
        question="What did Sharmishtha ask king Yayati to do so that she could become a mother?",
        why="22368: save her from sin, protect her virtue, and make her a mother by him.",
        flags=[],
    ),
    dict(
        primary=21820, extra=[21819], category="reason",
        question="According to this account of Duryodhana as a portion of Kali, what role did he play in starting the great destruction?",
        why="21820: he fanned the fire of hostility, the great fire of destruction.",
        extra_why="21819 independently says he was the cause of the slaughter of all creatures and ruined the earth.",
        flags=["other"],
        flag_notes="Uses neighboring 21818 for the Kali identification in the question wording. Review whether that extra context is fair.",
    ),
    dict(
        primary=20403, extra=[], category="factual",
        question="After Rahu's head was cut off, what enmity began, and what does that head still do to the sun and moon?",
        why="20403: lasting enmity with Chandra and Surya; the head still swallows them, causing eclipses.",
        flags=[],
    ),
    dict(
        primary=20954, extra=[], category="event",
        question="What fate did Kashyapa say was coming that very day for king Parikshit?",
        why="20954: Takshaka will today burn Parikshit of the Kuru race.",
        flags=[],
    ),
    dict(
        primary=21560, extra=[], category="factual",
        question="Where was this cosmic being said to have been born, and for what purpose in the world?",
        why="21560: he was born in the Andhaka-Vrishni race to increase piety.",
        flags=["ambiguous_answer"],
        flag_notes="The verse does not name Krishna in this row; the identification is from context. Review whether the question should name Krishna or leave it as 'this great being'.",
    ),
    dict(
        primary=22009, extra=[], category="factual",
        question="What future was predicted for the son Shakuntala would bear to Dushyanta?",
        why="22009: the son will be mighty and will rule the whole earth bounded by the sea.",
        flags=[],
    ),
    dict(
        primary=24436, extra=[], category="character_action",
        question="What did Bhima promise Hidimba's sister he would do to the rakshasa while she watched?",
        why="24436: he tells her not to fear and promises to kill the cannibal before her eyes.",
        flags=[],
    ),
    dict(
        primary=25737, extra=[], category="character_action",
        question="What did Dhritarashtra tell Duryodhana about revealing his thoughts to Vidura?",
        why="25737: he wants to act as they wish but does not want to show his thoughts to Vidura even by his demeanor.",
        flags=["other"],
        flag_notes="Same plotting conversation as the Drupada-bribe question (row 25741).",
    ),
    dict(
        primary=26063, extra=[], category="event",
        question="In the Naga house, what rite did Arjuna perform at the fire, and how did Agni respond?",
        why="26063: Arjuna poured libations before the fire; Agni was pleased by his fearlessness.",
        flags=[],
    ),
    dict(
        primary=26393, extra=[], category="reason",
        question="When Agni asked Arjuna for help burning the Khandava forest, what problem did Arjuna say he had with his bow?",
        why="26393: he has no bow suited to the strength of his arms and able to bear the force he puts forth in battle.",
        flags=[],
    ),
    dict(
        primary=29058, extra=[], category="event",
        question="When people followed Yudhishthira into exile out of grief, what did he tell them to do?",
        why="29058: go back; let their affection be directed to the relatives left behind as pledges.",
        flags=["other"],
        flag_notes="Source English is broken ('Go back. heart be...'). Same forest-departure cluster as the farewell question (row 28889).",
    ),
    dict(
        primary=29690, extra=[], category="location",
        question="Where did Krishna say he found Shalva standing with the Saubha car?",
        why="29690: in the midst of the ocean, on the Saubha car among heaving billows.",
        flags=["other"],
        flag_notes="Same Saubha episode as Krishna cutting off heads (row 29736).",
    ),
    dict(
        primary=30196, extra=[], category="philosophical",
        question="When Bhima urged action in the forest, what did Yudhishthira say about controlling the mind and about what had already happened?",
        why="30196: the mind cannot be controlled when bound by manliness, pride, and prowess; what happened was preordained.",
        flags=[],
    ),
    dict(
        primary=30788, extra=[], category="event",
        question="After Pushkara cheated Nala at dice, where did Nala go to live, and who went with him?",
        why="30788: he dwelt in the forest with his wife, pressed by distress.",
        flags=["other"],
        flag_notes="Nala cluster with the later dice-or-duel question (row 31560). Different moments.",
    ),
    dict(
        primary=19485, extra=[], category="character_action",
        question="At the opening of the Mahabharata, what did Dhritarashtra tell Sanjaya about his own attitude toward war and the destruction of his race?",
        why="19485: he says his mind was never for war and he took no pleasure in the destruction of his family.",
        flags=["other"],
        flag_notes="Anukramanika framing speech rather than an in-story event verse. Review suitability.",
    ),
    dict(
        primary=20536, extra=[], category="location",
        question="When the ocean rose around them, which island did the snakes reach together with their mother?",
        why="20536 names the island Ramaniyaka.",
        flags=[],
    ),
    dict(
        primary=21207, extra=[], category="factual",
        question="Who served as the Hota priest at Janamejaya's snake sacrifice, and from which lineage did he come?",
        why="21207: Chanda Bhargava, born in the race of Chyavana, was the Hota.",
        flags=[],
    ),
]


PILOT_CATEGORIES = {
    "pilot_01": "location",
    "pilot_02": "factual",
    "pilot_03": "event",
    "pilot_04": "factual",
    "pilot_05": "location",
    "pilot_06": "character_action",
    "pilot_07": "reason",
    "pilot_08": "philosophical",
    "pilot_09": "advice_teaching",
    "pilot_10": "philosophical",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOP and len(t) > 2]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def content_heavy_ngrams(token_ngrams: set[tuple[str, ...]], min_content: int) -> set[tuple[str, ...]]:
    kept = set()
    for gram in token_ngrams:
        n_content = sum(1 for t in gram if t not in STOP and len(t) > 2)
        if n_content >= min_content:
            kept.add(gram)
    return kept


def leakage_scores(question: str, en: str) -> dict:
    """Flag near-copies of Dutt English, not merely shared names or short function phrases."""
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
    suspicious = bool(shared6) or bool(shared4) or (bool(shared5) and jacc >= 0.28) or (jacc >= 0.42 and coverage >= 0.70)
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
    if pd.isna(chapter):
        chapter_out = None
    else:
        chapter_out = str(chapter)
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


def main() -> None:
    pilot = json.loads(PILOT_PATH.read_text())
    sample = json.loads(SAMPLE_PATH.read_text())
    frozen = {int(e["row_index"]) for e in sample["examples"]}
    df = pd.read_csv(CSV_PATH)

    questions = []
    for i, pq in enumerate(pilot["questions"], start=1):
        item = {
            "question_id": f"r100_{i:03d}",
            "source_id": pq["id"],
            "question": pq["question"],
            "primary_row_index": int(pq["primary_row_index"]),
            "relevant_row_indices": [int(x) for x in pq["relevant_row_indices"]],
            "book": pq["book"],
            "chapter": pq["chapter"],
            "shloka": pq["shloka"],
            "sn": pq["sn"],
            "en": pq["en"],
            "relevance_justification": pq["why_relevant"],
            "additional_gold_justification": None if len(pq["relevant_row_indices"]) == 1 else pq["why_relevant"],
            "question_category": PILOT_CATEGORIES[pq["id"]],
            "uncertainty_flags": [],
            "from_approved_pilot": True,
        }
        questions.append(item)

    assert len(NEW_QUESTIONS) == 90, len(NEW_QUESTIONS)
    for j, nq in enumerate(NEW_QUESTIONS, start=11):
        primary = int(nq["primary"])
        extra = [int(x) for x in nq.get("extra", [])]
        gold = [primary] + [x for x in extra if x != primary]
        fields = csv_fields(df.loc[primary])
        item = {
            "question_id": f"r100_{j:03d}",
            "source_id": None,
            "question": nq["question"],
            "primary_row_index": primary,
            "relevant_row_indices": gold,
            "book": fields["book"],
            "chapter": fields["chapter"],
            "shloka": fields["shloka"],
            "sn": fields["sn"],
            "en": fields["en"],
            "relevance_justification": nq["why"],
            "additional_gold_justification": nq.get("extra_why"),
            "question_category": nq["category"],
            "uncertainty_flags": list(nq.get("flags", [])),
            "flag_notes": nq.get("flag_notes"),
            "from_approved_pilot": False,
        }
        questions.append(item)

    # Automatic leakage check — do not silently accept suspicious items.
    auto_leakage = []
    for q in questions:
        scores = leakage_scores(q["question"], q["en"])
        q["leakage_check"] = scores
        if scores["suspicious"]:
            # Approved pilot questions were already human-reviewed; report but do not auto-flag them.
            if q.get("from_approved_pilot"):
                scores["suspicious_but_approved_pilot"] = True
            else:
                flags = set(q.get("uncertainty_flags") or [])
                flags.add("possible_english_leakage")
                q["uncertainty_flags"] = sorted(flags)
                auto_leakage.append(q["question_id"])

    # Near-duplicate question meaning: high content-token Jaccard between questions.
    near_dups = []
    contents = [(q["question_id"], set(content_tokens(q["question"]))) for q in questions]
    for i, (id_a, set_a) in enumerate(contents):
        for id_b, set_b in contents[i + 1 :]:
            union = set_a | set_b
            if not union:
                continue
            jacc = len(set_a & set_b) / len(union)
            if jacc >= 0.45:
                near_dups.append((id_a, id_b, round(jacc, 3)))
                for qid in (id_a, id_b):
                    q = next(x for x in questions if x["question_id"] == qid)
                    flags = set(q.get("uncertainty_flags") or [])
                    flags.add("near_duplicate_meaning")
                    q["uncertainty_flags"] = sorted(flags)

    # Validation
    errors = []
    if len(questions) != 100:
        errors.append(f"expected 100 questions, got {len(questions)}")

    for i, pq in enumerate(pilot["questions"]):
        q = questions[i]
        if q["question"] != pq["question"]:
            errors.append(f"pilot question text changed at {q['question_id']}")
        if q["primary_row_index"] != int(pq["primary_row_index"]):
            errors.append(f"pilot primary changed at {q['question_id']}")
        if q["relevant_row_indices"] != [int(x) for x in pq["relevant_row_indices"]]:
            errors.append(f"pilot gold changed at {q['question_id']}")
        if q["sn"] != pq["sn"] or q["en"] != pq["en"]:
            errors.append(f"pilot sn/en changed at {q['question_id']}")

    qtexts = [q["question"].strip() for q in questions]
    if any(not t for t in qtexts):
        errors.append("empty question text")
    if len(qtexts) != len(set(qtexts)):
        errors.append("duplicate question texts")

    primaries = [q["primary_row_index"] for q in questions]
    if len(primaries) != len(set(primaries)):
        dups = [p for p, c in Counter(primaries).items() if c > 1]
        errors.append(f"duplicate primary_row_index: {dups}")

    for q in questions:
        p = q["primary_row_index"]
        gold = q["relevant_row_indices"]
        if not gold:
            errors.append(f"{q['question_id']} missing gold")
        if p not in gold:
            errors.append(f"{q['question_id']} primary not in relevant_row_indices")
        if p not in df.index:
            errors.append(f"{q['question_id']} primary {p} not in combined.csv")
        if p not in frozen:
            errors.append(f"{q['question_id']} primary {p} not in frozen 500")
        for g in gold:
            if g not in df.index:
                errors.append(f"{q['question_id']} gold {g} not in combined.csv")
        if q["book"] not in {"ramayana", "mahabharata"}:
            errors.append(f"{q['question_id']} bad book")

    book_counts = Counter(q["book"] for q in questions)
    cat_counts = Counter(q["question_category"] for q in questions)
    single_gold = sum(1 for q in questions if len(q["relevant_row_indices"]) == 1)
    multi_gold = 100 - single_gold

    flag_buckets = {
        "possible_english_leakage": [],
        "ambiguous_answer": [],
        "uncertain_gold": [],
        "near_duplicate_meaning": [],
        "other": [],
    }
    for q in questions:
        for f in q.get("uncertainty_flags") or []:
            if f in flag_buckets:
                flag_buckets[f].append(q["question_id"])

    flagged_ids = sorted({qid for ids in flag_buckets.values() for qid in ids})
    flagged_questions = [q for q in questions if q["question_id"] in flagged_ids]

    payload = {
        "name": "realistic_eval_100_candidate",
        "status": "candidate_awaiting_human_review",
        "n": 100,
        "scientifically_validated": False,
        "book_counts": dict(book_counts),
        "source_pool": "eval/data/faiss_baseline_sample.json",
        "corpus": "res/combined.csv",
        "gold_id_field": "row_index",
        "approved_pilot_source": "eval/data/realistic_pilot_10.json",
        "purpose": "Candidate 100-question realistic-retrieval benchmark. First 10 questions are the approved pilot, copied exactly. Not frozen. Not scientifically validated. Do not run FAISS until after human review.",
        "labeling_rule": "A verse is gold only if retrieving it would reasonably count as a successful answer to the user's question. Nearby scene, background, or explanatory verses are not gold.",
        "rules": [
            "Queries are English user-style questions, not the exact CSV en field.",
            "primary_row_index must come from the frozen 500 pool and must be unique across the 100 questions.",
            "relevant_row_indices is the full gold set (primary first). Extra gold only if a verse independently answers.",
            "Do not treat this file as production retrieval input.",
            "Do not treat automatic validation as scientific validation.",
        ],
        "question_category_counts": dict(cat_counts),
        "single_gold_count": single_gold,
        "multi_gold_count": multi_gold,
        "automatic_validation_passed": not errors,
        "automatic_validation_errors": errors,
        "questions": questions,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    review = {
        "status": "candidate_awaiting_human_review",
        "scientifically_validated": False,
        "automatic_validation_passed": not errors,
        "automatic_validation_errors": errors,
        "total_questions": 100,
        "ramayana_count": book_counts.get("ramayana", 0),
        "mahabharata_count": book_counts.get("mahabharata", 0),
        "single_gold_count": single_gold,
        "multi_gold_count": multi_gold,
        "question_category_distribution": dict(cat_counts),
        "flag_counts": {k: len(v) for k, v in flag_buckets.items()},
        "flagged_question_ids": flagged_ids,
        "auto_leakage_question_ids": auto_leakage,
        "near_duplicate_pairs": near_dups,
        "flagged_questions": [
            {
                "question_id": q["question_id"],
                "question": q["question"],
                "primary_row_index": q["primary_row_index"],
                "relevant_row_indices": q["relevant_row_indices"],
                "book": q["book"],
                "question_category": q["question_category"],
                "uncertainty_flags": q.get("uncertainty_flags"),
                "flag_notes": q.get("flag_notes"),
                "en": q["en"],
                "leakage_check": q.get("leakage_check"),
            }
            for q in flagged_questions
        ],
    }
    REPORT_PATH.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n")

    print("WROTE", OUT_PATH)
    print("WROTE", REPORT_PATH)
    print("errors", errors or "NONE")
    print("books", dict(book_counts))
    print("gold", single_gold, "single /", multi_gold, "multi")
    print("categories", dict(cat_counts))
    print("flag_counts", {k: len(v) for k, v in flag_buckets.items()})
    print("flagged_ids", flagged_ids)
    print("auto_leakage", auto_leakage)
    print("near_dups", near_dups)


if __name__ == "__main__":
    main()
