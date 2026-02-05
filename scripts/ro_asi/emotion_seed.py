#!/usr/bin/env python3
"""
Romanian Emotion Seed Lexicon for RO-ASI Benchmark.

Defines manual emotion seed words with masculine/feminine gender variations,
mapped to Plutchik's 8 basic emotion categories.

Based on RoEmoLex paper (DOI: 10.24193/subbi.2017.2.04) examples and
common Romanian emotion vocabulary.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

# Plutchik's 8 basic emotions
EMOTIONS = [
    "anger",       # Furie
    "anticipation", # Anticipare
    "disgust",     # Dezgust
    "fear",        # Frică
    "joy",         # Bucurie
    "sadness",     # Tristețe
    "surprise",    # Surpriză
    "trust"        # Încredere
]

# Emotion adjectives with gender variations (masculine, feminine)
# Format: (masculine, feminine, english, [emotions])
EMOTION_ADJECTIVES = [
    # Joy
    ("fericit", "fericită", "happy", ["joy"]),
    ("bucuros", "bucuroasă", "joyful", ["joy"]),
    ("încântat", "încântată", "delighted", ["joy"]),
    ("vesel", "veselă", "cheerful", ["joy"]),
    ("entuziasmat", "entuziasmată", "excited", ["joy", "anticipation"]),
    ("mulțumit", "mulțumită", "satisfied", ["joy"]),
    ("împlinit", "împlinită", "fulfilled", ["joy"]),
    ("extaziat", "extaziată", "ecstatic", ["joy"]),

    # Sadness
    ("trist", "tristă", "sad", ["sadness"]),
    ("demoralizat", "demoralizată", "demoralized", ["sadness"]),
    ("deprimat", "deprimată", "depressed", ["sadness"]),
    ("dezamăgit", "dezamăgită", "disappointed", ["sadness"]),
    ("abătut", "abătută", "dejected", ["sadness"]),
    ("mâhnit", "mâhnită", "grieved", ["sadness"]),
    ("îndurerat", "îndurerată", "sorrowful", ["sadness"]),
    ("melancolie", "melancolică", "melancholic", ["sadness"]),
    ("amărât", "amărâtă", "bitter", ["sadness"]),
    ("singur", "singură", "lonely", ["sadness"]),
    ("plictisit", "plictisită", "bored", ["sadness", "disgust"]),
    ("obosit", "obosită", "tired", ["sadness"]),

    # Fear
    ("speriat", "speriată", "scared", ["fear"]),
    ("îngrijorat", "îngrijorată", "worried", ["fear"]),
    ("anxios", "anxioasă", "anxious", ["fear"]),
    ("înfricoșat", "înfricosată", "frightened", ["fear"]),
    ("îngrozit", "îngrozită", "terrified", ["fear"]),
    ("panicat", "panicată", "panicked", ["fear"]),
    ("temător", "temătoare", "fearful", ["fear"]),
    ("nervos", "nervoasă", "nervous", ["fear", "anger"]),
    ("stresat", "stresată", "stressed", ["fear", "anger"]),
    ("confuz", "confuză", "confused", ["fear", "surprise"]),
    ("nesigur", "nesigură", "insecure", ["fear"]),
    ("îngrozit", "îngrozită", "horrified", ["fear"]),

    # Anger
    ("furios", "furioasă", "angry", ["anger"]),
    ("supărat", "supărată", "upset", ["anger"]),
    ("înfuriat", "înfuriată", "enraged", ["anger"]),
    ("iritat", "iritată", "irritated", ["anger"]),
    ("frustrat", "frustrată", "frustrated", ["anger"]),
    ("indignat", "indignată", "indignant", ["anger"]),
    ("revoltat", "revoltată", "revolted", ["anger"]),
    ("turbat", "turbată", "furious", ["anger"]),
    ("exasperat", "exasperată", "exasperated", ["anger"]),

    # Disgust
    ("dezgustat", "dezgustată", "disgusted", ["disgust"]),
    ("scârbit", "scârbită", "repulsed", ["disgust"]),
    ("oripilat", "oripilată", "appalled", ["disgust"]),
    ("îngretoșat", "îngretoșată", "nauseated", ["disgust"]),

    # Surprise
    ("surprins", "surprinsă", "surprised", ["surprise"]),
    ("uimit", "uimită", "amazed", ["surprise"]),
    ("uluit", "uluită", "stunned", ["surprise"]),
    ("șocat", "șocată", "shocked", ["surprise"]),
    ("năucit", "năucită", "dazed", ["surprise"]),
    ("stupefiat", "stupefiată", "stupefied", ["surprise"]),

    # Trust
    ("încrezător", "încrezătoare", "trusting", ["trust"]),
    ("sigur", "sigură", "confident", ["trust"]),
    ("calm", "calmă", "calm", ["trust"]),
    ("liniștit", "liniștită", "peaceful", ["trust"]),
    ("relaxat", "relaxată", "relaxed", ["trust"]),
    ("senin", "senină", "serene", ["trust"]),

    # Anticipation
    ("optimist", "optimistă", "optimistic", ["anticipation", "joy"]),
    ("pesimist", "pesimistă", "pessimistic", ["anticipation", "sadness"]),
    ("nerăbdător", "nerăbdătoare", "eager", ["anticipation"]),
    ("curios", "curioasă", "curious", ["anticipation"]),

    # Mixed emotions
    ("vinovat", "vinovată", "guilty", ["sadness", "fear"]),
    ("rușinat", "rușinată", "ashamed", ["sadness", "fear"]),
    ("mândru", "mândră", "proud", ["joy", "trust"]),
    ("gelos", "geloasă", "jealous", ["anger", "sadness"]),
    ("iubit", "iubită", "loved", ["joy", "trust"]),
    ("recunoscător", "recunoscătoare", "grateful", ["joy", "trust"]),
    ("invidios", "invidioasă", "envious", ["anger", "sadness"]),
]

# Emotion nouns (gender-invariable or with common form)
# Format: (word, english, [emotions])
EMOTION_NOUNS = [
    ("bucurie", "joy", ["joy"]),
    ("tristețe", "sadness", ["sadness"]),
    ("frică", "fear", ["fear"]),
    ("teamă", "fear/dread", ["fear"]),
    ("furie", "fury", ["anger"]),
    ("mânie", "anger", ["anger"]),
    ("surpriză", "surprise", ["surprise"]),
    ("încredere", "trust", ["trust"]),
    ("dezgust", "disgust", ["disgust"]),
    ("scârbă", "repulsion", ["disgust"]),
    ("anxietate", "anxiety", ["fear"]),
    ("speranță", "hope", ["anticipation", "joy"]),
    ("disperare", "despair", ["sadness", "fear"]),
    ("panică", "panic", ["fear"]),
    ("rușine", "shame", ["sadness", "fear"]),
    ("vină", "guilt", ["sadness"]),
    ("mândrie", "pride", ["joy"]),
    ("gelozie", "jealousy", ["anger", "sadness"]),
    ("invidie", "envy", ["anger", "sadness"]),
    ("iubire", "love", ["joy", "trust"]),
    ("ură", "hatred", ["anger", "disgust"]),
    ("groază", "terror", ["fear"]),
    ("chin", "anguish", ["sadness", "fear"]),
    ("durere", "pain", ["sadness"]),
    ("pace", "peace", ["trust"]),
    ("liniște", "calm", ["trust"]),
    ("neliniște", "unease", ["fear"]),
    ("recunoștință", "gratitude", ["joy", "trust"]),
    ("curiozitate", "curiosity", ["anticipation"]),
    ("așteptare", "anticipation", ["anticipation"]),
    ("nostalgie", "nostalgia", ["sadness"]),
    ("melancolie", "melancholy", ["sadness"]),
    ("entuziasm", "enthusiasm", ["joy", "anticipation"]),
    ("plăcere", "pleasure", ["joy"]),
    ("neplăcere", "displeasure", ["anger", "disgust"]),
    ("satisfacție", "satisfaction", ["joy"]),
    ("nemulțumire", "dissatisfaction", ["anger", "sadness"]),
]

# Common state words (invariable)
# Format: (word, english, [emotions])
STATE_WORDS = [
    ("bine", "good/well", ["joy"]),
    ("rău", "bad", ["sadness", "anger"]),
    ("groaznic", "terrible", ["fear", "disgust"]),
    ("minunat", "wonderful", ["joy"]),
    ("oribil", "horrible", ["fear", "disgust"]),
    ("grozav", "great", ["joy"]),
    ("teribil", "terrible", ["fear"]),
    ("îngrozitor", "terrifying", ["fear"]),
    ("extraordinar", "extraordinary", ["joy", "surprise"]),
]


def build_emotion_seed() -> Dict:
    """
    Build the emotion seed dictionary with all word forms and their emotions.

    Returns:
        Dictionary with structure:
        {
            "adjectives": [{"masculine": str, "feminine": str, "english": str, "emotions": list}],
            "nouns": [{"word": str, "english": str, "emotions": list}],
            "state_words": [{"word": str, "english": str, "emotions": list}],
            "word_to_emotions": {word: [emotions]},  # flattened lookup
            "all_words": [str],  # all unique word forms
            "statistics": {...}
        }
    """
    seed = {
        "adjectives": [],
        "nouns": [],
        "state_words": [],
        "word_to_emotions": {},
        "all_words": set(),
        "statistics": {
            "total_adjectives": len(EMOTION_ADJECTIVES),
            "total_nouns": len(EMOTION_NOUNS),
            "total_state_words": len(STATE_WORDS),
            "total_word_forms": 0,
            "emotions_coverage": {e: 0 for e in EMOTIONS}
        }
    }

    # Process adjectives (both gender forms)
    for masc, fem, eng, emotions in EMOTION_ADJECTIVES:
        seed["adjectives"].append({
            "masculine": masc,
            "feminine": fem,
            "english": eng,
            "emotions": emotions
        })
        # Add both forms to lookup and word list
        seed["word_to_emotions"][masc] = emotions
        seed["word_to_emotions"][fem] = emotions
        seed["all_words"].add(masc)
        seed["all_words"].add(fem)
        # Update emotion coverage
        for e in emotions:
            seed["statistics"]["emotions_coverage"][e] += 2  # both forms

    # Process nouns
    for word, eng, emotions in EMOTION_NOUNS:
        seed["nouns"].append({
            "word": word,
            "english": eng,
            "emotions": emotions
        })
        seed["word_to_emotions"][word] = emotions
        seed["all_words"].add(word)
        for e in emotions:
            seed["statistics"]["emotions_coverage"][e] += 1

    # Process state words
    for word, eng, emotions in STATE_WORDS:
        seed["state_words"].append({
            "word": word,
            "english": eng,
            "emotions": emotions
        })
        seed["word_to_emotions"][word] = emotions
        seed["all_words"].add(word)
        for e in emotions:
            seed["statistics"]["emotions_coverage"][e] += 1

    # Convert set to sorted list for JSON serialization
    seed["all_words"] = sorted(list(seed["all_words"]))
    seed["statistics"]["total_word_forms"] = len(seed["all_words"])

    return seed


def save_emotion_seed(output_path: Path) -> Dict:
    """Build and save the emotion seed to JSON file."""
    seed = build_emotion_seed()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)

    print(f"Saved emotion seed to {output_path}")
    print(f"  - Adjectives: {seed['statistics']['total_adjectives']} entries "
          f"({seed['statistics']['total_adjectives'] * 2} word forms)")
    print(f"  - Nouns: {seed['statistics']['total_nouns']} entries")
    print(f"  - State words: {seed['statistics']['total_state_words']} entries")
    print(f"  - Total unique word forms: {seed['statistics']['total_word_forms']}")
    print(f"  - Emotion coverage: {seed['statistics']['emotions_coverage']}")

    return seed


def load_emotion_seed(seed_path: Path) -> Dict:
    """Load emotion seed from JSON file."""
    with open(seed_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_words_by_emotion(seed: Dict, emotion: str) -> List[str]:
    """Get all words associated with a specific emotion."""
    return [word for word, emotions in seed["word_to_emotions"].items()
            if emotion in emotions]


if __name__ == "__main__":
    # Default output path
    output_path = Path(__file__).parent.parent.parent / "data" / "emotion_seed.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed = save_emotion_seed(output_path)

    # Print some examples
    print("\nSample words by emotion:")
    for emotion in EMOTIONS:
        words = get_words_by_emotion(seed, emotion)[:5]
        print(f"  {emotion}: {', '.join(words)}")
