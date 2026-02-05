#!/usr/bin/env python3
"""
Romanian "I Feel" pattern matcher for RO-ASI benchmark.

Implements pattern matching for Romanian affective state expressions:
- Primary: "mă simt [state]" variations (I feel)
- Secondary: "sunt [adjective]", "îmi este [noun]", etc.

Features:
- Diacritic normalization (ă→a, ș→s, etc.) for robust matching
- Case-insensitive matching
- Sentence extraction from matched texts
- Multiple pattern categories with priority
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass


# Romanian diacritic mappings for normalization
DIACRITIC_MAP = {
    'ă': 'a', 'â': 'a', 'î': 'i',
    'ș': 's', 'ț': 't',
    'Ă': 'A', 'Â': 'A', 'Î': 'I',
    'Ș': 'S', 'Ț': 'T',
    # Common OCR/typing errors
    'ş': 's', 'ţ': 't',  # cedilla variants
    'Ş': 'S', 'Ţ': 'T',
}


def remove_diacritics(text: str) -> str:
    """Remove Romanian diacritics from text for matching."""
    result = []
    for char in text:
        result.append(DIACRITIC_MAP.get(char, char))
    return ''.join(result)


def normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase and remove diacritics."""
    return remove_diacritics(text.lower())


@dataclass
class PatternMatch:
    """Represents a pattern match in text."""
    pattern_name: str
    pattern_category: str
    matched_text: str
    seed_word: str
    seed_word_normalized: str
    start_pos: int
    end_pos: int
    emotions: List[str]


# Pattern definitions
# Each pattern is (name, category, regex_template)
# {SEED} will be replaced with the seed word regex
# {MOD} allows optional adverb modifiers (foarte, mai, puțin, etc.)

# Common Romanian adverb modifiers that can appear between verb and adjective
# foarte (very), mai (more), putin/puțin (a little), cam (somewhat),
# destul de (quite), asa de (so), tot (still), deja (already)
MODIFIER_PATTERN = r'(?:(?:foarte|mai|putin|pu[tț]in|cam|destul\s+de|a[sș]a\s+de|tot|deja|chiar|at[aâ]t\s+de)\s+)?'

# Curated list of emotion NOUNS for "am [noun]" and "îmi este [noun]" patterns
# These are specifically emotion/state nouns, NOT past participles or general nouns
EMOTION_NOUNS_ONLY = {
    # Fear-related
    "frica", "frică", "teama", "teamă", "groaza", "groază", "panica", "panică",
    "anxietate", "neliniste", "neliniște", "ingrijorare", "îngrijorare",
    # Sadness-related
    "tristete", "tristețe", "durere", "suferinta", "suferință", "chin",
    "dezamagire", "dezamăgire", "melancolie", "nostalgie", "dor",
    # Joy-related
    "bucurie", "fericire", "placere", "plăcere", "satisfactie", "satisfacție",
    "entuziasm", "veselie", "voie", "chef",
    # Anger-related
    "furie", "manie", "mânie", "nervozitate", "iritare", "frustrare",
    "indignare", "revolta", "revoltă", "ura", "ură",
    # Disgust-related
    "dezgust", "scarba", "scârbă", "greata", "greață", "repulsie",
    # Surprise-related
    "surpriza", "surpriză", "uimire", "mirare", "stupoare",
    # Trust-related
    "incredere", "încredere", "siguranta", "siguranță", "liniste", "liniște",
    "pace", "calm",
    # Other emotions
    "rusine", "rușine", "vina", "vină", "gelozie", "invidie",
    "mandrie", "mândrie", "recunostinta", "recunoștință", "iubire",
    "speranta", "speranță", "disperare", "nerabdare", "nerăbdare",
    "curiozitate",
}

# Emotion mappings for curated nouns (normalized forms)
NOUN_EMOTION_MAP = {
    # Fear
    "frica": ["fear"], "teama": ["fear"], "groaza": ["fear"],
    "panica": ["fear"], "anxietate": ["fear"], "neliniste": ["fear"],
    "ingrijorare": ["fear"],
    # Sadness
    "tristete": ["sadness"], "durere": ["sadness"], "suferinta": ["sadness"],
    "chin": ["sadness"], "dezamagire": ["sadness"], "melancolie": ["sadness"],
    "nostalgie": ["sadness"], "dor": ["sadness"],
    # Joy
    "bucurie": ["joy"], "fericire": ["joy"], "placere": ["joy"],
    "satisfactie": ["joy"], "entuziasm": ["joy", "anticipation"],
    "veselie": ["joy"], "voie": ["joy"], "chef": ["joy", "anticipation"],
    # Anger
    "furie": ["anger"], "manie": ["anger"], "nervozitate": ["anger", "fear"],
    "iritare": ["anger"], "frustrare": ["anger", "sadness"],
    "indignare": ["anger"], "revolta": ["anger"], "ura": ["anger", "disgust"],
    # Disgust
    "dezgust": ["disgust"], "scarba": ["disgust"], "greata": ["disgust"],
    "repulsie": ["disgust"],
    # Surprise
    "surpriza": ["surprise"], "uimire": ["surprise"], "mirare": ["surprise"],
    "stupoare": ["surprise"],
    # Trust
    "incredere": ["trust"], "siguranta": ["trust"], "liniste": ["trust"],
    "pace": ["trust"], "calm": ["trust"],
    # Mixed
    "rusine": ["sadness", "fear"], "vina": ["sadness"],
    "gelozie": ["anger", "sadness"], "invidie": ["anger", "sadness"],
    "mandrie": ["joy"], "recunostinta": ["joy", "trust"],
    "iubire": ["joy", "trust"], "speranta": ["anticipation", "joy"],
    "disperare": ["sadness", "fear"], "nerabdare": ["anticipation"],
    "curiozitate": ["anticipation"],
}

# Pattern definitions
# Format: (name, category, regex_template, noun_only)
# noun_only=True means use EMOTION_NOUNS_ONLY instead of full seed
PATTERNS = [
    # Primary patterns: "mă simt" (I feel) - most direct
    ("ma_simt_present", "primary",
     r'\b(m[aă]\s+simt|me\s+simt)\s+' + MODIFIER_PATTERN + r'({SEED})\b', False),

    ("ma_simteam_imperfect", "primary",
     r'\b(m[aă]\s+sim[tț]eam|me\s+simteam)\s+' + MODIFIER_PATTERN + r'({SEED})\b', False),

    ("mam_simtit_perfect", "primary",
     r'\b(m-?am\s+sim[tț]it)\s+' + MODIFIER_PATTERN + r'({SEED})\b', False),

    ("ma_voi_simti_future", "primary",
     r'\b(m[aă]\s+voi\s+sim[tț]i|me\s+voi\s+simti)\s+' + MODIFIER_PATTERN + r'({SEED})\b', False),

    ("simt_ca", "primary",
     r'\b(simt\s+c[aă])\s+' + MODIFIER_PATTERN + r'({SEED})\b', False),

    # "simt [emotion_noun]" - e.g., "simt tristețe", "simt bucurie" - NOUN ONLY
    ("simt_noun", "primary",
     r'\b(simt)\s+' + MODIFIER_PATTERN + r'({SEED})\b', True),

    # "simțeam [emotion_noun]" - imperfect - NOUN ONLY
    ("simteam_noun", "primary",
     r'\b(sim[tț]eam)\s+' + MODIFIER_PATTERN + r'({SEED})\b', True),

    # First person plural
    ("ne_simtim_present", "primary",
     r'\b(ne\s+sim[tț]im)\s+' + MODIFIER_PATTERN + r'({SEED})\b', False),

    ("neam_simtit_perfect", "primary",
     r'\b(ne-?am\s+sim[tț]it)\s+' + MODIFIER_PATTERN + r'({SEED})\b', False),

    # Secondary patterns: "sunt" (I am)
    ("sunt_adj_present", "secondary",
     r'\b(sunt)\s+' + MODIFIER_PATTERN + r'({SEED})\b', False),

    ("eram_adj_imperfect", "secondary",
     r'\b(eram)\s+' + MODIFIER_PATTERN + r'({SEED})\b', False),

    ("am_fost_adj_perfect", "secondary",
     r'\b(am\s+fost)\s+' + MODIFIER_PATTERN + r'({SEED})\b', False),

    # First person plural "suntem"
    ("suntem_adj_present", "secondary",
     r'\b(suntem)\s+' + MODIFIER_PATTERN + r'({SEED})\b', False),

    # Dative constructions: "îmi este" (it is to me) - NOUN ONLY
    ("imi_este_present", "secondary",
     r'\b([îi]mi\s+este|imi\s+este)\s+' + MODIFIER_PATTERN + r'({SEED})\b', True),

    ("imi_era_imperfect", "secondary",
     r'\b([îi]mi\s+era|imi\s+era)\s+' + MODIFIER_PATTERN + r'({SEED})\b', True),

    # Short forms: "mi-e" (it's to me) - NOUN ONLY
    ("mie_short", "secondary",
     r'\b(mi-?e|mi-?i)\s+' + MODIFIER_PATTERN + r'({SEED})\b', True),

    # "am [noun]" (I have [emotion]) - NOUN ONLY to avoid participles
    ("am_noun_present", "secondary",
     r'\b(am)\s+' + MODIFIER_PATTERN + r'({SEED})\b', True),

    ("aveam_noun_imperfect", "secondary",
     r'\b(aveam)\s+' + MODIFIER_PATTERN + r'({SEED})\b', True),
]


def build_seed_regex(seed_words: List[str]) -> str:
    """
    Build regex alternation for seed words.

    Handles both original and normalized forms.
    """
    # Normalize all seed words and create unique set
    normalized_seeds = set()
    for word in seed_words:
        normalized_seeds.add(normalize_text(word))

    # Sort by length (longest first) to avoid partial matches
    sorted_seeds = sorted(normalized_seeds, key=len, reverse=True)

    # Escape special regex characters
    escaped = [re.escape(word) for word in sorted_seeds]

    return '|'.join(escaped)


def compile_patterns(
    seed_words: List[str],
    noun_only_words: Optional[Set[str]] = None
) -> List[Tuple[str, str, re.Pattern, bool]]:
    """
    Compile pattern regexes with seed words.

    Args:
        seed_words: Full list of seed words for general patterns
        noun_only_words: Restricted set for noun-only patterns (defaults to EMOTION_NOUNS_ONLY)

    Returns list of (name, category, compiled_pattern, noun_only) tuples.
    """
    if noun_only_words is None:
        noun_only_words = EMOTION_NOUNS_ONLY

    seed_regex = build_seed_regex(seed_words)
    noun_regex = build_seed_regex(list(noun_only_words))

    compiled = []
    for name, category, template, noun_only in PATTERNS:
        # Use noun-only regex for noun patterns, full regex otherwise
        current_regex = noun_regex if noun_only else seed_regex
        pattern_str = template.replace('{SEED}', current_regex)
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
            compiled.append((name, category, pattern, noun_only))
        except re.error as e:
            print(f"Warning: Failed to compile pattern {name}: {e}")

    return compiled


def extract_sentence(text: str, match_start: int, match_end: int) -> str:
    """
    Extract the sentence containing the match.

    Uses sentence boundaries: . ! ? and newlines.
    """
    # Find sentence start (look backwards)
    sentence_start = 0
    for i in range(match_start - 1, -1, -1):
        if text[i] in '.!?\n':
            sentence_start = i + 1
            break

    # Find sentence end (look forwards)
    sentence_end = len(text)
    for i in range(match_end, len(text)):
        if text[i] in '.!?\n':
            sentence_end = i + 1
            break

    return text[sentence_start:sentence_end].strip()


def find_original_word(original_text: str, normalized_match: str, approx_pos: int) -> str:
    """
    Find the original (non-normalized) word in the text.

    Args:
        original_text: Original text with diacritics
        normalized_match: The matched word (normalized)
        approx_pos: Approximate position in normalized text

    Returns:
        Original word from text
    """
    # Simple approach: normalize text around position and find word boundaries
    # This handles diacritic differences

    words_in_area = []
    text_segment = original_text[max(0, approx_pos - 50):approx_pos + 50]

    for word in re.findall(r'\b\w+\b', text_segment, re.UNICODE):
        if normalize_text(word) == normalized_match:
            return word

    # Fallback: return normalized
    return normalized_match


class PatternMatcher:
    """
    Romanian affective state pattern matcher.

    Usage:
        matcher = PatternMatcher(word_to_emotions)
        matches = matcher.find_matches(text)

        # Or with separate noun list from curated seed:
        matcher = PatternMatcher(word_to_emotions, noun_words=seed.get('nouns', []))
    """

    def __init__(
        self,
        word_to_emotions: Dict[str, List[str]],
        noun_words: Optional[List[str]] = None
    ):
        """
        Initialize matcher with emotion seed dictionary.

        Args:
            word_to_emotions: Dictionary mapping words to emotion lists
            noun_words: Optional list of noun words for noun-only patterns.
                       If None, uses EMOTION_NOUNS_ONLY constant.
        """
        self.word_to_emotions = word_to_emotions
        self.seed_words = list(word_to_emotions.keys())

        # Build normalized lookup
        self.normalized_to_original: Dict[str, List[str]] = {}
        self.normalized_to_emotions: Dict[str, List[str]] = {}

        for word, emotions in word_to_emotions.items():
            normalized = normalize_text(word)
            if normalized not in self.normalized_to_original:
                self.normalized_to_original[normalized] = []
            self.normalized_to_original[normalized].append(word)

            if normalized not in self.normalized_to_emotions:
                self.normalized_to_emotions[normalized] = []
            # Merge emotions from all original forms
            for e in emotions:
                if e not in self.normalized_to_emotions[normalized]:
                    self.normalized_to_emotions[normalized].append(e)

        # Determine noun words to use
        if noun_words is not None:
            noun_word_set = set(noun_words)
        else:
            noun_word_set = EMOTION_NOUNS_ONLY

        # Build normalized lookup for emotion nouns only
        self.noun_only_normalized = set()
        for word in noun_word_set:
            self.noun_only_normalized.add(normalize_text(word))

        # Compile patterns
        all_normalized_seeds = list(self.normalized_to_original.keys())
        self.compiled_patterns = compile_patterns(all_normalized_seeds, noun_word_set)

        # Count noun-only patterns
        noun_only_count = sum(1 for _, _, _, noun_only in self.compiled_patterns if noun_only)

        print(f"PatternMatcher initialized:")
        print(f"  Seed words: {len(self.seed_words)}")
        print(f"  Noun words: {len(noun_word_set)}")
        print(f"  Patterns: {len(self.compiled_patterns)} ({noun_only_count} noun-only)")

    def find_matches(
        self,
        text: str,
        extract_sentences: bool = True,
        max_matches: int = 100
    ) -> List[PatternMatch]:
        """
        Find all pattern matches in text.

        Args:
            text: Text to search
            extract_sentences: Whether to extract full sentences
            max_matches: Maximum matches to return (prevent runaway)

        Returns:
            List of PatternMatch objects
        """
        matches = []
        normalized_text = normalize_text(text)

        for pattern_name, category, pattern, noun_only in self.compiled_patterns:
            for match in pattern.finditer(normalized_text):
                if len(matches) >= max_matches:
                    break

                # Get the seed word from the match (second capture group)
                groups = match.groups()
                if len(groups) >= 2:
                    seed_word_normalized = groups[1].lower()
                else:
                    continue

                # Look up emotions - try main lookup first, then noun-specific
                emotions = self.normalized_to_emotions.get(seed_word_normalized, [])

                # For noun-only patterns, also check NOUN_EMOTION_MAP
                if not emotions and noun_only:
                    emotions = NOUN_EMOTION_MAP.get(seed_word_normalized, [])

                if not emotions:
                    continue

                # Find original form in text
                original_words = self.normalized_to_original.get(seed_word_normalized, [seed_word_normalized])
                original_seed = find_original_word(text, seed_word_normalized, match.start())

                # Get matched text segment from original text
                # Note: positions might differ due to diacritics, approximate
                matched_text = match.group(0)

                # Extract sentence if requested
                if extract_sentences:
                    sentence = extract_sentence(text, match.start(), match.end())
                else:
                    sentence = matched_text

                pm = PatternMatch(
                    pattern_name=pattern_name,
                    pattern_category=category,
                    matched_text=sentence,
                    seed_word=original_seed,
                    seed_word_normalized=seed_word_normalized,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    emotions=emotions
                )
                matches.append(pm)

        return matches

    def match_text(self, text: str) -> Optional[PatternMatch]:
        """
        Find first match in text (for quick filtering).

        Returns None if no match found.
        """
        matches = self.find_matches(text, extract_sentences=True, max_matches=1)
        return matches[0] if matches else None

    def has_affective_pattern(self, text: str) -> bool:
        """Check if text contains any affective state pattern."""
        return self.match_text(text) is not None


def test_patterns():
    """Test pattern matching with example sentences."""
    # Sample emotion seed for testing (includes plural forms)
    test_seed = {
        "fericit": ["joy"],
        "fericită": ["joy"],
        "fericiți": ["joy"],  # masculine plural
        "fericite": ["joy"],  # feminine plural
        "trist": ["sadness"],
        "tristă": ["sadness"],
        "triști": ["sadness"],  # masculine plural
        "triste": ["sadness"],  # feminine plural
        "frică": ["fear"],
        "teamă": ["fear"],
        "furios": ["anger"],
        "furioasă": ["anger"],
        "furioși": ["anger"],  # masculine plural
        "surprins": ["surprise"],
        "bine": ["joy"],
        "rău": ["sadness", "anger"],
        "calm": ["trust"],
        "calmă": ["trust"],
        "relaxat": ["trust"],
    }

    matcher = PatternMatcher(test_seed)

    test_cases = [
        # Primary patterns - should match
        ("Mă simt fericit astăzi.", True, "ma_simt_present"),
        ("ma simt trist si obosit", True, "ma_simt_present"),
        ("M-am simțit foarte bine la petrecere.", True, "mam_simtit_perfect"),
        ("Mă simțeam relaxat în vacanță.", True, "ma_simteam_imperfect"),
        ("Ne simțim triști după ce am auzit vestea.", True, "ne_simtim_present"),
        ("Ne-am simțit fericiți când am câștigat.", True, "neam_simtit_perfect"),
        # Secondary patterns - should match
        ("Sunt furios pe situația actuală.", True, "sunt_adj_present"),
        ("Sunt foarte calm acum.", True, "sunt_adj_present"),
        ("Eram trist când am plecat.", True, "eram_adj_imperfect"),
        ("Am fost surprins de rezultat.", True, "am_fost_adj_perfect"),
        ("Suntem fericiți împreună.", True, "suntem_adj_present"),
        ("Îmi este frică de viitor.", True, "imi_este_present"),
        ("Îmi era teamă că va ploua.", True, "imi_era_imperfect"),
        ("Mi-e frică de câini.", True, "mie_short"),
        ("Mi-e teamă să încerc.", True, "mie_short"),
        ("Am frică de înălțimi.", True, "am_noun_present"),
        ("Aveam teamă de eșec.", True, "aveam_noun_imperfect"),
        # Should NOT match - third person
        ("El se simte fericit.", False, None),
        ("Ea este tristă.", False, None),
        ("Ei sunt fericiți.", False, None),
        # Should NOT match - no seed word
        ("Mă simt bizar.", False, None),
        ("Sunt confuzat.", False, None),
    ]

    print("\nTest Results:")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_text, should_match, expected_pattern in test_cases:
        matches = matcher.find_matches(test_text)
        has_match = len(matches) > 0

        # Check if result matches expectation
        if has_match == should_match:
            if has_match and expected_pattern:
                # Also check pattern name
                if matches[0].pattern_name == expected_pattern:
                    status = "PASS"
                    passed += 1
                else:
                    status = f"FAIL (expected {expected_pattern}, got {matches[0].pattern_name})"
                    failed += 1
            else:
                status = "PASS"
                passed += 1
        else:
            status = f"FAIL (expected {'match' if should_match else 'no match'})"
            failed += 1

        # Print result
        print(f"[{status}] {test_text}")
        if matches:
            m = matches[0]
            print(f"        Pattern: {m.pattern_name}, Seed: {m.seed_word} → {m.emotions}")
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")


if __name__ == "__main__":
    test_patterns()
