"""Claim extraction, citation parsing, and entity/constraint analysis."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from app.citations.models import Claim

# Common English abbreviations and technical prefixes that should not cause sentence splits
_PROTECTED_PATTERNS = [
    r"\be\.g\.",
    r"\bi\.e\.",
    r"\betc\.",
    r"\bDr\.",
    r"\bMr\.",
    r"\bMrs\.",
    r"\bMs\.",
    r"\bDept\.",
    r"\bNo\.",
    r"\bFig\.",
    r"\bVol\.",
    r"\bRef\.",
    r"\bal\.",
    r"\bvs\.",
    r"\bapprox\.",
    r"\bInc\.",
    r"\bCorp\.",
    r"\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.",
    r"\bNX-[A-Z]{3}-\d+",
    r"\bDOC\d+_CHUNK_\d+",
]

# Standard sentinel phrase for refusal
INSUFFICIENT_INFO_SENTINEL = (
    "I do not have sufficient information in the provided context to answer this question."
)


def split_sentences(text: str) -> List[str]:
    """Split natural language answer text into discrete sentences.

    Protects abbreviations, numbers, codes, and bullet points from premature splits.

    Parameters
    ----------
    text : str
        Input answer text.

    Returns
    -------
    List[str]
        List of non-empty segmented sentences.
    """
    if not text or not text.strip():
        return []

    # Clean leading/trailing whitespace
    raw = text.strip()

    # If the text is exactly or contains the standard refusal, treat as single sentence
    if INSUFFICIENT_INFO_SENTINEL.lower() in raw.lower():
        return [raw]

    # Temporarily replace protected abbreviation periods with a sentinel placeholder
    protected_map: Dict[str, str] = {}
    temp_text = raw

    for idx, pat in enumerate(_PROTECTED_PATTERNS):
        def _replace_protected(m: re.Match) -> str:
            token = m.group(0)
            placeholder = f"__PROT_{idx}_{len(protected_map)}__"
            protected_map[placeholder] = token
            return placeholder

        temp_text = re.sub(pat, _replace_protected, temp_text, flags=re.IGNORECASE)

    # Also protect decimal numbers like 3.5, $50.00
    def _replace_num(m: re.Match) -> str:
        token = m.group(0)
        placeholder = f"__NUM_{len(protected_map)}__"
        protected_map[placeholder] = token
        return placeholder

    temp_text = re.sub(r"\b\d+\.\d+\b", _replace_num, temp_text)

    # Protect punctuation directly followed by citation bracket(s) (e.g. "Friday. [Source 1]" or "Friday. [Source 1][Source 2]")
    punct_map: Dict[str, str] = {}

    def _protect_punct(m: re.Match) -> str:
        punc = m.group(1)
        brackets = m.group(2)
        ph = f"__PUNCT_{len(punct_map)}__"
        punct_map[ph] = punc
        return f"{ph}{brackets}"

    temp_text = re.sub(r"([.!?])(\s*(?:\[[^\]]+\]\s*)+)", _protect_punct, temp_text)

    # Split boundaries after citations that follow punctuation
    temp_text = re.sub(r"(__PUNCT_\d+__\s*(?:\[[^\]]+\]\s*)+)\s+", r"\1__SENT_SPLIT__", temp_text)

    # Standard sentence boundaries
    temp_text = re.sub(r"([.!?])\s+", r"\1__SENT_SPLIT__", temp_text)

    # Split on explicit boundaries or newlines
    sentence_splits = re.split(r"__SENT_SPLIT__|\n+", temp_text)

    sentences: List[str] = []
    for s in sentence_splits:
        s_clean = s.strip()
        # Remove list markers like "- ", "* ", "1. "
        s_clean = re.sub(r"^(?:[-*•]|\d+\.)\s+", "", s_clean).strip()
        if not s_clean:
            continue

        # Restore punct placeholders
        for ph, orig_punc in punct_map.items():
            s_clean = s_clean.replace(ph, orig_punc)

        # Restore protected tokens
        for placeholder, original in protected_map.items():
            s_clean = s_clean.replace(placeholder, original)

        if s_clean:
            sentences.append(s_clean)

    return sentences if sentences else [raw]


def parse_citations_from_text(text: str) -> Tuple[str, List[int], List[str]]:
    """Extract inline citation tags and produce clean claim text.

    Supported patterns:
    - [Source 1], [Source 1, 2], [Source 1, Source 2], [Sources 1, 2]
    - [1], [1, 2], [1, 3]
    - [DOC001_CHUNK_01]
    - Multiple consecutive brackets: [Source 1][Source 2]

    Parameters
    ----------
    text : str
        Raw sentence text with potential citation markers.

    Returns
    -------
    Tuple[str, List[int], List[str]]
        (clean_text, citation_indices, cited_chunk_ids)
    """
    citation_indices: Set[int] = set()
    cited_chunk_ids: Set[str] = set()

    clean = text

    # 1. Match [Source 1], [Source 1, 2], [Source 1, Source 2], [Sources 1, 2], [source: 1]
    source_matches = re.finditer(
        r"\[(?:Sources?|Ref)?(?:\s*:\s*)?\s*(\d+(?:\s*,\s*(?:Sources?|Ref)?(?:\s*:\s*)?\s*\d+)*)\]",
        clean,
        flags=re.IGNORECASE,
    )
    for match in source_matches:
        content = match.group(1)
        nums = re.findall(r"\d+", content)
        for num in nums:
            citation_indices.add(int(num))

    # 2. Match explicit chunk IDs like [DOC001_CHUNK_01]
    chunk_matches = re.finditer(r"\[(DOC\d+_CHUNK_\d+)\]", clean)
    for match in chunk_matches:
        cited_chunk_ids.add(match.group(1))

    # Clean all citation brackets from text
    clean = re.sub(
        r"\[(?:Sources?|Ref)?(?:\s*:\s*)?\s*\d+(?:\s*,\s*(?:Sources?|Ref)?(?:\s*:\s*)?\s*\d+)*\]",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\[DOC\d+_CHUNK_\d+\]", "", clean)
    clean = re.sub(r"\s+\.", ".", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    # Remove trailing/leading punctuation artifacts left by bracket stripping
    clean = clean.strip(" ,;.")
    clean = re.sub(r"\s+\.", ".", clean)

    return clean, sorted(list(citation_indices)), sorted(list(cited_chunk_ids))


def extract_claims(answer: str) -> List[Claim]:
    """Segment an answer string into a list of structured Claim models.

    Parameters
    ----------
    answer : str
        Full generated answer text.

    Returns
    -------
    List[Claim]
        List of structured Claim models.
    """
    if not answer or not answer.strip():
        return []

    sentences = split_sentences(answer)
    claims: List[Claim] = []

    for idx, s in enumerate(sentences, start=1):
        clean_text, indices, chunk_ids = parse_citations_from_text(s)

        # Check if sentence is a non-factual refusal or greeting
        is_refusal = (
            "sufficient information" in clean_text.lower()
            or "do not have sufficient" in clean_text.lower()
        )
        is_greeting = clean_text.lower() in {"hello", "hi", "good morning", "thank you"}

        is_factual = not (is_refusal or is_greeting)

        if not clean_text:
            continue

        claims.append(
            Claim(
                claim_id=f"C{idx:03d}",
                text=clean_text,
                raw_text=s,
                citation_indices=indices,
                cited_chunk_ids=chunk_ids,
                is_factual=is_factual,
            )
        )

    return claims


# ---------------------------------------------------------------------------
# Entity, Numerical, and Constraint Extraction
# ---------------------------------------------------------------------------


def extract_codes(text: str) -> Set[str]:
    """Extract operational module codes (e.g. 'NX-FAC-100')."""
    return set(re.findall(r"\bNX-[A-Z]{3}-\d+\b", text))


def extract_dates(text: str) -> Set[str]:
    """Extract explicit date expressions (e.g. 'Dec 12', 'January 15, 2024', '15th of March')."""
    patterns = [
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s+\d{4})?\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+of\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,\s+\d{4})?\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]
    dates: Set[str] = set()
    for pat in patterns:
        for match in re.finditer(pat, text, flags=re.IGNORECASE):
            dates.add(match.group(0).lower())
    return dates


def extract_weekdays(text: str) -> Set[str]:
    """Extract days of the week (e.g. 'Monday', 'Friday')."""
    pattern = r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
    return {m.group(0).lower() for m in re.finditer(pattern, text, flags=re.IGNORECASE)}


def extract_times(text: str) -> Set[str]:
    """Extract clock times (e.g. '9 AM', '17:00', '08:00 to 18:00', '5:30 PM')."""
    pattern = r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm|hrs|hours)\b|\b\d{2}:\d{2}\b"
    return {m.group(0).lower().replace(" ", "") for m in re.finditer(pattern, text, flags=re.IGNORECASE)}


def extract_numbers(text: str) -> Set[str]:
    """Extract numeric values, quantities, currency, and percentages (e.g. '$50', '$1,000', '24/7', '100%', '3.5')."""
    # Exclude citation markers
    clean = re.sub(r"\[.*?\]", "", text)
    # Match currency ($50, $1,000), percentages (100%), ratios (24/7), decimals (3.5), integers (42)
    patterns = [
        r"\$?\b\d{1,3}(?:,\d{3})+(?:\.\d+)?%?(?:/\d+)?\b",
        r"\$?\b\d+(?:\.\d+)?%?(?:/\d+)?\b",
    ]
    matches: Set[str] = set()
    for pat in patterns:
        for m in re.findall(pat, clean):
            matches.add(m.lower())
    return matches


def extract_entities(text: str) -> Dict[str, Set[str]]:
    """Extract all critical entities and constraints from text.

    Returns
    -------
    Dict[str, Set[str]]
        Dictionary categorized by 'codes', 'dates', 'weekdays', 'times', 'numbers'.
    """
    return {
        "codes": extract_codes(text),
        "dates": extract_dates(text),
        "weekdays": extract_weekdays(text),
        "times": extract_times(text),
        "numbers": extract_numbers(text),
    }


def extract_polarity_cues(text: str) -> Dict[str, Any]:
    """Extract negation, restriction, and polarity indicators from text."""
    lowered = text.lower()
    negations = set(
        re.findall(
            r"\b(not|never|cannot|no|without|prohibited|forbidden|disallowed|restricted|closed|non-exempt|neither|nor)\b",
            lowered,
        )
    )
    affirmatives = set(
        re.findall(
            r"\b(allowed|permitted|unrestricted|open|available|enabled|mandatory|required|exempt|authorized)\b",
            lowered,
        )
    )
    return {
        "has_negation": bool(negations),
        "negation_terms": negations,
        "affirmative_terms": affirmatives,
    }
