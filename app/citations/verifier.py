"""Multi-signal evidence alignment and citation claim verification engine."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from app.citations.extractor import (
    extract_entities,
    extract_polarity_cues,
)
from app.citations.models import (
    Claim,
    ClaimVerification,
    UnsupportedReason,
)
from app.config import (
    CITATION_ENTITY_MATCH_THRESHOLD,
    CITATION_MIN_OVERLAP_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Standard English stopwords for content token filtering
_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "based", "provided", "documentation", "document",
    "according", "chunk", "module", "title", "department", "content",
}


def _stem_token(word: str) -> str:
    """Deterministic suffix stripping for English inflections to support benign paraphrasing."""
    w = word.lower()
    for suffix in (
        "ations", "ation", "ments", "ment", "ingly", "able", "ible",
        "ings", "ing", "ies", "ied", "sses", "ses", "ed", "ly", "es", "s"
    ):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            if suffix == "ies":
                return w[:-3] + "y"
            return w[:-len(suffix)]
    return w


def tokenize_content_words(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric content words, removing stopwords."""
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-$%]+\b", text.lower())
    return [t for t in raw_tokens if t not in _STOPWORDS and len(t) > 1]


def compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute Longest Common Subsequence (LCS) length of two token sequences."""
    if not seq1 or not seq2:
        return 0

    m, n = len(seq1), len(seq2)
    # Use space-optimized DP table
    dp = [0] * (n + 1)

    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[n]


_CONTRADICTION_PAIRS = [
    ("unrestricted", "restricted"),
    ("restricted", "unrestricted"),
    ("permitted", "prohibited"),
    ("prohibited", "permitted"),
    ("permitted", "forbidden"),
    ("forbidden", "permitted"),
    ("allowed", "forbidden"),
    ("forbidden", "allowed"),
    ("allowed", "disallowed"),
    ("disallowed", "allowed"),
    ("allowed", "prohibited"),
    ("prohibited", "allowed"),
    ("open", "closed"),
    ("closed", "open"),
    ("mandatory", "optional"),
    ("optional", "mandatory"),
    ("required", "exempt"),
    ("exempt", "required"),
    ("required", "prohibited"),
    ("prohibited", "required"),
    ("authorized", "unauthorized"),
    ("unauthorized", "authorized"),
]


class ClaimVerifier:
    """Multi-signal claim verification engine combining entity, polarity, and lexical alignment."""

    def __init__(
        self,
        min_overlap_threshold: float = CITATION_MIN_OVERLAP_THRESHOLD,
        entity_match_threshold: float = CITATION_ENTITY_MATCH_THRESHOLD,
    ) -> None:
        self.min_overlap_threshold = min_overlap_threshold
        self.entity_match_threshold = entity_match_threshold

    def evaluate_support_against_chunk(
        self, claim: Claim, chunk_text: str
    ) -> Tuple[bool, float, UnsupportedReason, Dict[str, float]]:
        """Evaluate whether a single evidence chunk supports a given factual claim.

        Parameters
        ----------
        claim : Claim
            The factual assertion to verify.
        chunk_text : str
            Raw text content of the candidate evidence chunk.

        Returns
        -------
        Tuple[bool, float, UnsupportedReason, Dict[str, float]]
            (is_supported, support_score, failure_reason, diagnostic_signals)
        """
        if not chunk_text or not chunk_text.strip():
            return False, 0.0, UnsupportedReason.NO_SUPPORTING_SOURCE, {}

        claim_entities = extract_entities(claim.text)
        chunk_entities = extract_entities(chunk_text)
        claim_text_lower = claim.text.lower()
        chunk_text_lower = chunk_text.lower()

        # -------------------------------------------------------------------
        # 1. Entity & Numerical & Temporal Constraint Alignment
        # -------------------------------------------------------------------
        # Check codes
        for code in claim_entities["codes"]:
            if code not in chunk_entities["codes"] and code.lower() not in chunk_text_lower:
                return (
                    False,
                    0.0,
                    UnsupportedReason.ENTITY_MISMATCH,
                    {"entity_match": 0.0, "code_mismatch": 1.0},
                )

        # Check dates
        for date_val in claim_entities["dates"]:
            if date_val not in chunk_entities["dates"] and date_val not in chunk_text_lower:
                return (
                    False,
                    0.0,
                    UnsupportedReason.TEMPORAL_MISMATCH,
                    {"entity_match": 0.0, "date_mismatch": 1.0},
                )

        # Check weekdays (e.g. Wednesday vs Friday)
        for day in claim_entities["weekdays"]:
            if day not in chunk_entities["weekdays"] and day not in chunk_text_lower:
                return (
                    False,
                    0.0,
                    UnsupportedReason.TEMPORAL_MISMATCH,
                    {"entity_match": 0.0, "weekday_mismatch": 1.0},
                )

        # Check times
        for time_val in claim_entities["times"]:
            if (
                time_val not in chunk_entities["times"]
                and time_val not in chunk_text_lower.replace(" ", "")
                and time_val.replace(" ", "") not in chunk_text_lower.replace(" ", "")
            ):
                return (
                    False,
                    0.0,
                    UnsupportedReason.TEMPORAL_MISMATCH,
                    {"entity_match": 0.0, "time_mismatch": 1.0},
                )

        # Check numbers / quantities / currency
        for num_val in claim_entities["numbers"]:
            # Check if num_val, clean digits, or formatted version exists in chunk
            num_clean = re.sub(r"[^\d.]", "", num_val)
            if (
                num_val not in chunk_entities["numbers"]
                and num_clean not in chunk_text_lower
                and num_val not in chunk_text_lower
            ):
                return (
                    False,
                    0.0,
                    UnsupportedReason.NUMERICAL_MISMATCH,
                    {"entity_match": 0.0, "number_mismatch": 1.0},
                )

        # -------------------------------------------------------------------
        # 2. Polarity / Negation Consistency Check
        # -------------------------------------------------------------------
        has_antonym_contradiction = False
        for claim_term, chunk_term in _CONTRADICTION_PAIRS:
            # Word boundary check for polarity terms
            if re.search(r"\b" + re.escape(claim_term) + r"\b", claim_text_lower) and re.search(
                r"\b" + re.escape(chunk_term) + r"\b", chunk_text_lower
            ):
                has_antonym_contradiction = True
                break

        # Check explicit negation contradiction (both directions: insertion and removal)
        if not has_antonym_contradiction:
            neg_patterns = [
                r"\b(?:not|never|cannot|no|without|neither|nor)\s+(?:to\s+)?(?:be\s+)?(?:allowed|permitted|open|required|mandatory|restricted|authorized|needed|governed|available|exempt)\b",
                r"\b(?:does|do|is|are|was|were|will|shall|can|could|should|would)\s+not\s+(?:require|permit|allow|restrict|need|authorize|apply)\b",
                r"\bnot\s+require[ds]?\b",
                r"\bwithout\s+(?:prior\s+|supervisor\s+)?(?:approval|authorization|permit)\b",
                r"\bit\s+is\s+not\s+the\s+case\b",
            ]
            aff_patterns = [
                r"\b(?:allowed|permitted|open|required|mandatory|restricted|authorized|needed|governed|available)\b",
                r"\b(?:requires?|permits?|allows?|authorizes?)\b",
                r"\brequires?\s+(?:prior\s+|supervisor\s+)?(?:approval|authorization|permit)\b",
            ]

            claim_has_neg = any(re.search(p, claim_text_lower) for p in neg_patterns)
            chunk_has_neg = any(re.search(p, chunk_text_lower) for p in neg_patterns)
            claim_has_aff = any(re.search(p, claim_text_lower) for p in aff_patterns)
            chunk_has_aff = any(re.search(p, chunk_text_lower) for p in aff_patterns)

            # Negation insertion (claim negates what chunk affirms)
            if claim_has_neg and chunk_has_aff and not chunk_has_neg:
                has_antonym_contradiction = True
            # Negation removal (claim affirms what chunk negates)
            elif chunk_has_neg and claim_has_aff and not claim_has_neg:
                has_antonym_contradiction = True

        if has_antonym_contradiction:
            return (
                False,
                0.0,
                UnsupportedReason.CONTRADICTION,
                {"polarity_match": 0.0, "contradiction": 1.0},
            )

        # -------------------------------------------------------------------
        # 3. Content-Token Precision & LCS Directional Overlap with Stem Normalization
        # -------------------------------------------------------------------
        claim_tokens = tokenize_content_words(claim.text)
        chunk_tokens = tokenize_content_words(chunk_text)

        if not claim_tokens:
            # Vacuously grounded if claim contains no content words (e.g. pure filler)
            return True, 1.0, UnsupportedReason.NONE, {"token_overlap": 1.0, "entity_match": 1.0}

        claim_stems = [_stem_token(t) for t in claim_tokens]
        chunk_stems = [_stem_token(t) for t in chunk_tokens]
        chunk_token_set = set(chunk_tokens)
        chunk_stem_set = set(chunk_stems)

        matched_tokens = [
            t for t, stem in zip(claim_tokens, claim_stems)
            if t in chunk_token_set or stem in chunk_stem_set
        ]
        token_precision = len(matched_tokens) / len(claim_tokens)

        lcs_len = compute_lcs_length(claim_stems, chunk_stems)
        lcs_ratio = lcs_len / len(claim_stems)

        # Combined directional support score
        support_score = 0.6 * token_precision + 0.4 * lcs_ratio
        support_score = min(max(support_score, 0.0), 1.0)

        signals = {
            "token_precision": round(token_precision, 4),
            "lcs_ratio": round(lcs_ratio, 4),
            "support_score": round(support_score, 4),
            "entity_match": 1.0,
            "polarity_match": 1.0,
        }

        # -------------------------------------------------------------------
        # 4. Decision Threshold
        # -------------------------------------------------------------------
        if support_score >= self.min_overlap_threshold:
            return True, support_score, UnsupportedReason.NONE, signals
        else:
            return False, support_score, UnsupportedReason.NO_SUPPORTING_SOURCE, signals

    def verify_claim(
        self, claim: Claim, sources: List[Any]
    ) -> ClaimVerification:
        """Verify a single claim against the retrieved evidence sources.

        Parameters
        ----------
        claim : Claim
            The segmented claim to verify.
        sources : List[RAGSource]
            List of retrieved sources with text metadata.

        Returns
        -------
        ClaimVerification
            Detailed verification report for this claim.
        """
        # Non-factual claims (e.g. conversational pleasantries or refusals) are vacuously grounded
        if not claim.is_factual:
            return ClaimVerification(
                claim=claim,
                is_grounded=True,
                is_citation_correct=True,
                has_citation=bool(claim.citation_indices or claim.cited_chunk_ids),
                support_score=1.0,
                supporting_chunk_ids=[],
                cited_chunk_ids=[],
                unsupported_reason=UnsupportedReason.NONE,
                signals={"non_factual": 1.0},
            )

        # 1. Resolve cited chunk references and check index validity
        has_citation = bool(claim.citation_indices or claim.cited_chunk_ids)
        invalid_index_found = False
        cited_sources: List[Any] = []
        cited_chunk_ids: List[str] = []

        num_sources = len(sources)

        for idx in claim.citation_indices:
            if idx < 1 or idx > num_sources:
                invalid_index_found = True
            else:
                s = sources[idx - 1]
                cited_sources.append(s)
                cited_chunk_ids.append(getattr(s, "chunk_id", f"SOURCE_{idx}"))

        for cid in claim.cited_chunk_ids:
            matching = [s for s in sources if getattr(s, "chunk_id", None) == cid]
            if matching:
                cited_sources.extend(matching)
                cited_chunk_ids.append(cid)
            else:
                invalid_index_found = True

        if invalid_index_found:
            return ClaimVerification(
                claim=claim,
                is_grounded=False,
                is_citation_correct=False,
                has_citation=has_citation,
                support_score=0.0,
                supporting_chunk_ids=[],
                cited_chunk_ids=cited_chunk_ids,
                unsupported_reason=UnsupportedReason.INVALID_CITATION_INDEX,
                signals={"invalid_citation_index": 1.0},
            )

        # 2. Check support against cited chunks (Citation Correctness)
        citation_correct = False
        best_support_score = 0.0
        supporting_chunks: List[str] = []
        primary_signals: Dict[str, float] = {}
        primary_reason = UnsupportedReason.NONE

        if cited_sources:
            all_cited_supported = True
            for s in cited_sources:
                title = getattr(s, "title", "") or ""
                dept = getattr(s, "department", "") or ""
                text = getattr(s, "text", "") or ""
                chunk_text = f"{title} {dept} {text}".strip()
                chunk_id = getattr(s, "chunk_id", "UNKNOWN")
                supported, score, reason, signals = self.evaluate_support_against_chunk(
                    claim, chunk_text
                )
                if supported:
                    supporting_chunks.append(chunk_id)
                    if score > best_support_score:
                        best_support_score = score
                        primary_signals = signals
                else:
                    all_cited_supported = False
                    if primary_reason == UnsupportedReason.NONE:
                        primary_reason = reason
                        primary_signals = signals

            citation_correct = all_cited_supported and (len(supporting_chunks) > 0)

        # 3. Check support across all retrieved candidate sources (Claim Groundedness)
        is_grounded = bool(supporting_chunks)
        if not is_grounded:
            for s in sources:
                title = getattr(s, "title", "") or ""
                dept = getattr(s, "department", "") or ""
                text = getattr(s, "text", "") or ""
                chunk_text = f"{title} {dept} {text}".strip()
                chunk_id = getattr(s, "chunk_id", "UNKNOWN")
                supported, score, reason, signals = self.evaluate_support_against_chunk(
                    claim, chunk_text
                )
                if supported:
                    is_grounded = True
                    supporting_chunks.append(chunk_id)
                    if score > best_support_score:
                        best_support_score = score
                        primary_signals = signals
                else:
                    if primary_reason == UnsupportedReason.NONE and reason != UnsupportedReason.NONE:
                        primary_reason = reason
                        primary_signals = signals
                    elif reason in {
                        UnsupportedReason.TEMPORAL_MISMATCH,
                        UnsupportedReason.NUMERICAL_MISMATCH,
                        UnsupportedReason.ENTITY_MISMATCH,
                        UnsupportedReason.CONTRADICTION,
                    }:
                        primary_reason = reason
                        primary_signals = signals

        # 4. Determine final unsupported reason
        final_reason = UnsupportedReason.NONE
        if not is_grounded:
            final_reason = (
                primary_reason
                if primary_reason != UnsupportedReason.NONE
                else UnsupportedReason.NO_SUPPORTING_SOURCE
            )
        elif has_citation and not citation_correct:
            final_reason = UnsupportedReason.WRONG_CITATION
        elif is_grounded and not has_citation:
            final_reason = UnsupportedReason.MISSING_CITATION

        return ClaimVerification(
            claim=claim,
            is_grounded=is_grounded,
            is_citation_correct=citation_correct if has_citation else False,
            has_citation=has_citation,
            support_score=best_support_score,
            supporting_chunk_ids=supporting_chunks,
            cited_chunk_ids=cited_chunk_ids,
            unsupported_reason=final_reason,
            signals=primary_signals,
        )
