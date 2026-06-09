#!/usr/bin/env python3
"""
Chat Feature Pipeline for injection detection.
Extracts TF-IDF, structural, keyword, and pattern features from chat messages.
Follows sklearn convention (fit/transform/fit_transform), mirrors LogFeaturePipeline.
"""

import re
import math
import numpy as np
from collections import Counter
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, issparse


class ChatFeaturePipeline:
    """Extract features from chat messages for injection detection."""

    def __init__(self, max_tfidf_word_features=500, max_tfidf_char_features=200):
        self.max_tfidf_word_features = max_tfidf_word_features
        self.max_tfidf_char_features = max_tfidf_char_features

        # TF-IDF vectorizers
        self.word_tfidf = TfidfVectorizer(
            max_features=max_tfidf_word_features,
            ngram_range=(1, 2),
            strip_accents='unicode',
            lowercase=True,
            stop_words=None,  # Keep all words, attack keywords are important
        )
        self.char_tfidf = TfidfVectorizer(
            analyzer='char_wb',
            max_features=max_tfidf_char_features,
            ngram_range=(3, 5),
            strip_accents='unicode',
            lowercase=True,
        )

        self._is_fitted = False
        self.feature_names_ = None

    # Keyword dictionaries for density features
    KEYWORD_GROUPS = {
        "injection": [
            "ignore", "disregard", "forget", "bypass", "skip",
            "override", "dismiss", "previous instructions",
        ],
        "persona_change": [
            "you are now", "pretend you are", "act as", "roleplay",
            "simulate being", "become a", "you are dan",
        ],
        "hypothetical": [
            "hypothetically", "in theory", "imagine if", "fictional",
            "in a story", "what if", "theoretically", "suppose",
        ],
        "dan_jailbreak": [
            "dan mode", "do anything now", "jailbreak", "unrestricted",
            "no restrictions", "free from constraints", "god mode",
            "sudo mode", "developer mode",
        ],
        "credential": [
            "password", "credential", "api key", "secret key",
            "access token", "private key", "auth token",
        ],
        "system_prompt": [
            "system prompt", "initial instructions", "core directives",
            "your instructions", "your prompt", "your rules",
        ],
        "encoding": [
            "base64", "decode", "encode", "hex", "rot13",
            "unicode", "obfuscate",
        ],
        "logic_trap": [
            "you must answer", "you're obligated", "proves you're biased",
            "if you can't answer", "if you refuse", "you have to respond",
            "violates your purpose", "prove you're not",
        ],
        "urgency": [
            "urgent", "immediately", "emergency", "critical",
            "right now", "hurry", "asap", "time sensitive",
        ],
        "context_manipulation": [
            "new conversation", "different assistant", "change topic",
            "debug mode", "maintenance mode", "admin mode",
        ],
        "code_framing": [
            "write code", "write python", "write a script",
            "code that would", "just the code", "code example",
        ],
        "social_engineering": [
            "i'm from it", "security audit", "compliance",
            "authorized to", "my supervisor", "the ceo",
            "penetration test", "red team",
        ],
    }

    # Pattern regex for structural detection
    PATTERN_CHECKS = {
        "has_instruction_override": r"(?:ignore|disregard|forget|bypass|override)\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules|restrictions|directives|safety)",
        "has_role_reassignment": r"(?:you are now|pretend you|act as|roleplay as|simulate being|from now on you)",
        "has_base64": r"(?:[A-Za-z0-9+/]{20,}={0,2}|base64|decode\s+this)",
        "has_markup": r"<[a-zA-Z][^>]*>|</[a-zA-Z]+>",
        "has_json": r'\{[^}]*"[^"]*"\s*:',
        "has_url": r"https?://[^\s]+",
    }

    def fit(self, texts: List[str]) -> 'ChatFeaturePipeline':
        """Fit the pipeline on training texts."""
        self.word_tfidf.fit(texts)
        self.char_tfidf.fit(texts)

        # Build feature name list
        self.feature_names_ = (
            [f"tfidf_word_{f}" for f in self.word_tfidf.get_feature_names_out()] +
            [f"tfidf_char_{f}" for f in self.char_tfidf.get_feature_names_out()] +
            self._structural_feature_names() +
            [f"kw_{group}" for group in self.KEYWORD_GROUPS] +
            list(self.PATTERN_CHECKS.keys()) +
            ["question_count", "exclamation_count"]
        )

        self._is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts into feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fit before transform")

        # TF-IDF features
        word_features = self.word_tfidf.transform(texts)
        char_features = self.char_tfidf.transform(texts)

        # Structural + keyword + pattern features
        extra_features = np.array([self._extract_extra_features(t) for t in texts])

        # Combine sparse TF-IDF with dense features
        if issparse(word_features):
            word_dense = word_features.toarray()
        else:
            word_dense = word_features
        if issparse(char_features):
            char_dense = char_features.toarray()
        else:
            char_dense = char_features

        combined = np.hstack([word_dense, char_dense, extra_features])
        return combined

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> List[str]:
        """Return feature names after fitting."""
        return self.feature_names_

    def _structural_feature_names(self) -> List[str]:
        return [
            "msg_length", "word_count", "avg_word_length",
            "special_char_ratio", "uppercase_ratio", "digit_ratio",
            "entropy", "unique_word_ratio", "sentence_count",
            "max_word_length", "whitespace_ratio",
            "punctuation_ratio", "newline_count",
        ]

    def _extract_extra_features(self, text: str) -> np.ndarray:
        """Extract structural, keyword, and pattern features from a single text."""
        features = []

        # --- Structural features (13) ---
        msg_length = len(text)
        words = text.split()
        word_count = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_char_ratio = special_chars / max(msg_length, 1)
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / max(msg_length, 1)
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / max(msg_length, 1)
        entropy = self._shannon_entropy(text)
        unique_words = len(set(w.lower() for w in words))
        unique_word_ratio = unique_words / max(word_count, 1)
        sentence_count = max(len(re.split(r'[.!?]+', text)), 1)
        max_word_length = max((len(w) for w in words), default=0)
        whitespace_ratio = sum(1 for c in text if c.isspace()) / max(msg_length, 1)
        punctuation_count = sum(1 for c in text if c in '.,;:!?-()[]{}"\'/\\')
        punctuation_ratio = punctuation_count / max(msg_length, 1)
        newline_count = text.count('\n')

        features.extend([
            msg_length, word_count, avg_word_length,
            special_char_ratio, uppercase_ratio, digit_ratio,
            entropy, unique_word_ratio, sentence_count,
            max_word_length, whitespace_ratio,
            punctuation_ratio, newline_count,
        ])

        # --- Keyword density features (12) ---
        text_lower = text.lower()
        for group_name, keywords in self.KEYWORD_GROUPS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            density = count / max(word_count, 1)
            features.append(density)

        # --- Pattern features (6 regex + 2 counts = 8) ---
        for pattern_name, pattern in self.PATTERN_CHECKS.items():
            features.append(1.0 if re.search(pattern, text, re.IGNORECASE) else 0.0)

        question_count = text.count('?')
        exclamation_count = text.count('!')
        features.extend([question_count, exclamation_count])

        return np.array(features, dtype=np.float64)

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        counter = Counter(text)
        length = len(text)
        entropy = 0.0
        for count in counter.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
