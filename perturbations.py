from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .bm25_retriever import BM25, tokenize
from .schemas import HotpotRecord, Paragraph, Variant


def _paragraph_similarity(query: str, paragraph: Paragraph) -> float:
    query_terms = set(tokenize(query))
    paragraph_terms = set(tokenize(paragraph.text))
    if not query_terms or not paragraph_terms:
        return 0.0
    return len(query_terms & paragraph_terms) / len(query_terms | paragraph_terms)


def build_distractor_pool(
    records: Sequence[HotpotRecord],
    max_paragraphs: int = 5000,
) -> List[Paragraph]:
    pool: List[Paragraph] = []
    for record in records:
        support_titles = set(record.support_titles)
        for paragraph in record.context:
            if paragraph.title not in support_titles:
                pool.append(paragraph)
                if len(pool) >= max_paragraphs:
                    return pool
    return pool


def _choose_replacement(
    record: HotpotRecord,
    removed_title: str,
    pool: Sequence[Paragraph],
    rng: random.Random,
) -> Paragraph:
    context_titles = {paragraph.title for paragraph in record.context}
    candidates = [
        paragraph
        for paragraph in pool
        if paragraph.source_record_id != record.record_id
        and paragraph.title not in context_titles
    ]
    if not candidates:
        raise ValueError(f"No replacement paragraph available for {record.record_id}")
    removed = next(
        paragraph for paragraph in record.context if paragraph.title == removed_title
    )
    target_length = len(tokenize(removed.text))
    candidates.sort(
        key=lambda paragraph: (
            abs(len(tokenize(paragraph.text)) - target_length),
            paragraph.title,
        )
    )
    best_distance = abs(len(tokenize(candidates[0].text)) - target_length)
    tied = [
        paragraph
        for paragraph in candidates
        if abs(len(tokenize(paragraph.text)) - target_length) == best_distance
    ]
    return rng.choice(tied)


def _choose_hard_distractor(
    record: HotpotRecord,
    pool: Sequence[Paragraph],
    rng: random.Random,
) -> Paragraph:
    excluded_titles = set(record.support_titles) | {
        paragraph.title for paragraph in record.context
    }
    candidates = [
        paragraph
        for paragraph in pool
        if paragraph.source_record_id != record.record_id
        and paragraph.title not in excluded_titles
    ]
    if not candidates:
        raise ValueError("Hard distractor pool is empty")
    scored = [
        (_paragraph_similarity(record.question, paragraph), paragraph)
        for paragraph in candidates
    ]
    scored.sort(key=lambda item: (-item[0], item[1].source_record_id, item[1].title))
    top = [paragraph for score, paragraph in scored[: min(25, len(scored))]]
    return rng.choice(top)


def _replace_title(
    context: Sequence[Paragraph],
    old_title: str,
    replacement: Paragraph,
    record_id: str,
) -> Tuple[Paragraph, ...]:
    output: List[Paragraph] = []
    for index, paragraph in enumerate(context):
        if paragraph.title == old_title:
            output.append(
                Paragraph(
                    paragraph_id=f"{record_id}:p{index}",
                    title=replacement.title,
                    sentences=replacement.sentences,
                    source_record_id=replacement.source_record_id,
                )
            )
        else:
            output.append(
                Paragraph(
                    paragraph_id=f"{record_id}:p{index}",
                    title=paragraph.title,
                    sentences=paragraph.sentences,
                    source_record_id=paragraph.source_record_id,
                )
            )
    return tuple(output)


def make_variant(
    record: HotpotRecord,
    condition: str,
    hard_pool: Sequence[Paragraph],
    rng: random.Random,
) -> Variant:
    if condition not in {"C0", "C1", "C2", "C3"}:
        raise ValueError(f"Unknown condition: {condition}")
    support_title = record.support_titles[0]
    ordinary_replacement = _choose_replacement(record, support_title, hard_pool, rng)
    context = tuple(record.context)
    perturbation: Dict[str, Any] = {"condition": condition}

    if condition in {"C1", "C3"}:
        context = _replace_title(
            context=context,
            old_title=support_title,
            replacement=ordinary_replacement,
            record_id=record.record_id,
        )
        perturbation["missing_support_title"] = support_title
        perturbation["replacement_title"] = ordinary_replacement.title

    if condition in {"C2", "C3"}:
        current_titles = {paragraph.title for paragraph in context}
        hard = _choose_hard_distractor(record, hard_pool, rng)
        replace_candidates = [
            paragraph for paragraph in context if paragraph.title not in record.support_titles
        ]
        replace_candidates = [
            paragraph for paragraph in replace_candidates if paragraph.title != hard.title
        ]
        if not replace_candidates:
            raise ValueError(f"No ordinary paragraph available for {record.record_id}")
        replace_target = sorted(
            replace_candidates,
            key=lambda paragraph: (
                abs(len(tokenize(paragraph.text)) - len(tokenize(hard.text))),
                paragraph.title,
            ),
        )[0]
        context = _replace_title(
            context=context,
            old_title=replace_target.title,
            replacement=hard,
            record_id=record.record_id,
        )
        perturbation["hard_distractor_source_id"] = hard.source_record_id
        perturbation["hard_distractor_title"] = hard.title
        perturbation["replaced_distractor_title"] = replace_target.title

    return Variant(
        record_id=record.record_id,
        condition=condition,
        question=record.question,
        answer=record.answer,
        context=context,
        supporting_facts=record.supporting_facts,
        perturbation=perturbation,
    )


def build_variants(
    records: Sequence[HotpotRecord],
    hard_pool: Sequence[Paragraph],
    conditions: Sequence[str] = ("C0", "C1", "C2", "C3"),
    seed: int = 20261019,
) -> List[Variant]:
    variants: List[Variant] = []
    for index, record in enumerate(records):
        for condition_index, condition in enumerate(conditions):
            rng = random.Random(seed + index * 101 + condition_index)
            variants.append(make_variant(record, condition, hard_pool, rng))
    return variants

