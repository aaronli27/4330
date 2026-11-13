from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: str
    title: str
    sentences: Tuple[str, ...]
    source_record_id: str

    @property
    def text(self) -> str:
        return " ".join(self.sentences).strip()

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["sentences"] = list(self.sentences)
        value["text"] = self.text
        return value


@dataclass(frozen=True)
class HotpotRecord:
    record_id: str
    question: str
    answer: str
    level: str
    question_type: str
    context: Tuple[Paragraph, ...]
    supporting_facts: Tuple[Tuple[str, int], ...]

    @property
    def support_titles(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(title for title, _ in self.supporting_facts))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self.record_id,
            "question": self.question,
            "answer": self.answer,
            "level": self.level,
            "type": self.question_type,
            "context": [
                [paragraph.title, list(paragraph.sentences)]
                for paragraph in self.context
            ],
            "supporting_facts": [list(item) for item in self.supporting_facts],
        }


@dataclass(frozen=True)
class Variant:
    record_id: str
    condition: str
    question: str
    answer: str
    context: Tuple[Paragraph, ...]
    supporting_facts: Tuple[Tuple[str, int], ...]
    perturbation: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "condition": self.condition,
            "question": self.question,
            "answer": self.answer,
            "context": [paragraph.to_dict() for paragraph in self.context],
            "supporting_facts": [list(item) for item in self.supporting_facts],
            "perturbation": self.perturbation,
        }


def parse_variant(raw: Dict[str, Any]) -> Variant:
    context = tuple(
        Paragraph(
            paragraph_id=str(item["paragraph_id"]),
            title=str(item["title"]),
            sentences=tuple(str(sentence) for sentence in item["sentences"]),
            source_record_id=str(item.get("source_record_id", raw["record_id"])),
        )
        for item in raw["context"]
    )
    return Variant(
        record_id=str(raw["record_id"]),
        condition=str(raw["condition"]),
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        context=context,
        supporting_facts=tuple(
            (str(title), int(sentence_index))
            for title, sentence_index in raw["supporting_facts"]
        ),
        perturbation=dict(raw.get("perturbation", {})),
    )


def parse_record(raw: Dict[str, Any]) -> HotpotRecord:
    record_id = str(raw["_id"])
    context_raw = raw["context"]
    context: List[Paragraph] = []
    for index, item in enumerate(context_raw):
        title, sentences = item
        context.append(
            Paragraph(
                paragraph_id=f"{record_id}:p{index}",
                title=str(title),
                sentences=tuple(str(sentence) for sentence in sentences),
                source_record_id=record_id,
            )
        )

    supporting_facts = tuple(
        (str(title), int(sentence_index))
        for title, sentence_index in raw["supporting_facts"]
    )
    return HotpotRecord(
        record_id=record_id,
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        level=str(raw.get("level", "")),
        question_type=str(raw.get("type", "")),
        context=tuple(context),
        supporting_facts=supporting_facts,
    )


def validate_record(record: HotpotRecord) -> List[str]:
    errors: List[str] = []
    if len(record.context) != 10:
        errors.append("context_length_not_10")
    titles = [paragraph.title for paragraph in record.context]
    if len(set(titles)) != len(titles):
        errors.append("duplicate_context_title")
    title_map = {paragraph.title: paragraph for paragraph in record.context}
    for title, sentence_index in record.supporting_facts:
        paragraph = title_map.get(title)
        if paragraph is None:
            errors.append("support_title_missing")
            continue
        if sentence_index < 0 or sentence_index >= len(paragraph.sentences):
            errors.append("support_sentence_index_invalid")
    if not record.question.strip():
        errors.append("empty_question")
    if not record.answer.strip():
        errors.append("empty_answer")
    return sorted(set(errors))


def variant_support_titles_present(variant: Variant) -> bool:
    titles = {paragraph.title for paragraph in variant.context}
    return all(title in titles for title, _ in variant.supporting_facts)

