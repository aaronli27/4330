from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
REPORT_DOCX = ROOT / "多跳RAG最终实验报告.docx"


def read_summary(stage: str) -> Dict[str, Any]:
    path = RESULTS / f"qwen_{stage}_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def set_run_font(run, size=None, bold=None, color="000000", name="Aptos"):
    run.font.name = name
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{key}"), name)


def add_run(paragraph, text, size=None, bold=None, color="000000", name="Aptos"):
    run = paragraph.add_run(text)
    set_run_font(run, size=size, bold=bold, color=color, name=name)
    return run


def set_style_font(style, size, bold=False, name="Aptos"):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{key}"), name)


def remove_borders(element):
    ppr = element.find(qn("w:pPr"))
    if ppr is not None:
        borders = ppr.find(qn("w:pBdr"))
        if borders is not None:
            ppr.remove(borders)


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=100, bottom=100, end=100):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_borders(table, color="D9D9D9"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        item = borders.find(qn(f"w:{edge}"))
        if item is None:
            item = OxmlElement(f"w:{edge}")
            borders.append(item)
        item.set(qn("w:val"), "single")
        item.set(qn("w:sz"), "6")
        item.set(qn("w:space"), "0")
        item.set(qn("w:color"), color)


def add_table(doc, headers, rows, widths, font_size=8.6):
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    for idx, header in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.width = widths[idx]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        shade_cell(cell, "315B74")
        set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        add_run(p, header, size=font_size, bold=True, color="FFFFFF")
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:tblHeader"))
    for ridx, row_values in enumerate(rows):
        row = table.add_row()
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for cidx, value in enumerate(row_values):
            cell = row.cells[cidx]
            cell.width = widths[cidx]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if ridx % 2:
                shade_cell(cell, "F1F6F9")
            set_cell_margins(cell)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            if cidx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_run(p, str(value), size=font_size, bold=(cidx == 0))
    set_borders(table)
    return table


def make_figures(standard: Dict[str, Any], unseen: Dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    methods = ["G0", "G1", "G2", "G3"]
    labels = ["One-shot", "Fixed-depth", "Global gate", "Local gate"]
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#A64D79"]

    plt.figure(figsize=(7.1, 4.0))
    x = list(range(len(methods)))
    for condition, name in [("C1", "Missing hop"), ("C3", "Unseen combined")]:
        values = [standard[m][condition]["unsafe_answer"] for m in methods]
        plt.plot(x, values, marker="o", linewidth=2, label=name)
    plt.xticks(x, labels, rotation=15)
    plt.ylim(0, 0.5)
    plt.ylabel("Unsafe-answer rate")
    plt.title("Standard test reliability")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "standard_reliability.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.1, 4.0))
    for condition, name in [("C1", "Missing hop"), ("C3", "Unseen combined")]:
        values = [unseen[m][condition]["abstention_correct"] for m in methods]
        plt.plot(x, values, marker="o", linewidth=2, label=name)
    plt.xticks(x, labels, rotation=15)
    plt.ylim(0, 1.0)
    plt.ylabel("Correct abstention rate")
    plt.title("Unseen test abstention performance")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "unseen_abstention.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.1, 4.0))
    for condition, name in [("C0", "Complete"), ("C3", "Combined perturbation")]:
        values = [standard[m][condition]["total_tokens"] for m in methods]
        plt.plot(x, values, marker="o", linewidth=2, label=name)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Average total tokens")
    plt.title("Standard test cost")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "standard_cost.png", dpi=220)
    plt.close()


def add_body_paragraph(doc, text, after=5):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.12
    add_run(p, text)
    return p


def add_label_paragraph(doc, label, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.12
    add_run(p, label, bold=True)
    add_run(p, text)
    return p


def build_report() -> None:
    standard = read_summary("standard_test")
    unseen = read_summary("unseen_test")
    calibration = json.loads(
        (RESULTS / "qwen_calibration_summary.json").read_text(encoding="utf-8")
    )
    make_figures(standard, unseen)

    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)

    set_style_font(doc.styles["Normal"], 10.1)
    doc.styles["Normal"].paragraph_format.space_after = Pt(5)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.12
    set_style_font(doc.styles["Title"], 22, bold=True)
    remove_borders(doc.styles["Title"]._element)
    for name, size, before, after in (
        ("Heading 1", 15, 13, 6),
        ("Heading 2", 12, 9, 4),
    ):
        style = doc.styles[name]
        set_style_font(style, size, bold=True)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    title = doc.add_paragraph(style="Title")
    remove_borders(title._p)
    add_run(title, "A Controlled Cost Risk Evaluation of Evidence Coverage Gating for Selective Multi hop RAG", size=22, bold=True)
    sub = doc.add_paragraph()
    sub.paragraph_format.space_after = Pt(10)
    add_run(sub, "Final experimental report", size=11, color="555555")
    meta = doc.add_paragraph()
    meta.paragraph_format.space_after = Pt(10)
    add_run(meta, "System: ", bold=True)
    add_run(meta, "Haystack-style pipeline, BM25, Qwen2.5-1.5B-Instruct (4-bit)")
    add_run(meta, "    Dataset: ", bold=True)
    add_run(meta, "HotpotQA distractor")

    doc.add_heading("Abstract", level=1)
    add_body_paragraph(
        doc,
        "This project evaluates whether evidence-coverage gating can improve the reliability and cost efficiency of selective multi-hop retrieval-augmented generation. We use the HotpotQA distractor setting as a controlled ten-paragraph candidate pool and compare four methods under the same model and token budget: one-shot BM25 retrieval, fixed-depth multi-hop retrieval, global evidence sufficiency gating, and subquestion-level evidence-coverage gating. We create complete-evidence, missing-hop, hard-distractor, and combined perturbation conditions. The final evaluation uses 100 calibration questions, 100 standard-test questions, and 100 unseen-test questions, with four variants per question. Results show that global gating is generally more reliable than local gating on missing-evidence conditions. On the unseen combined condition, global gating reaches 0.82 correct abstention and 0.18 unsafe-answer rate, while local gating reaches 0.77 and 0.23, respectively. Local coverage gating therefore does not demonstrate a consistent advantage in this implementation; the main contribution is a reproducible cost-risk audit and an explicit analysis of failure modes."
    )

    doc.add_heading("1 Introduction", level=1)
    add_body_paragraph(
        doc,
        "Retrieval-augmented generation is often evaluated as if every question should receive an answer. In multi-hop question answering, however, a single missing bridge fact can make a fluent answer unsupported. A reliable system should distinguish answerable evidence from incomplete evidence and abstain when the available context cannot justify a response. This project studies that decision problem under a fixed computational budget."
    )
    add_body_paragraph(
        doc,
        "The study is deliberately framed as a controlled evaluation rather than a claim of a new abstention principle. Recent work has already examined sufficient context, selective generation, evidence verification, and additional retrieval. Our question is narrower: under the same candidate pool and language-model token budget, does local subquestion coverage provide a better decision signal than a global sufficiency judgment, especially on evidence perturbations not used for calibration?"
    )

    doc.add_heading("2 Research Questions", level=1)
    add_label_paragraph(doc, "RQ1. ", "Does gating reduce unsupported answers when a necessary evidence paragraph is removed?")
    add_label_paragraph(doc, "RQ2. ", "Does subquestion-level coverage outperform global gating on an unseen combination of missing evidence and hard distractors?")
    add_label_paragraph(doc, "RQ3. ", "What is the cost of reliability in terms of model tokens, calls, and latency?")

    doc.add_heading("3 Method", level=1)
    doc.add_heading("3.1 Dataset and protocol", level=2)
    add_body_paragraph(
        doc,
        "The downloaded HotpotQA distractor development file contains 7,405 records. After requiring exactly ten candidate paragraphs and valid supporting-fact indices, 7,344 records remain. We split original question IDs into pilot (50), calibration (100), standard test (100), and unseen test (100). Each question is expanded into four conditions: C0 complete evidence; C1 removal of one supporting paragraph with a length-matched replacement; C2 replacement of an ordinary distractor with a high-similarity paragraph from the calibration donor pool; and C3 the combination of C1 and C2. The supporting-fact annotations are used only for offline construction and evaluation, never as model input."
    )
    doc.add_heading("3.2 Compared methods", level=2)
    add_table(
        doc,
        ["Method", "Description"],
        [
            ("G0", "One BM25 retrieval over the ten-paragraph candidate pool, followed by direct generation."),
            ("G1", "Question decomposition, fixed two-round retrieval, and direct generation."),
            ("G2", "Global sufficiency judgment; retrieve again when insufficient, otherwise answer or abstain."),
            ("G3", "Subquestion-level coverage judgment; retrieve for uncovered subquestions, otherwise answer or abstain."),
        ],
        [Inches(0.75), Inches(6.1)],
    )
    add_body_paragraph(
        doc,
        "All methods use the same Qwen2.5-1.5B-Instruct model in 4-bit mode, BM25 retrieval, deterministic decoding, and a maximum of 4,096 total input and output tokens per question. Every result records status, answer, evidence IDs, token counts, model calls, latency, and gate confidence."
    )
    doc.add_heading("3.3 Metrics", level=2)
    add_body_paragraph(
        doc,
        "Complete-evidence conditions are evaluated with answer exact match and token F1, while evidence quality is reported at paragraph level. Missing-evidence conditions use unsafe-answer rate and correct-abstention rate. We also report coverage, AURC, total tokens, and latency. Unsafe answer is intentionally separated from wrong answer because a model can answer correctly from parametric memory while lacking support in the supplied evidence."
    )

    doc.add_heading("4 Results", level=1)
    doc.add_heading("4.1 Standard test", level=2)
    standard_rows = []
    for method, label in zip(["G0", "G1", "G2", "G3"], ["One-shot", "Fixed-depth", "Global gate", "Local gate"]):
        standard_rows.append(
            (
                label,
                f"{standard[method]['C0']['answer_f1']:.3f}",
                f"{standard[method]['C1']['unsafe_answer']:.2f}",
                f"{standard[method]['C1']['abstention_correct']:.2f}",
                f"{standard[method]['C3']['unsafe_answer']:.2f}",
                f"{standard[method]['C3']['abstention_correct']:.2f}",
                f"{standard[method]['C3']['total_tokens']:.0f}",
            )
        )
    add_table(
        doc,
        ["Method", "C0 F1", "C1 Unsafe", "C1 Abstain", "C3 Unsafe", "C3 Abstain", "C3 Tokens"],
        standard_rows,
        [Inches(1.25), Inches(0.72), Inches(0.85), Inches(0.9), Inches(0.85), Inches(0.9), Inches(1.0)],
        font_size=8.2,
    )
    add_body_paragraph(
        doc,
        "On standard test, G0 has the highest complete-evidence F1 (0.253), but it is least selective. G2 reduces C1 unsafe answers to 0.34 compared with 0.37 for G0 and 0.31 for G1; G3 reaches 0.31. On C3, G2 reaches 0.33 unsafe answers and G3 reaches 0.26. The local gate therefore has a modest safety advantage over the global gate on this split, but it also consumes about 2,774 tokens per C3 item compared with 2,482 for G2."
    )
    doc.add_picture(str(FIGURES / "standard_reliability.png"), width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run(cap, "Figure 1. Unsafe-answer rate on the standard test.", size=8.5, color="555555")

    doc.add_heading("4.2 Unseen combined perturbation", level=2)
    unseen_rows = []
    for method, label in zip(["G0", "G1", "G2", "G3"], ["One-shot", "Fixed-depth", "Global gate", "Local gate"]):
        unseen_rows.append(
            (
                label,
                f"{unseen[method]['C0']['answer_f1']:.3f}",
                f"{unseen[method]['C3']['answer_f1']:.3f}",
                f"{unseen[method]['C3']['unsafe_answer']:.2f}",
                f"{unseen[method]['C3']['abstention_correct']:.2f}",
                f"{unseen[method]['C3']['coverage']:.2f}",
            )
        )
    add_table(
        doc,
        ["Method", "C0 F1", "C3 F1", "C3 Unsafe", "C3 Abstain", "C3 Coverage"],
        unseen_rows,
        [Inches(1.3), Inches(0.8), Inches(0.8), Inches(1.0), Inches(1.0), Inches(1.0)],
        font_size=8.5,
    )
    add_body_paragraph(
        doc,
        "The unseen test is the strongest test of generalization because C3 was not used for threshold calibration. G2 is the most reliable method: it achieves 0.18 unsafe answers and 0.82 correct abstention. G3 achieves 0.23 and 0.77. G0 and G1 are less reliable, with unsafe rates of 0.21 and 0.24. The local gate does not outperform the global gate on this unseen condition. Its additional decomposition and local judgment increase cost without producing a consistent safety gain."
    )
    doc.add_picture(str(FIGURES / "unseen_abstention.png"), width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run(cap, "Figure 2. Correct abstention on the unseen test.", size=8.5, color="555555")

    doc.add_heading("4.3 Calibration and cost", level=2)
    add_body_paragraph(
        doc,
        "Calibration used 100 questions and four variants per question. The generated confidence values were concentrated near the same values, so threshold selection did not materially separate the methods. This is an important negative result: the current confidence signal is not well calibrated enough to support a strong threshold-optimization claim. G0 uses about 1,000 tokens per item, G1 about 1,450, G2 about 2,400, and G3 about 2,800. The extra cost of G3 is therefore substantial."
    )
    doc.add_picture(str(FIGURES / "standard_cost.png"), width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run(cap, "Figure 3. Average total-token cost on the standard test.", size=8.5, color="555555")

    doc.add_heading("5 Error Analysis", level=1)
    add_body_paragraph(
        doc,
        "The logs show three recurring failure modes. First, the generator sometimes abstains even when the local gate marks all subquestions as covered; this is conservative but reduces complete-evidence coverage. Second, the gate can overestimate coverage when a high-similarity distractor mentions the same entities as the missing hop. Third, malformed or explanatory JSON occasionally appears, although the parser handled these cases without crashing and the observed parse-error rates remained low. These failures suggest that a future local gate should verify the content of each supporting sentence, not only whether a paragraph was retrieved."
    )

    doc.add_heading("6 Limitations", level=1)
    for text in [
        "The HotpotQA distractor protocol is a closed ten-paragraph candidate pool, not a full Wikipedia open-domain retrieval benchmark.",
        "The project uses paragraph-level evidence IDs; it does not claim official sentence-level supporting-fact F1.",
        "The Qwen2.5-1.5B model is small, and all results depend on its prompting and 4-bit inference behavior.",
        "C1 removal is constructed from supporting-fact titles and requires manual semantic validation; some remaining paragraphs may still contain enough information.",
        "Confidence calibration was weak, so the reported threshold experiment should be interpreted as an audit rather than a validated calibration method.",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Inches(0.24)
        p.paragraph_format.first_line_indent = Inches(-0.16)
        p.paragraph_format.space_after = Pt(3)
        add_run(p, text)

    doc.add_heading("7 Conclusion", level=1)
    add_body_paragraph(
        doc,
        "The controlled study demonstrates that evidence gating can reduce unsafe answers, but it does not support the hypothesis that subquestion-level coverage is consistently better than global sufficiency gating. G2 is the strongest practical method in this implementation: it provides the best unseen-test abstention behavior at lower cost than G3. The project therefore contributes a reproducible protocol, complete cost-risk measurements, and a clear negative result about the current local gate. A future improvement should focus on sentence-level entailment, better confidence calibration, and evidence verification that is robust to entity-matched distractors."
    )

    doc.add_heading("References", level=1)
    refs = [
        "Yang et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP, 2018.",
        "Sufficient Context. ICLR, 2025. Work on context sufficiency and selective generation.",
        "SURE-RAG. arXiv preprint, 2026. Work on multi-hop evidence sufficiency and selective answering.",
        "Qwen Team. Qwen2.5 Technical Report and Qwen2.5-1.5B-Instruct model card.",
        "deepset. Haystack open-source RAG framework.",
    ]
    for ref in refs:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.left_indent = Inches(0.26)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.paragraph_format.space_after = Pt(3)
        add_run(p, ref, size=9.2)

    doc.core_properties.title = "Final Experimental Report for Selective Multi-hop RAG"
    doc.core_properties.subject = "Controlled cost-risk evaluation of evidence coverage gating"
    doc.core_properties.author = ""
    doc.save(REPORT_DOCX)
    print(REPORT_DOCX)


if __name__ == "__main__":
    build_report()
