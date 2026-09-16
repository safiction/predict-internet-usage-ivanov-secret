"""Generate the anonymous seven-section technical report from current artifacts."""

import re
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import confusion_matrix

from src.config import PROJECT_ROOT, RESULTS_DIR, TARGET
from src.experiment import PRIMARY, file_digest


def markdown_table(frame, columns):
    table = frame[columns].copy()
    for column in table.select_dtypes(include="number"):
        if column in {"class", "support", "repeats"}:
            table[column] = table[column].map(lambda value: str(int(value)))
        elif column == "true_positives":
            table[column] = table[column].map(lambda value: f"{value:.1f}")
        else:
            table[column] = table[column].map(lambda value: f"{value:.4f}")
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)],
    ])


def make_plots(summary, oof, root=PROJECT_ROOT):
    figures = root / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot = summary.sort_values("mean_oof_qwk")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    means = plot.mean_oof_qwk.to_numpy()
    ax.errorbar(means, np.arange(len(plot)), xerr=np.vstack([means - plot.ci_low, plot.ci_high - means]),
                fmt="o", capsize=3, color="#315b85")
    ax.set_yticks(np.arange(len(plot)), plot.setup)
    ax.set_xlabel("Mean OOF QWK; conditional participant-bootstrap 95% interval")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "nested_qwk.png", dpi=150)
    plt.close(fig)
    confusion = confusion_matrix(oof[TARGET], oof[PRIMARY], labels=[0, 1, 2, 3])
    # Counts sum across repeats; the caption distinguishes them from unique IDs.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(confusion, cmap="Blues")
    for row in range(4):
        for column in range(4):
            ax.text(column, row, str(confusion[row, column]), ha="center", va="center",
                    color="white" if confusion[row, column] > confusion.max() / 2 else "black")
    ax.set(xticks=range(4), yticks=range(4), xlabel="Predicted sii", ylabel="True sii",
           title="Primary ensemble: counts summed across CV repeats")
    fig.tight_layout()
    fig.savefig(figures / "nested_confusion.png", dpi=150)
    plt.close(fig)


def render_pdf(markdown, output, root=PROJECT_ROOT):
    styles = getSampleStyleSheet()
    styles["Heading2"].keepWithNext = True
    styles["Title"].keepWithNext = True
    styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=7))
    styles.add(ParagraphStyle(name="ReportTable", parent=styles["BodyText"], fontSize=8, leading=10))
    story, paragraph = [], []

    def flush():
        if paragraph:
            text = escape(" ".join(paragraph).replace("`", ""))
            story.append(Paragraph(text, styles["ReportBody"]))
            paragraph.clear()

    lines, i = markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
        elif line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            story.append(Paragraph(escape(line.lstrip("# ")), styles["Title" if level == 1 else "Heading2"]))
        elif line.startswith("```"):
            flush()
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(code), ParagraphStyle(name="Code", fontName="Courier", fontSize=8, leading=11)))
            story.append(Spacer(1, 8))
        elif line.startswith("!["):
            flush()
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            if not match:
                raise ValueError("Malformed report figure")
            path = root / match.group(2)
            image = Image(str(path))
            ratio = min(6.7 * inch / image.imageWidth, 4.8 * inch / image.imageHeight)
            image.drawWidth, image.drawHeight = image.imageWidth * ratio, image.imageHeight * ratio
            story.extend([image, Spacer(1, 10)])
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                values = [v.strip() for v in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", v) for v in values):
                    rows.append([Paragraph(escape(v), styles["ReportTable"]) for v in values])
                i += 1
            i -= 1
            widths = [6.7 * inch / len(rows[0])] * len(rows[0])
            if len(rows[0]) > 2:
                widths = [2.7 * inch] + [(4.0 * inch) / (len(rows[0]) - 1)] * (len(rows[0]) - 1)
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4ecf4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
            ]))
            story.extend([table, Spacer(1, 10)])
        else:
            paragraph.append(line.strip())
        i += 1
    flush()

    def page_number(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(A4[0] - 48, 25, str(doc.page))

    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=48, leftMargin=48,
                            topMargin=45, bottomMargin=45, title="Anonymous technical report",
                            author="", subject="Ordinal tabular ML project")
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


def build_report(root=PROJECT_ROOT):
    import json
    from src.verify import verify_artifacts
    root = Path(root)
    verify_artifacts(root=root)
    results = root / "results"
    metadata = json.loads((results / "nested_metadata.json").read_text())
    summary = pd.read_csv(results / "nested_cv_results.csv")
    deltas = pd.read_csv(results / "nested_paired_deltas.csv")
    classes = pd.read_csv(results / "nested_class_metrics.csv")
    oof = pd.read_csv(results / "nested_oof_predictions.csv")
    make_plots(summary, oof, root)
    config, data = metadata["config"], metadata["data"]
    primary = summary.loc[summary.setup.eq(PRIMARY)].iloc[0]
    class_mean = classes.loc[classes.setup.eq(PRIMARY)].groupby("class")[["precision", "recall", "f1-score", "support", "true_positives"]].mean().reset_index()
    severe = class_mean.loc[class_mean["class"].eq(3)].iloc[0]
    statement = []
    for row in deltas.itertuples():
        verdict = "excludes zero" if row.interval_excludes_zero else "includes zero; a reliable gain is not established"
        statement.append(f"Against {row.reference}, the observed change is {row.delta_qwk:+.4f}; its interval {verdict}.")
    versions = metadata["versions"]
    replacements = {
        "DATA_DESCRIPTION": f"The raw training table contains {data['raw_train_rows']} participants: {data['labeled_rows']} labeled and {data['unlabeled_rows']} unlabeled. Class counts are " + ", ".join(f"{k}: {v}" for k, v in data["class_counts"].items()) + f". The example test contains {data['test_rows']} rows.",
        "PROTOCOL_DESCRIPTION": f"The measured run uses {len(config['outer_seeds'])} repeated ID-stable stratified outer CV runs, with seeds {config['outer_seeds']}, {config['outer_splits']} outer folds, and {config['inner_splits']} inner folds per outer training partition. CPU pools are limited to {config['threads']} threads. Early stopping has patience {config['stopping_rounds']}.",
        "RESULTS_TABLE": markdown_table(summary, ["setup", "mean_oof_qwk", "ci_low", "ci_high"]),
        "DELTA_TABLE": markdown_table(deltas, ["reference", "delta_qwk", "ci_low", "ci_high"]),
        "DELTA_INTERPRETATION": " ".join(statement),
        "CLASS_TABLE": markdown_table(class_mean, ["class", "precision", "recall", "f1-score", "true_positives"]),
        "ERROR_DESCRIPTION": f"Across repeats, severe-class recall is {severe['recall']:.3f}, with a mean {severe['true_positives']:.1f} true positives out of {int(severe['support'])} severe participants per repeat. The table averages class metrics across repeats; fractional true-positive counts are repeat means, not fractional participants. Confusion-matrix counts sum across repeats and are not counts of unique children.",
        "ENVIRONMENT_DESCRIPTION": "Reference runtime: " + ", ".join(f"{k} {v}" for k, v in versions.items()) + ".",
        "CONCLUSION": f"The prespecified tuned equal ensemble achieves mean nested OOF QWK {primary['mean_oof_qwk']:.4f}, with a conditional bootstrap interval [{primary['ci_low']:.4f}, {primary['ci_high']:.4f}]. " + " ".join(statement) + " The project implements several baselines, a trained proposed ensemble, and explicit inner-only hyperparameter modifications with measured outcomes; these are not claims of universal superiority.",
    }
    rendered = (root / "docs" / "project.md").read_text()
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    if re.search(r"\{\{.*?\}\}", rendered):
        raise ValueError("An unfilled report placeholder remains")
    (root / "docs" / "project.rendered.md").write_text(rendered)
    render_pdf(rendered, root / "project.pdf", root)
    provenance_files = (
        "docs/project.md", "src/report.py", "docs/project.rendered.md", "project.pdf",
        "results/figures/nested_qwk.png", "results/figures/nested_confusion.png",
    )
    provenance = {
        "experiment_artifact_sha256": metadata["artifact_sha256"],
        "report_sha256": {name: file_digest(root / name) for name in provenance_files},
    }
    (root / "docs" / "report_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Generated project.pdf and docs/project.rendered.md from verified results")


if __name__ == "__main__":
    build_report()
