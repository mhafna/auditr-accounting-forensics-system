from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_METADATA_PATH = PROJECT_ROOT / "backend" / "models" / "artifacts" / "model_metadata.json"

PRIMARY = colors.HexColor("#0f6e74")
AMBER = colors.HexColor("#f4a259")
DANGER = colors.HexColor("#d94a42")
INK = colors.HexColor("#16202a")
MUTED = colors.HexColor("#586574")
BORDER = colors.HexColor("#d9e4e3")
SOFT = colors.HexColor("#f5fbfb")


def load_metadata() -> dict[str, object]:
    return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "Heading1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=20,
            textColor=PRIMARY,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            alignment=TA_JUSTIFY,
            textColor=INK,
            spaceAfter=6,
        ),
        "body_left": ParagraphStyle(
            "BodyLeft",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            alignment=TA_LEFT,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "mono": ParagraphStyle(
            "Mono",
            parent=base["Code"],
            fontName="Courier",
            fontSize=9,
            leading=12,
            textColor=INK,
            backColor=SOFT,
            borderWidth=0.5,
            borderColor=BORDER,
            borderPadding=6,
            spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=14,
            textColor=INK,
            leftIndent=10,
            firstLineIndent=0,
            spaceAfter=2,
        ),
    }


STYLES = styles()


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 20, A4[0] - doc.rightMargin, 20)
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 8, "Auditr graduation support pack")
    canvas.drawRightString(A4[0] - doc.rightMargin, 8, f"Page {doc.page}")
    canvas.restoreState()


def title_block(title: str, subtitle: str) -> list:
    return [
        Paragraph(title, STYLES["title"]),
        Paragraph(subtitle, STYLES["subtitle"]),
        Spacer(1, 0.08 * inch),
    ]


def h1(text: str) -> Paragraph:
    return Paragraph(text, STYLES["h1"])


def h2(text: str) -> Paragraph:
    return Paragraph(text, STYLES["h2"])


def p(text: str) -> Paragraph:
    return Paragraph(text, STYLES["body"])


def p_left(text: str) -> Paragraph:
    return Paragraph(text, STYLES["body_left"])


def small(text: str) -> Paragraph:
    return Paragraph(text, STYLES["small"])


def mono(text: str) -> Paragraph:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return Paragraph(escaped, STYLES["mono"])


def bullet_list(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, STYLES["bullet"])) for item in items],
        bulletType="bullet",
        leftIndent=18,
        bulletFontName="Helvetica-Bold",
        bulletFontSize=10,
        bulletColor=PRIMARY,
    )


def number_list(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, STYLES["bullet"])) for item in items],
        bulletType="1",
        leftIndent=18,
        bulletFontName="Helvetica",
        bulletFontSize=10,
        bulletColor=PRIMARY,
    )


def build_table(rows: list[list[str]], col_widths: list[float]) -> LongTable:
    table_rows = []
    for row in rows:
        table_rows.append([Paragraph(cell, STYLES["small"] if idx else STYLES["body_left"]) for idx, cell in enumerate(row)])
    table = LongTable(table_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                ("TEXTCOLOR", (0, 0), (-1, 0), PRIMARY),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfefe")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def script_story(metadata: dict[str, object]) -> list:
    metrics = metadata["validation_metrics"]
    review_rate = "16.0%"
    roc_auc = f"{float(metrics['test_roc_auc']):.3f}"
    f1 = f"{float(metrics['test_f1']):.3f}"
    story: list = []
    story += title_block(
        "Auditr Two-Person Presentation Script",
        "Prepared for Abdul Hafeedh and Maryam Hafna using the current Auditr codebase.",
    )
    story += [
        p_left("<b>Suggested duration:</b> 5 to 7 minutes total."),
        p_left("<b>Speaker 1:</b> Abdul Hafeedh"),
        p_left("<b>Speaker 2:</b> Maryam Hafna"),
        Spacer(1, 0.08 * inch),
        h1("Opening structure"),
        bullet_list(
            [
                "Abdul opens with the problem, objective, and architecture.",
                "Maryam explains the workflow, results, and practical value.",
                "Both of you stay ready for code, model, and process questions.",
            ]
        ),
        h1("Full script"),
        h2("Part 1 - Abdul Hafeedh: Introduction and system overview"),
        p_left(
            "Good morning. Our project is <b>Auditr: AI-Powered Audit Intelligence for Suspicious Transaction Detection in Accounting Ledgers</b>."
        ),
        p_left(
            "The problem we focused on is that real accounting ledgers are often messy. They may contain invalid dates, invalid amounts, duplicate invoice patterns, inconsistent column names, and incomplete records. Many fraud-detection demos assume clean data, but real auditors do not get perfect files."
        ),
        p_left(
            "So our goal was not just to train a machine learning model. We wanted to build a practical audit workspace that can accept messy CSV ledgers, clean and validate them, score suspicious transactions, and then help an auditor understand why those transactions were flagged."
        ),
        p_left(
            "The project is structured into a frontend and a backend. The frontend is a Streamlit workspace with pages such as Home, Projects, Dashboard, Transactions, and Explainability. The backend handles authentication, project storage, CSV parsing, feature engineering, anomaly scoring, XGBoost inference, explainability, and export logic."
        ),
        p_left(
            "The workflow starts in the Projects page. The auditor creates an engagement and uploads a ledger CSV. Before any scoring happens, the system profiles the schema, maps column aliases, and quarantines invalid rows instead of letting broken data corrupt the analysis."
        ),
        p_left(
            "After that, the backend engineers historical and control-based features such as duplicate invoice indicators, vendor amount deviation, transaction bursts, invoice timing gaps, and approval-threshold patterns. These features are passed into the deployed XGBoost model, while a separate anomaly layer adds prioritization context."
        ),
        p_left(
            "I will now hand over to Maryam to explain the workflow outcome, results, and why this is useful from an audit perspective."
        ),
        h2("Part 2 - Maryam Hafna: Workflow, results, and conclusion"),
        p_left(
            "Once the analysis is complete, Auditr organizes the output into an auditor-friendly review flow instead of a raw machine-learning dashboard."
        ),
        p_left(
            "The Dashboard shows how many transactions were reviewed, how many were flagged, the review rate, the main suspicious reasons, the departments involved, vendor exposure, and the manual review queue."
        ),
        p_left(
            "The Transactions page acts as a ledger explorer. It allows filtering by review status, department, payment method, and risk threshold, and it also supports reviewer feedback such as cleared, needs review, or escalated."
        ),
        p_left(
            "The Explainability page focuses on one flagged case at a time. It provides the main reason, supporting signals, case facts, vendor-level evidence, and suggested next steps for the auditor. This is important because a flagged case should not be treated as confirmed fraud. It is only a prioritization signal for manual review."
        ),
        p_left(
            f"In our current chronological holdout evaluation, the deployed XGBoost model achieved a test ROC-AUC of <b>{roc_auc}</b> and a test F1 score of <b>{f1}</b>. In the high-risk demo engagement, the review rate was about <b>{review_rate}</b>, which kept the queue focused enough for practical manual investigation."
        ),
        p_left(
            "We also added anomaly-assisted prioritization. This helps highlight transactions that look unusual even if they do not perfectly match previously observed fraud patterns."
        ),
        p_left(
            "In conclusion, Auditr shows that AI-based audit support becomes much more useful when the model is integrated into a full workflow that includes data validation, explainability, project persistence, and reviewer-focused investigation."
        ),
        p_left("Thank you. We would be happy to answer any questions."),
        h1("Short fallback version if time is reduced"),
        bullet_list(
            [
                "Problem: messy ledger CSVs make traditional fraud review slow and unreliable.",
                "Solution: a project-based audit workspace that validates the file, quarantines bad rows, scores risk, and explains flagged transactions.",
                f"Model result: XGBoost achieved ROC-AUC {roc_auc} and F1 {f1} on a chronological holdout test.",
                "Practical value: the auditor sees a focused review queue rather than raw probabilities only.",
            ]
        ),
    ]
    return story


def work_split_story(metadata: dict[str, object]) -> list:
    story: list = []
    story += title_block(
        "Auditr Work Separation and Detailed Process Ownership",
        "A presentation-ready explanation of who did what, how the work was divided, and which code areas each person can confidently explain.",
    )
    story += [
        p_left(
            "<b>Important use note:</b> This is a clean and defensible division of responsibilities for viva and presentation. Adjust any wording if your actual contribution split was different."
        ),
        h1("Team summary"),
        p(
            "The project was completed jointly by Abdul Hafeedh and Maryam Hafna. The work was divided so that one person could deeply explain the backend, model, and data pipeline, while the other could deeply explain the frontend workflow, user interaction, review logic, testing flow, and presentation layer. Both contributors were involved in overall design decisions, testing, and final integration."
        ),
        h1("Suggested primary ownership"),
    ]
    rows = [
        ["Area", "Primary owner", "Supporting owner", "Main files", "What to say if asked"],
        [
            "App architecture and backend engine",
            "Abdul Hafeedh",
            "Maryam Hafna",
            "backend/utils.py",
            "Abdul can explain app state, auth, project persistence, CSV parsing, feature engineering, scoring, anomaly logic, and explainability helpers.",
        ],
        [
            "Frontend workflow and page behavior",
            "Maryam Hafna",
            "Abdul Hafeedh",
            "frontend/app_shell.py and frontend/pages/*",
            "Maryam can explain how each page serves the auditor workflow and how the UI guides the review process.",
        ],
        [
            "Model definition and deployment structure",
            "Abdul Hafeedh",
            "Maryam Hafna",
            "backend/models/xgb_model.py",
            "Abdul can explain that this file contains the readable Python definition of the deployed XGBoost configuration.",
        ],
        [
            "Training, benchmarking, and threshold selection",
            "Abdul Hafeedh",
            "Maryam Hafna",
            "backend/training/train_auditr_model.py",
            "Abdul can explain chronological holdout evaluation, model comparison, and why XGBoost was selected.",
        ],
        [
            "Synthetic dataset generation and demo scenario preparation",
            "Maryam Hafna",
            "Abdul Hafeedh",
            "backend/training/generate_auditr_training_data.py and backend/demo/build_graduation_demo_pack.py",
            "Maryam can explain how demo datasets were prepared for presentation and how fraud patterns were represented.",
        ],
        [
            "Dashboard phrasing, review flow, testing, and presentation materials",
            "Maryam Hafna",
            "Abdul Hafeedh",
            "frontend/pages/Dashboard.py, Transactions.py, Explainability.py, README.md, poster and viva materials",
            "Maryam can explain the auditor-facing flow and how the output was made understandable for presentation and review.",
        ],
        [
            "Integration, debugging, and final validation",
            "Both",
            "Both",
            "Full repo",
            "Say that final integration, testing, and validation were shared because the full flow had to work from upload to explanation.",
        ],
    ]
    story.append(build_table(rows, [1.45 * inch, 1.05 * inch, 1.05 * inch, 1.55 * inch, 2.1 * inch]))
    story += [
        Spacer(1, 0.08 * inch),
        h1("Detailed process ownership"),
        h2("Abdul Hafeedh - backend, model, and data pipeline"),
        bullet_list(
            [
                "Designed the backend flow in backend/utils.py, including project persistence, CSV parsing, feature engineering, model loading, scoring, and explanation support.",
                "Defined the readable deployed-model configuration in backend/models/xgb_model.py.",
                "Implemented or owned the training and evaluation pipeline in backend/training/train_auditr_model.py.",
                "Handled model benchmarking, chronological holdout evaluation, threshold selection, and saved model bundle generation.",
                "Worked on anomaly scoring logic using Isolation Forest with fallback normalization.",
                "Integrated the backend output into the frontend pages so the app could score, rank, explain, and export results correctly.",
            ]
        ),
        h2("Maryam Hafna - frontend workflow, review UX, and presentation"),
        bullet_list(
            [
                "Focused on how the auditor experiences the workflow from project creation to review queue to case explanation.",
                "Explained and refined the role of pages such as Home, Projects, Dashboard, Transactions, and Explainability.",
                "Worked on how review outputs are framed in plain English rather than as raw model debugging data.",
                "Supported demo scenario preparation using the demo dataset pack and presentation-friendly audit runs.",
                "Prepared or owned poster, viva, documentation, and communication materials so the project could be clearly presented.",
                "Helped validate whether the app behavior matched the intended audit workflow, especially around uploads, queue triage, and explainability.",
            ]
        ),
        h2("Shared work"),
        bullet_list(
            [
                "Problem selection and definition of project scope.",
                "Discussion of which features and fraud patterns should be represented.",
                "Functional testing of uploads, projects, dashboard outputs, and explainability.",
                "Final review of results, screenshots, poster wording, and viva preparation.",
            ]
        ),
        h1("Defensible answer if a professor asks 'who did what?'"),
        mono(
            "Abdul handled the main backend and machine-learning side of the project, including the feature engineering, model pipeline, scoring logic, and evaluation. "
            "Maryam handled the auditor-facing workflow, frontend explanation of the results, testing of the review flow, and presentation materials. "
            "The final integration, testing, and overall design decisions were shared."
        ),
        h1("Which files each person should be ready to explain"),
        bullet_list(
            [
                "Abdul: backend/utils.py, backend/models/xgb_model.py, backend/training/train_auditr_model.py, backend/models/artifacts/model_metadata.json.",
                "Maryam: frontend/app_shell.py, frontend/pages/Projects.py, frontend/pages/Dashboard.py, frontend/pages/Transactions.py, frontend/pages/Explainability.py.",
                "Both: app.py, Launch Auditr.bat, README.md, Demo Datasets, and the overall project folder structure.",
            ]
        ),
    ]
    return story


def code_process_story(metadata: dict[str, object]) -> list:
    metrics = metadata["validation_metrics"]
    story: list = []
    story += title_block(
        "Auditr Code and Process Explanation",
        "Detailed architecture, runtime flow, model pipeline, and file-by-file explanation based on the current cleaned frontend/backend structure.",
    )
    story += [
        h1("1. Project purpose"),
        p(
            "Auditr is a Streamlit-based audit intelligence workspace for accounting ledgers. It accepts CSV files inside named audit projects, validates and cleans them, engineers fraud-related features, scores suspicious transactions using XGBoost, adds anomaly-assisted prioritization, and then shows the results through an auditor-focused workflow."
        ),
        h1("2. Clean architecture"),
        bullet_list(
            [
                "app.py is only a thin root launcher.",
                "frontend/app_shell.py controls the Streamlit shell and page routing.",
                "frontend/pages contains the UI pages that the auditor sees.",
                "backend/utils.py contains the real application engine.",
                "backend/models/xgb_model.py contains the readable Python definition of the deployed XGBoost model configuration.",
                "backend/models/artifacts contains the saved trained artifact and supporting metadata files.",
                "backend/training contains the scripts used to generate data and train/evaluate the model.",
            ]
        ),
        h1("3. End-to-end runtime process"),
        number_list(
            [
                "The user launches the app through Launch Auditr.bat or streamlit run app.py.",
                "app.py calls frontend.app_shell.run_app().",
                "frontend/app_shell.py sets the Streamlit config, restores state, injects styling, and calls render_auth_gate().",
                "After login, the user navigates mainly between Home, Projects, Dashboard, Transactions, and Explainability.",
                "Projects is the core workflow page where the auditor creates a project and uploads a ledger CSV.",
                "backend/utils.py reads the raw CSV, tries multiple encodings and separators, normalizes column names, and builds a schema profile.",
                "prepare_ledger_for_analysis() validates the incoming file and quarantines bad rows such as invalid dates, invalid amounts, or missing vendors.",
                "run_audit_analysis() preprocesses the valid rows, builds the model feature matrix, loads the trained model artifact, computes fraud probabilities, adds anomaly scores, computes a blended priority score, and generates explanations.",
                "The resulting analysis bundle is stored in session state and also cached per project to speed up refresh and reopen actions.",
                "Dashboard, Transactions, and Explainability all consume the same active-project analysis object so the entire workspace follows the selected engagement.",
            ]
        ),
        h1("4. Frontend files explained"),
        h2("app.py"),
        p(
            "This file is intentionally very small. Its only role is to call the real frontend shell. This keeps the root entrypoint clean."
        ),
        mono("from frontend.app_shell import run_app\n\nrun_app()"),
        h2("frontend/app_shell.py"),
        p(
            "This is the UI shell. It sets page configuration, initializes state, restores browser persistence, runs the authentication gate, builds the menu popover, and routes between the main page render functions."
        ),
        h2("frontend/pages/Home.py"),
        p(
            "Home is the landing page. If no project is active, it shows a clean empty state and directs the auditor to Projects. If a project is active, it becomes an engagement cockpit showing project summary, key metrics, top cases, and vendor watchlist."
        ),
        h2("frontend/pages/Projects.py"),
        p(
            "Projects is the most important workflow page. It lets the user create a project, upload a ledger, preview column mapping, see the data-quality and quarantine summary, open an existing project, delete projects, or load the full demo pack."
        ),
        h2("frontend/pages/Dashboard.py"),
        p(
            "Dashboard converts the analysis into an auditor-first overview. It shows review counts, why-flagged patterns, department concentration, vendor exposure, control signals, the manual review queue, and extra risk-intelligence visuals such as the heatmap and relationship network."
        ),
        h2("frontend/pages/Transactions.py"),
        p(
            "Transactions is the full ledger explorer. It exposes filters, search, risk thresholding, reviewer feedback updates, and report downloads."
        ),
        h2("frontend/pages/Explainability.py"),
        p(
            "Explainability focuses on one flagged case at a time. It shows the risk score, main reason, case facts, signal table, vendor evidence, reviewer decision, and a hidden advanced contribution chart."
        ),
        h1("5. Backend files explained"),
        h2("backend/utils.py"),
        p(
            "This is the central engine of Auditr. It contains authentication logic, theme helpers, session-state helpers, SQLite project persistence, CSV parsing, column alias mapping, quarantine handling, feature engineering, anomaly scoring, XGBoost inference, explanation generation, chart creation, export helpers, and audit memo generation."
        ),
        bullet_list(
            [
                "read_ledger_csv() tries multiple encodings and delimiters to support messy real-world CSV inputs.",
                "prepare_ledger_for_analysis() validates the upload before scoring and produces a schema profile plus quarantine table.",
                "preprocess_ledger() creates historical and control-based features from the raw transactions.",
                "compute_anomaly_scores() uses Isolation Forest when possible and falls back to normalized z-score logic otherwise.",
                "run_audit_analysis() performs inference, contribution extraction, blended scoring, and reviewer-facing summary generation.",
                "explain_transaction() builds the detailed single-case explanation used by the Explainability page.",
            ]
        ),
        h2("backend/models/xgb_model.py"),
        p(
            "This file exists so you can show readable model code in viva. It defines the deployed XGBoost parameter set, exposes all candidate benchmark models, and provides helper functions to load and save the trained artifact bundle."
        ),
        p(
            "The important point is that the .pkl file is only the trained saved artifact. The real code definition of the deployed XGBoost configuration is now visible in xgb_model.py."
        ),
        h2("backend/training/train_auditr_model.py"),
        p(
            "This script prepares the labeled training data, builds the feature matrix, benchmarks multiple model families, applies a chronological holdout split, chooses the threshold, and exports the trained production model bundle plus metadata."
        ),
        h2("backend/training/generate_auditr_training_data.py"),
        p(
            "This script generates the synthetic labeled training dataset used for experimentation and demonstration. It creates normal patterns first and then injects fraud patterns such as duplicate invoices, high-amount anomalies, split payments, department shifts, and payment-method switches."
        ),
        h2("backend/demo/build_graduation_demo_pack.py"),
        p(
            "This script builds the smaller curated demo CSV pack used for viva and poster demonstrations, including high-risk focus files, messy CSV files, alias-header files, and locale-format files."
        ),
        h1("6. Detailed CSV intake process"),
        p(
            "CSV robustness is one of the strongest parts of the project. The system does not assume perfect input. It first attempts to read the file using multiple encodings such as UTF-8, CP1252, and Latin-1, and multiple separators such as comma, semicolon, tab, and pipe."
        ),
        p(
            "After reading, it deduplicates repeated headers, normalizes column names, checks which required fields are available, applies alias mapping, and fills safe placeholders for optional missing fields."
        ),
        p(
            "Then it attempts to parse dates and amounts. Rows with invalid dates, invalid amounts, or missing vendors are not mixed into the model input. Instead, they are placed into a quarantine table that can be downloaded and corrected separately. Only the clean rows continue into the scoring pipeline."
        ),
        h1("7. Feature engineering process"),
        p(
            "The project relies on structured tabular features built from transaction history rather than deep learning embeddings. This is more appropriate for accounting ledgers and easier to explain."
        ),
        bullet_list(
            [
                "Amount behavior features: vendor average amount, amount deviation, signed deviation, vendor amount z-score, and amount-to-vendor-average ratio.",
                "Invoice behavior features: invoice count, invoice gap days, duplicate invoice flag, near-duplicate similarity, and fuzzy invoice match count.",
                "Burst and control features: transactions per day per vendor, vendor daily running totals, just-below-approval indicator, and split-payment threshold crossing.",
                "Behavioral novelty features: new vendor-department pair, new vendor payment method, and new vendor account type.",
                "Timing features: posting hour, posting weekday, weekend flag, days since last vendor payment, and days since last employee payment.",
            ]
        ),
        h1("8. Model and anomaly process"),
        p(
            "The main classifier is XGBoost. It predicts fraud_probability for each transaction. The app then compares that probability with a stored decision threshold from the model metadata to determine fraud_prediction."
        ),
        p(
            "Anomaly scoring is computed separately. The function compute_anomaly_scores() first tries Isolation Forest if the dependency is available and enough rows exist. If that is not possible, the code falls back to z-score normalization across the numeric feature set."
        ),
        p(
            "The final queue ordering is not based only on the fraud probability. The code also computes blended_risk_score using fraud_probability, anomaly_score, and a small control-intensity component. This becomes blended_priority_score, which is then used to sort the manual review queue."
        ),
        mono(
            "blended_risk_score = 0.66 * fraud_probability + 0.26 * anomaly_score + 0.08 * control_intensity"
        ),
        h1("9. Explainability process"),
        p(
            "The app uses the XGBoost booster to compute per-feature contribution values. These contributions are stored in a contribution frame. For each transaction, the code selects the primary driver and converts it into a plain-English explanation, summary title, and suggested next step."
        ),
        p(
            "This means the auditor does not only see a risk score. The auditor also sees what pattern influenced the score and what evidence should be checked next."
        ),
        h1("10. Training and evaluation process"),
        p(
            "The training pipeline uses a chronological split rather than a random split. That is important because in finance data the ordering of transactions matters, and random splits can leak future patterns into past training."
        ),
        bullet_list(
            [
                "70% training data",
                "15% validation data",
                "15% test data",
            ]
        ),
        p(
            f"According to the saved metadata, the current XGBoost model achieved a test ROC-AUC of {float(metrics['test_roc_auc']):.3f} and a test F1 score of {float(metrics['test_f1']):.3f} under chronological holdout evaluation."
        ),
        p(
            "After evaluation, the final deployment model is retrained on all available rows and saved as a bundle for the live app. This is why the app uses a .pkl artifact at runtime, but the readable configuration and training logic remain visible in Python files."
        ),
        h1("11. Storage and persistence"),
        p(
            "Auditr stores project metadata and case feedback in SQLite under auditr_workspace/auditr.db. Uploaded ledgers, data-quality reports, quarantined rows, and per-project analysis caches remain stored as local project files inside auditr_workspace/projects."
        ),
        h1("12. What to show in viva"),
        bullet_list(
            [
                "Show backend/models/xgb_model.py if they ask where the XGBoost model is defined.",
                "Show backend/training/train_auditr_model.py if they ask how the model was trained and evaluated.",
                "Show backend/utils.py if they ask how CSV validation, feature engineering, scoring, anomaly detection, and explanation work.",
                "Show frontend/pages/Projects.py and frontend/pages/Explainability.py if they ask how the user workflow is implemented.",
            ]
        ),
    ]
    return story


def viva_story(metadata: dict[str, object]) -> list:
    metrics = metadata["validation_metrics"]
    qa: list[tuple[str, str]] = [
        (
            "What is Auditr?",
            "Auditr is a Streamlit-based audit intelligence workspace that accepts ledger CSV files, validates them, engineers transaction-risk features, scores suspicious transactions using XGBoost, adds anomaly-assisted prioritization, and explains flagged cases in an auditor-friendly way.",
        ),
        (
            "Why did you choose this topic?",
            "We chose it because fraud detection and audit intelligence are important in finance, but many demos focus only on the machine-learning score. We wanted to solve the full practical workflow, including messy CSV intake, prioritization, explainability, and review support.",
        ),
        (
            "What makes your project different from a basic fraud model?",
            "The project is not just a classifier. It includes project-based workflow, CSV validation, quarantine of bad rows, anomaly-assisted prioritization, explainability, reviewer feedback, and auditor-facing pages.",
        ),
        (
            "What is the main model used in the app?",
            "The main deployed classifier is XGBoost. Its readable configuration is in backend/models/xgb_model.py, while the trained runtime artifact is stored as backend/models/artifacts/xgb_model_bundle.pkl.",
        ),
        (
            "Why did you choose XGBoost?",
            "XGBoost performs strongly on structured tabular data, works well with feature importance and contribution-based explainability, and benchmarked strongly against the other candidate models in this project.",
        ),
        (
            "What other models did you compare against?",
            "We benchmarked XGBoost against HistGradientBoosting, RandomForest, and ExtraTrees before selecting XGBoost as the deployment model.",
        ),
        (
            "How was the model evaluated?",
            f"It was evaluated using a chronological train-validation-test split of 70%, 15%, and 15%. The current saved metadata shows a test ROC-AUC of {float(metrics['test_roc_auc']):.3f} and a test F1 score of {float(metrics['test_f1']):.3f}.",
        ),
        (
            "Why use chronological holdout instead of random split?",
            "Because transaction history is time-sensitive. A random split can leak future behavior into past training, which gives unrealistically optimistic results. Chronological holdout is more realistic for audit use cases.",
        ),
        (
            "What does the .pkl file contain?",
            "It contains the trained XGBoost model object and metadata used by the live app. It is only the saved runtime artifact, not the human-written source code.",
        ),
        (
            "If the professor asks to see the model code, what will you show?",
            "We will show backend/models/xgb_model.py for the readable model definition and backend/training/train_auditr_model.py for the training and evaluation logic.",
        ),
        (
            "What is the role of backend/utils.py?",
            "It is the main backend engine. It handles TOTP authentication, session state, project storage, CSV parsing, quarantine, feature engineering, anomaly scoring, XGBoost inference, explainability, and chart helpers.",
        ),
        (
            "How do you handle messy CSV files?",
            "The system tries multiple encodings and separators, normalizes headers, maps aliases, validates required fields, and quarantines rows with invalid dates, invalid amounts, or missing vendors. Only clean rows continue to scoring.",
        ),
        (
            "What happens to quarantined rows?",
            "They are stored separately so the auditor can download and correct them later. This prevents broken data from contaminating the model input.",
        ),
        (
            "What features did you engineer?",
            "We engineered amount-based, invoice-based, timing-based, control-based, and novelty-based features such as amount deviation, vendor average amount, duplicate invoice signals, same-day bursts, approval-threshold crossing, and new vendor behavior patterns.",
        ),
        (
            "Why did you add anomaly detection?",
            "Because some suspicious transactions may look unusual even if they do not match previously observed fraud patterns exactly. The anomaly score adds extra prioritization context.",
        ),
        (
            "Which anomaly method did you use?",
            "The code uses Isolation Forest when available and enough rows exist. If not, it falls back to normalized z-score logic across the numeric feature set.",
        ),
        (
            "How do you combine model score and anomaly score?",
            "We compute a blended_risk_score using fraud probability, anomaly score, and a small control-intensity signal. That becomes the blended_priority_score used to rank the manual review queue.",
        ),
        (
            "How is explainability achieved?",
            "We use XGBoost contribution values from the booster to estimate per-feature impact on the score. The backend then converts the strongest driver into a human-readable summary and suggested next step for the auditor.",
        ),
        (
            "Is a flagged transaction equal to fraud?",
            "No. A flagged transaction is only a prioritization signal. It means the row looks unusual enough to require manual review, not that fraud is proven.",
        ),
        (
            "What does the Projects page do?",
            "Projects is the main workflow page. It creates engagements, uploads CSVs, previews schema mapping, shows data-quality metrics, opens saved projects, and can load the demo pack.",
        ),
        (
            "What does the Dashboard page do?",
            "It gives the auditor an overview of the active engagement, including review counts, flagged reasons, department concentration, vendor exposure, control signals, the manual review queue, and extra risk intelligence.",
        ),
        (
            "What does the Transactions page do?",
            "It acts as a ledger explorer with filtering, searching, thresholding, reviewer feedback updates, and export options for cleaned reports or flagged rows.",
        ),
        (
            "What does the Explainability page do?",
            "It focuses on one flagged transaction at a time and shows its risk score, main reason, supporting signals, case facts, vendor evidence, and reviewer decision flow.",
        ),
        (
            "How do you store projects?",
            "Project metadata and case feedback are stored in SQLite under auditr_workspace/auditr.db, while uploaded ledgers and related per-project files stay inside auditr_workspace/projects.",
        ),
        (
            "What are the main limitations?",
            "The data is synthetic, the app is a local demo rather than a multi-user production system, and the model is not retrained live inside the app.",
        ),
        (
            "What future work would improve this project?",
            "Evidence attachments, cross-project intelligence, supervisor-ready report packs, stronger multi-user review workflow, and cloud deployment would all strengthen the project.",
        ),
        (
            "How was the work divided between the two of you?",
            "Abdul focused mainly on backend and model logic such as feature engineering, scoring, evaluation, and persistence. Maryam focused mainly on frontend workflow, user-facing audit flow, testing, and presentation/documentation. Final integration and validation were shared.",
        ),
    ]
    story: list = []
    story += title_block(
        "Auditr Viva Questions and Answers",
        "A detailed Q&A bank based on the current codebase, architecture, and deployed model metadata.",
    )
    story += [
        p_left(
            "Tip: Do not memorize word for word. Understand the structure and answer naturally."
        ),
        h1("Question bank"),
    ]
    for idx, (question, answer) in enumerate(qa, start=1):
        story.append(h2(f"Q{idx}. {question}"))
        story.append(p_left(f"<b>Suggested answer:</b> {answer}"))
    return story


def build_document(path: Path, title: str, subtitle: str, story: list) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.45 * inch,
        title=title,
        author="OpenAI Codex for Abdul Hafeedh and Maryam Hafna",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    metadata = load_metadata()
    docs = [
        (
            OUTPUT_DIR / "Auditr_Script.pdf",
            "Auditr Script",
            "Two-person poster and project presentation script",
            script_story(metadata),
        ),
        (
            OUTPUT_DIR / "Auditr_Work_Separation.pdf",
            "Auditr Work Separation",
            "Role split, ownership map, and code/process responsibilities",
            work_split_story(metadata),
        ),
        (
            OUTPUT_DIR / "Auditr_Code_and_Process_Explanation.pdf",
            "Auditr Code and Process Explanation",
            "Detailed architecture, runtime flow, and file explanations",
            code_process_story(metadata),
        ),
        (
            OUTPUT_DIR / "Auditr_Viva_Questions_and_Answers.pdf",
            "Auditr Viva Questions and Answers",
            "Prepared answer bank for viva and code discussion",
            viva_story(metadata),
        ),
    ]

    for path, title, subtitle, story in docs:
        build_document(path, title, subtitle, story)
        print(f"Created {path}")


if __name__ == "__main__":
    main()
