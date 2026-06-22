from __future__ import annotations

import base64
import difflib
import hashlib
import html
import hmac
import io
import json
import os
import re
from pathlib import Path
import pickle
import sqlite3
import shutil
import struct
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb

from backend.models.xgb_model import MODEL_BUNDLE_PATH, load_saved_model_bundle

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover - optional dependency
    IsolationForest = None

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency
    nx = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = MODEL_BUNDLE_PATH
WORKSPACE_ROOT = PROJECT_ROOT / "auditr_workspace"
PROJECTS_ROOT = WORKSPACE_ROOT / "projects"
PROJECTS_INDEX_PATH = WORKSPACE_ROOT / "projects_index.json"
PROJECTS_DB_PATH = WORKSPACE_ROOT / "auditr.db"
DEMO_DATASETS_ROOT = PROJECT_ROOT / "Demo Datasets"
PROJECT_ANALYSIS_CACHE_FILE = "analysis_cache.pkl"
PROJECT_ANALYSIS_CACHE_VERSION = 1
SAMPLE_DATASETS = {
    "High Risk Demo": DEMO_DATASETS_ROOT / "demo_high_risk.csv",
    "Subtle Risk Demo": DEMO_DATASETS_ROOT / "demo_subtle (1).csv",
    "Normal Demo": DEMO_DATASETS_ROOT / "demo_normal (1).csv",
}
REQUIRED_COLUMNS = [
    "transaction_id",
    "date",
    "amount",
    "vendor",
    "department",
    "account_type",
    "payment_method",
    "employee",
    "invoice_id",
]
ESSENTIAL_INPUT_COLUMNS = [
    "date",
    "amount",
    "vendor",
]
OPTIONAL_INPUT_DEFAULTS = {
    "department": "Unassigned",
    "account_type": "Unspecified",
    "payment_method": "Other",
    "employee": "Unknown employee",
}
FEEDBACK_STATUS_OPTIONS = ["Needs review", "Cleared", "Escalated"]
RISK_LEVEL_ORDER = ["Low", "Medium", "High", "Critical"]
COLUMN_ALIAS_EXAMPLES = {
    "transaction_id": ["Transaction ID", "TransactionID", "Txn ID", "Reference Number"],
    "date": ["Date", "Transaction Date", "Posting Date", "Payment Date"],
    "amount": ["Amount", "Transaction Amount", "Payment Amount", "Value"],
    "vendor": ["Vendor", "Supplier", "Payee", "Vendor Name"],
    "department": ["Department", "Dept", "Cost Center", "Business Unit"],
    "account_type": ["Account Type", "Account Category", "Expense Category", "GL Category"],
    "payment_method": ["Payment Method", "Payment Type", "Payment Mode", "Method"],
    "employee": ["Employee", "Employee ID", "Requester", "Prepared By"],
    "invoice_id": ["Invoice ID", "Invoice Number", "Invoice No", "Bill Number"],
}
COLUMN_ALIASES = {
    "transactionid": "transaction_id",
    "transactionno": "transaction_id",
    "transactionnumber": "transaction_id",
    "txnid": "transaction_id",
    "reference": "transaction_id",
    "referencenumber": "transaction_id",
    "transactiondate": "date",
    "postingdate": "date",
    "paymentdate": "date",
    "txndate": "date",
    "valuedate": "date",
    "transactionamount": "amount",
    "paymentamount": "amount",
    "value": "amount",
    "grossamount": "amount",
    "supplier": "vendor",
    "suppliername": "vendor",
    "vendorname": "vendor",
    "payee": "vendor",
    "dept": "department",
    "costcenter": "department",
    "businessunit": "department",
    "function": "department",
    "account": "account_type",
    "accountcategory": "account_type",
    "expensecategory": "account_type",
    "glcategory": "account_type",
    "paymenttype": "payment_method",
    "paymentmode": "payment_method",
    "method": "payment_method",
    "employeeid": "employee",
    "requester": "employee",
    "preparedby": "employee",
    "submittedby": "employee",
    "invoice": "invoice_id",
    "invoicenumber": "invoice_id",
    "invoiceno": "invoice_id",
    "billnumber": "invoice_id",
}
PAYMENT_METHOD_DISPLAY = {
    "ACH": "Bank transfer (ACH)",
    "Wire": "Bank transfer (Wire)",
    "Card": "Company card",
    "Check": "Check",
    "Other": "Other / not specified",
}
BASE_FEATURES = [
    "amount",
    "vendor_transaction_count",
    "vendor_avg_amount",
    "amount_deviation_signed",
    "amount_deviation_abs",
    "amount_to_vendor_avg_ratio",
    "vendor_amount_zscore",
    "is_round_number",
    "transactions_per_day_vendor",
    "invoice_count",
    "invoice_gap_days",
    "fuzzy_invoice_match_count",
    "max_invoice_similarity",
    "vendor_department_count",
    "employee_vendor_count",
    "employee_department_count",
    "department_avg_amount",
    "department_amount_deviation",
    "vendor_payment_method_count",
    "vendor_account_type_count",
    "vendor_day_running_total_prior",
    "vendor_day_running_total_with_current",
    "is_just_below_approval",
    "crosses_approval_with_same_day_vendor",
    "is_new_vendor_department_pair",
    "is_new_vendor_payment_method",
    "is_new_vendor_account_type",
    "days_since_vendor_last_payment",
    "days_since_employee_last_payment",
    "posting_hour",
    "posting_weekday",
    "is_weekend",
    "is_duplicate_invoice",
]
CATEGORICAL_PREFIXES = {
    "department": "department_",
    "account_type": "account_type_",
    "payment_method": "payment_method_",
}
FEATURE_LABELS = {
    "amount": "Transaction amount",
    "vendor_transaction_count": "Earlier vendor transaction count",
    "vendor_avg_amount": "Vendor historical average",
    "amount_deviation_signed": "Signed vendor amount deviation",
    "amount_deviation_abs": "Large change from vendor pattern",
    "amount_to_vendor_avg_ratio": "Unexpected payment size",
    "vendor_amount_zscore": "Vendor amount z-score",
    "is_round_number": "Rounded amount pattern",
    "transactions_per_day_vendor": "Repeated same-day payments",
    "invoice_count": "Repeated invoice number",
    "invoice_gap_days": "Invoice returned quickly",
    "fuzzy_invoice_match_count": "Near-duplicate invoice number",
    "max_invoice_similarity": "Near-duplicate invoice score",
    "vendor_department_count": "Vendor-department history",
    "employee_vendor_count": "Employee-vendor history",
    "employee_department_count": "Employee-department history",
    "department_avg_amount": "Department historical average",
    "department_amount_deviation": "Department amount deviation",
    "vendor_payment_method_count": "Vendor payment-method history",
    "vendor_account_type_count": "Vendor account-type history",
    "vendor_day_running_total_prior": "Existing same-day vendor total",
    "vendor_day_running_total_with_current": "Same-day vendor total",
    "is_just_below_approval": "Just below approval limit",
    "crosses_approval_with_same_day_vendor": "Split-payment threshold crossing",
    "is_new_vendor_department_pair": "New vendor-department pairing",
    "is_new_vendor_payment_method": "New vendor payment method",
    "is_new_vendor_account_type": "New vendor account coding",
    "days_since_vendor_last_payment": "Gap since last vendor payment",
    "days_since_employee_last_payment": "Gap since last employee payment",
    "is_duplicate_invoice": "Duplicate invoice signal",
    "posting_hour": "Unusual payment time",
    "posting_weekday": "Day of week pattern",
    "is_weekend": "Weekend entry",
}
PAGE_META = {
    "home": {
        "label": "Home",
        "eyebrow": "Auditr workspace",
        "copy": "See the active engagement, understand the workflow, and jump into the next audit action without searching for the right page.",
    },
    "projects": {
        "label": "Projects",
        "eyebrow": "Project library",
        "copy": "Create, open, update, and remove audit engagements so the rest of the app always follows the selected project.",
    },
    "dashboard": {
        "label": "Dashboard",
        "eyebrow": "Auditor overview",
        "copy": "See what needs attention first, why the model is concerned, and where the risk sits in plain English.",
    },
    "transactions": {
        "label": "Transactions",
        "eyebrow": "Transaction search",
        "copy": "Filter the ledger, isolate high-risk entries, and export an audit-ready report with clear review labels.",
    },
    "explainability": {
        "label": "Explainability",
        "eyebrow": "Single transaction analysis",
        "copy": "Open one flagged transaction at a time and translate the model output into an auditor-friendly explanation.",
    },
    "profile": {
        "label": "Profile",
        "eyebrow": "Workspace identity",
        "copy": "Check who is signed in, how the current session is secured, and what the app is using right now.",
    },
    "settings": {
        "label": "Settings",
        "eyebrow": "Workspace controls",
        "copy": "Manage the theme, swap data sources, and review the TOTP security setup used by this local demo.",
    },
    "support": {
        "label": "Help & Support",
        "eyebrow": "Guidance and troubleshooting",
        "copy": "Understand what Auditr expects, what the payment terms mean, and how to fix common upload or review issues.",
    },
}
THEMES = {
    "Canvas": {
        "bg_top": "#f5efe4",
        "bg_bottom": "#ecf3ef",
        "surface": "rgba(255, 255, 255, 0.86)",
        "surface_strong": "#ffffff",
        "surface_alt": "#f3f7f6",
        "text": "#16202a",
        "muted": "#586574",
        "line": "rgba(22, 32, 42, 0.10)",
        "accent": "#0f6e74",
        "accent_soft": "#dcefee",
        "amber": "#f4a259",
        "danger": "#d94a42",
        "success": "#118562",
        "shadow": "rgba(22, 32, 42, 0.08)",
        "plot_bg": "rgba(255,255,255,0)",
    },
    "Midnight": {
        "bg_top": "#0f1723",
        "bg_bottom": "#111d2c",
        "surface": "rgba(17, 24, 39, 0.88)",
        "surface_strong": "#111827",
        "surface_alt": "#172233",
        "text": "#eef4ff",
        "muted": "#a8b4c4",
        "line": "rgba(255, 255, 255, 0.10)",
        "accent": "#55c2c7",
        "accent_soft": "#17353c",
        "amber": "#ffb55f",
        "danger": "#ff6b63",
        "success": "#4ecf8b",
        "shadow": "rgba(0, 0, 0, 0.22)",
        "plot_bg": "rgba(0,0,0,0)",
    },
    "Fern": {
        "bg_top": "#eef6ef",
        "bg_bottom": "#f8f3e8",
        "surface": "rgba(255, 255, 255, 0.88)",
        "surface_strong": "#ffffff",
        "surface_alt": "#eef5f0",
        "text": "#17231e",
        "muted": "#5e6a63",
        "line": "rgba(23, 35, 30, 0.10)",
        "accent": "#2f7d4d",
        "accent_soft": "#dceede",
        "amber": "#f0a451",
        "danger": "#d95d39",
        "success": "#2f7d4d",
        "shadow": "rgba(23, 35, 30, 0.08)",
        "plot_bg": "rgba(255,255,255,0)",
    },
}
DEFAULT_THEME = "Canvas"
DEFAULT_PAGE = "home"
DEFAULT_AUTH_USER = "auditor"
DEFAULT_AUTH_SECRET = "JBSWY3DPEHPK3PXP"
THEME_COOKIE_NAME = "auditr_theme"
AUTH_COOKIE_NAME = "auditr_session"
PROJECT_COOKIE_NAME = "auditr_project"
THEME_COOKIE_MAX_AGE = 60 * 60 * 24 * 120
AUTH_SESSION_MAX_AGE = 60 * 60 * 12
PROJECT_COOKIE_MAX_AGE = 60 * 60 * 24 * 30
DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_PRIORITY_THRESHOLD = 0.85
DEFAULT_WATCHLIST_THRESHOLD = 0.35
DEFAULT_MODEL_VERSION = "auditr-xgb-controls-v4"
APPROVAL_THRESHOLDS = (2500.0, 5000.0, 10000.0)
PROJECT_STATUS_OPTIONS = ["In progress", "Waiting for evidence", "Under review", "Closed"]
PROJECT_RECORD_FIELDS = [
    "id",
    "name",
    "client",
    "status",
    "notes",
    "created_at_utc",
    "updated_at_utc",
    "source_file",
    "ledger_path",
    "rows",
    "flagged_count",
    "review_rate",
]


def derive_category_columns(feature_names: list[str]) -> dict[str, list[str]]:
    return {
        source_column: [
            column
            for column in feature_names
            if column.startswith(prefix) and column not in BASE_FEATURES
        ]
        for source_column, prefix in CATEGORICAL_PREFIXES.items()
    }


def ensure_workspace_store() -> None:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_project_database()


def _project_record_from_row(row: sqlite3.Row | dict[str, object]) -> dict[str, object]:
    record = dict(row)
    record["rows"] = int(record.get("rows", 0) or 0)
    record["flagged_count"] = int(record.get("flagged_count", 0) or 0)
    record["review_rate"] = float(record.get("review_rate", 0.0) or 0.0)
    record["notes"] = str(record.get("notes", "") or "")
    return record


def _project_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(PROJECTS_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _upsert_project_record(project: dict[str, object]) -> None:
    payload = _project_record_from_row(project)
    placeholders = ", ".join("?" for _ in PROJECT_RECORD_FIELDS)
    with _project_db_connection() as connection:
        connection.execute(
            f"""
            INSERT OR REPLACE INTO projects ({", ".join(PROJECT_RECORD_FIELDS)})
            VALUES ({placeholders})
            """,
            tuple(payload.get(field) for field in PROJECT_RECORD_FIELDS),
        )


def _migrate_legacy_project_index(connection: sqlite3.Connection) -> None:
    if not PROJECTS_INDEX_PATH.exists():
        return

    try:
        legacy_projects = json.loads(PROJECTS_INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    if not isinstance(legacy_projects, list):
        return

    connection.execute("DELETE FROM projects")
    for project in legacy_projects:
        payload = _project_record_from_row(project)
        connection.execute(
            f"""
            INSERT OR REPLACE INTO projects ({", ".join(PROJECT_RECORD_FIELDS)})
            VALUES ({", ".join("?" for _ in PROJECT_RECORD_FIELDS)})
            """,
            tuple(payload.get(field) for field in PROJECT_RECORD_FIELDS),
        )


def _ensure_project_database() -> None:
    with _project_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                client TEXT NOT NULL,
                status TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                source_file TEXT NOT NULL,
                ledger_path TEXT NOT NULL,
                rows INTEGER NOT NULL DEFAULT 0,
                flagged_count INTEGER NOT NULL DEFAULT 0,
                review_rate REAL NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_projects_updated_at ON projects(updated_at_utc DESC)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS case_feedback (
                project_id TEXT NOT NULL,
                transaction_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Needs review',
                note TEXT NOT NULL DEFAULT '',
                updated_at_utc TEXT NOT NULL,
                PRIMARY KEY (project_id, transaction_id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_case_feedback_project ON case_feedback(project_id)"
        )
        project_count = int(connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0])
        if project_count == 0:
            _migrate_legacy_project_index(connection)


def default_model_metadata(
    model: object,
    feature_names: list[str],
    category_columns: dict[str, list[str]],
) -> dict[str, object]:
    return {
        "model_version": DEFAULT_MODEL_VERSION,
        "feature_set_version": "historical-controls-v4",
        "trained_at_utc": None,
        "training_csv": None,
        "decision_threshold": DEFAULT_DECISION_THRESHOLD,
        "priority_threshold": DEFAULT_PRIORITY_THRESHOLD,
        "watchlist_threshold": DEFAULT_WATCHLIST_THRESHOLD,
        "validation_strategy": "legacy",
        "validation_metrics": {},
        "cross_validation_metrics": {},
        "feature_names": feature_names,
        "base_features": BASE_FEATURES,
        "category_columns": category_columns,
        "model_type": type(model).__name__,
    }


def normalize_column_key(column_name: str) -> str:
    return "".join(character for character in str(column_name).lower() if character.isalnum())


def slugify_project_name(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in str(value))
    compact = "-".join(part for part in cleaned.split("-") if part)
    return compact or "audit-project"


def safe_file_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {".", "_", "-"} else "_" for character in str(value))
    return cleaned or "ledger.csv"


def deduplicate_headers(columns: list[object]) -> tuple[list[str], int]:
    seen: dict[str, int] = {}
    deduped: list[str] = []
    duplicate_count = 0

    for raw_name in columns:
        base_name = str(raw_name).strip() or "unnamed_column"
        if base_name not in seen:
            seen[base_name] = 0
            deduped.append(base_name)
            continue

        seen[base_name] += 1
        duplicate_count += 1
        deduped.append(f"{base_name}__dup{seen[base_name]}")
    return deduped, duplicate_count


def _column_display_map(dataframe: pd.DataFrame) -> dict[str, str]:
    return {
        f"{index + 1:02d}. {column}": str(column)
        for index, column in enumerate(dataframe.columns)
    }


def column_display_options(dataframe: pd.DataFrame) -> list[str]:
    return list(_column_display_map(dataframe).keys())


def suggest_column_mapping(dataframe: pd.DataFrame) -> dict[str, str | None]:
    suggested: dict[str, str | None] = {column: None for column in REQUIRED_COLUMNS}
    assigned_sources: set[str] = set()
    canonical_columns = {normalize_column_key(column): column for column in REQUIRED_COLUMNS}

    for source_column in dataframe.columns:
        stripped = str(source_column).strip()
        normalized_key = normalize_column_key(stripped)
        direct_match = canonical_columns.get(normalized_key)
        if direct_match and suggested[direct_match] is None:
            suggested[direct_match] = stripped
            assigned_sources.add(stripped)

    for source_column in dataframe.columns:
        stripped = str(source_column).strip()
        if stripped in assigned_sources:
            continue
        normalized_key = normalize_column_key(stripped)
        alias_target = COLUMN_ALIASES.get(normalized_key)
        if alias_target and alias_target in suggested and suggested[alias_target] is None:
            suggested[alias_target] = stripped
            assigned_sources.add(stripped)

    return suggested


def profile_ledger_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(columns=["Column", "Type", "Non-null", "Missing", "Sample"])

    profile_rows: list[dict[str, object]] = []
    for column in dataframe.columns:
        series = dataframe[column]
        non_null = int(series.notna().sum())
        missing = int(series.isna().sum())
        sample_values = series.dropna().astype(str).head(3).tolist()
        sample_text = " | ".join(sample_values) if sample_values else "(all missing)"
        profile_rows.append(
            {
                "Column": str(column),
                "Type": str(series.dtype),
                "Non-null": non_null,
                "Missing": missing,
                "Sample": sample_text[:120],
            }
        )
    return pd.DataFrame(profile_rows)


def _parse_localized_amount(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text:
        return float("nan")

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()

    cleaned = re.sub(r"[^\d,.\-]", "", text.replace(" ", ""))
    if cleaned.count("-") > 1:
        return float("nan")

    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        if re.search(r",\d{1,2}$", cleaned):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "." in cleaned:
        if not re.search(r"\.\d{1,2}$", cleaned):
            cleaned = cleaned.replace(".", "")

    try:
        amount = float(cleaned)
    except ValueError:
        return float("nan")
    return -amount if negative else amount


def parse_amount_series(series: pd.Series) -> pd.Series:
    return series.map(_parse_localized_amount).astype(float)


def parse_date_series(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except Exception:
        parsed = pd.to_datetime(series, errors="coerce")
    unresolved = parsed.isna()
    if unresolved.any():
        try:
            parsed_day_first = pd.to_datetime(series[unresolved], errors="coerce", format="mixed", dayfirst=True)
        except Exception:
            parsed_day_first = pd.to_datetime(series[unresolved], errors="coerce", dayfirst=True)
        parsed.loc[unresolved] = parsed_day_first
    return parsed


def normalize_input_columns(
    dataframe: pd.DataFrame,
    explicit_mapping: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    ledger = dataframe.copy()
    ledger.columns, _duplicate_count = deduplicate_headers(list(ledger.columns))
    rename_map: dict[str, str] = {}
    assigned_targets: set[str] = set()
    assigned_sources: set[str] = set()
    canonical_columns = {normalize_column_key(column): column for column in REQUIRED_COLUMNS}
    explicit_mapping = explicit_mapping or {}

    for target_column in REQUIRED_COLUMNS:
        source_column = str(explicit_mapping.get(target_column) or "").strip()
        if not source_column:
            continue
        if source_column in ledger.columns and target_column not in assigned_targets:
            rename_map[source_column] = target_column
            assigned_sources.add(source_column)
            assigned_targets.add(target_column)

    for column in ledger.columns:
        if column in assigned_sources:
            continue
        stripped = str(column).strip()
        if stripped in REQUIRED_COLUMNS and stripped not in assigned_targets:
            rename_map[column] = stripped
            assigned_sources.add(column)
            assigned_targets.add(stripped)
            continue

        normalized_key = normalize_column_key(stripped)
        alias_target = canonical_columns.get(normalized_key) or COLUMN_ALIASES.get(normalized_key)
        if alias_target and alias_target not in assigned_targets:
            rename_map[column] = alias_target
            assigned_sources.add(column)
            assigned_targets.add(alias_target)

    ledger = ledger.rename(columns=rename_map)
    ledger.columns = [str(column).strip() for column in ledger.columns]

    if "transaction_id" not in ledger.columns:
        ledger["transaction_id"] = [f"AUTO-{index + 1:06d}" for index in range(len(ledger))]

    for column, default_value in OPTIONAL_INPUT_DEFAULTS.items():
        if column not in ledger.columns:
            ledger[column] = default_value

    if "invoice_id" not in ledger.columns:
        ledger["invoice_id"] = ledger["transaction_id"].astype(str).map(lambda value: f"NO-INVOICE-{value}")

    return ledger


def prepare_ledger_for_analysis(
    dataframe: pd.DataFrame,
    explicit_mapping: dict[str, str | None] | None = None,
) -> dict[str, object]:
    prepared = normalize_input_columns(dataframe, explicit_mapping=explicit_mapping)
    if prepared.empty:
        raise ValueError("The uploaded ledger is empty.")

    for essential in ESSENTIAL_INPUT_COLUMNS:
        if essential not in prepared.columns:
            raise ValueError(
                "Ledger is missing essential columns. Auditr needs at least: "
                + ", ".join(ESSENTIAL_INPUT_COLUMNS)
            )

    working = prepared.copy()
    working["_source_row"] = np.arange(1, len(working) + 1)
    working["vendor"] = working["vendor"].astype(str).str.strip()
    working["_parsed_date"] = parse_date_series(working["date"])
    working["_parsed_amount"] = parse_amount_series(working["amount"])

    invalid_date = working["_parsed_date"].isna()
    invalid_amount = working["_parsed_amount"].isna()
    missing_vendor = working["vendor"].eq("") | working["vendor"].str.lower().isin({"nan", "none"})
    quarantined_mask = invalid_date | invalid_amount | missing_vendor

    quarantined = working.loc[quarantined_mask].copy()
    clean_rows = working.loc[~quarantined_mask].copy()
    clean_rows["date"] = clean_rows["_parsed_date"]
    clean_rows["amount"] = clean_rows["_parsed_amount"]

    quarantine_view = pd.DataFrame(
        {
            "source_row": quarantined["_source_row"].astype(int),
            "reason": np.select(
                [
                    invalid_date.loc[quarantined.index] & invalid_amount.loc[quarantined.index],
                    invalid_date.loc[quarantined.index],
                    invalid_amount.loc[quarantined.index],
                    missing_vendor.loc[quarantined.index],
                ],
                [
                    "Invalid date and amount",
                    "Invalid date",
                    "Invalid amount",
                    "Missing vendor",
                ],
                default="Row validation failure",
            ),
            "raw_date": quarantined["date"].astype(str),
            "raw_amount": quarantined["amount"].astype(str),
            "vendor": quarantined["vendor"].astype(str),
            "transaction_id": quarantined["transaction_id"].astype(str),
        }
    )

    clean_rows = clean_rows.drop(columns=["_parsed_date", "_parsed_amount", "_source_row"])
    if clean_rows.empty:
        raise ValueError(
            "No valid rows remain after parsing dates and amounts. Download the quarantined rows, fix them, and upload again."
        )

    read_meta = dataframe.attrs.get("auditr_read_meta", {})
    quality_report = {
        "total_rows": int(len(working)),
        "clean_rows": int(len(clean_rows)),
        "quarantined_rows": int(len(quarantined)),
        "quarantine_rate": float(len(quarantined) / max(len(working), 1)),
        "invalid_date_rows": int(invalid_date.sum()),
        "invalid_amount_rows": int(invalid_amount.sum()),
        "missing_vendor_rows": int(missing_vendor.sum()),
        "duplicate_header_count": int(read_meta.get("duplicate_header_count", 0)),
    }
    return {
        "prepared": clean_rows,
        "quarantine": quarantine_view,
        "quality_report": quality_report,
        "schema_profile": profile_ledger_schema(dataframe),
        "suggested_mapping": suggest_column_mapping(dataframe),
    }


def normalize_invoice_id(invoice_id: str) -> str:
    return "".join(character for character in str(invoice_id).upper() if character.isalnum())


def friendly_payment_method(value: str) -> str:
    method = str(value).strip()
    if not method:
        return PAYMENT_METHOD_DISPLAY["Other"]
    return PAYMENT_METHOD_DISPLAY.get(method, method)


def invoice_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(difflib.SequenceMatcher(None, left, right).ratio())


def compute_fuzzy_invoice_features(ledger: pd.DataFrame) -> pd.DataFrame:
    fuzzy_counts = np.zeros(len(ledger), dtype=int)
    fuzzy_scores = np.zeros(len(ledger), dtype=float)

    for _vendor, vendor_rows in ledger.groupby("vendor", sort=False):
        prior_norms: list[str] = []
        prior_amounts: list[float] = []
        for row_index in vendor_rows.index:
            current_norm = ledger.at[row_index, "invoice_id_normalized"]
            current_amount = float(ledger.at[row_index, "amount"])
            near_match_count = 0
            best_score = 0.0

            for previous_norm, previous_amount in zip(prior_norms, prior_amounts):
                amount_ratio_gap = abs(current_amount - previous_amount) / max(current_amount, previous_amount, 1.0)
                if amount_ratio_gap > 0.03:
                    continue
                score = invoice_similarity(current_norm, previous_norm)
                if 0.86 <= score < 0.999:
                    near_match_count += 1
                    best_score = max(best_score, score)

            fuzzy_counts[row_index] = near_match_count
            fuzzy_scores[row_index] = best_score
            prior_norms.append(current_norm)
            prior_amounts.append(current_amount)

    ledger["fuzzy_invoice_match_count"] = fuzzy_counts
    ledger["max_invoice_similarity"] = fuzzy_scores
    return ledger


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict[str, object]:
    artifact = load_saved_model_bundle(MODEL_PATH)

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        metadata = dict(artifact.get("metadata", {}))
    else:
        model = artifact
        metadata = {}

    feature_names = list(metadata.get("feature_names") or model.feature_names_in_)
    category_columns = metadata.get("category_columns") or derive_category_columns(feature_names)
    metadata = {**default_model_metadata(model, feature_names, category_columns), **metadata}
    metadata["feature_names"] = feature_names
    metadata["category_columns"] = category_columns
    booster = model.get_booster()
    return {
        "model": model,
        "booster": booster,
        "feature_names": feature_names,
        "category_columns": category_columns,
        "metadata": metadata,
    }


def read_ledger_csv(raw_bytes: bytes) -> pd.DataFrame:
    encoding_candidates = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
    separator_attempts = (
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
        {"sep": "|"},
    )

    for encoding in encoding_candidates:
        for read_kwargs in separator_attempts:
            try:
                dataframe = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, **read_kwargs)
            except UnicodeDecodeError:
                break
            except (pd.errors.ParserError, ValueError):
                continue
            else:
                parsed = dataframe.dropna(axis=1, how="all")
                deduped_columns, duplicate_count = deduplicate_headers(list(parsed.columns))
                parsed.columns = deduped_columns
                parsed.attrs["auditr_read_meta"] = {
                    "encoding": encoding,
                    "duplicate_header_count": duplicate_count,
                    "column_count": len(parsed.columns),
                    "row_count": int(parsed.shape[0]),
                }
                return parsed

    raise ValueError(
        "Auditr could not read this CSV. Supported uploads currently need a readable text encoding "
        "and a delimiter such as comma, semicolon, tab, or pipe."
    )


def read_ledger_csv_file(csv_path: str | Path) -> pd.DataFrame:
    return read_ledger_csv(Path(csv_path).read_bytes())


@st.cache_data(show_spinner=False)
def load_sample_dataset(sample_name: str) -> pd.DataFrame:
    return read_ledger_csv_file(SAMPLE_DATASETS[sample_name])


def ensure_app_state() -> None:
    defaults = {
        "theme_name": DEFAULT_THEME,
        "current_page": DEFAULT_PAGE,
        "authenticated": False,
        "auth_user": None,
        "authenticated_at": None,
        "clear_auth_cookie": False,
        "review_threshold": 50,
        "plain_english_mode": True,
        "analysis": None,
        "dataset_label": None,
        "active_source_key": None,
        "current_project_id": None,
        "current_project_name": None,
        "last_data_quality_report": None,
        "last_quarantined_rows": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.get("theme_name") not in THEMES:
        st.session_state["theme_name"] = DEFAULT_THEME
    _restore_persisted_state()


def current_analysis() -> dict[str, object] | None:
    return st.session_state.get("analysis")


def current_model_metadata() -> dict[str, object]:
    return load_model_bundle()["metadata"]


def current_project_name() -> str | None:
    return st.session_state.get("current_project_name")


def current_project_id() -> str | None:
    return st.session_state.get("current_project_id")


def current_page() -> str:
    return st.session_state.get("current_page", DEFAULT_PAGE)


def set_view(view_name: str) -> None:
    st.session_state["current_page"] = view_name


def current_dataset_label() -> str:
    return st.session_state.get("dataset_label") or "No dataset loaded"


def current_theme_name() -> str:
    theme_name = st.session_state.get("theme_name", DEFAULT_THEME)
    return theme_name if theme_name in THEMES else DEFAULT_THEME


def set_theme(theme_name: str) -> None:
    if theme_name in THEMES:
        st.session_state["theme_name"] = theme_name


def read_projects_index() -> list[dict[str, object]]:
    ensure_workspace_store()
    with _project_db_connection() as connection:
        rows = connection.execute(
            f"SELECT {', '.join(PROJECT_RECORD_FIELDS)} FROM projects ORDER BY updated_at_utc DESC"
        ).fetchall()
    return [_project_record_from_row(row) for row in rows]


def write_projects_index(projects: list[dict[str, object]]) -> None:
    ensure_workspace_store()
    with _project_db_connection() as connection:
        connection.execute("DELETE FROM projects")
        for project in projects:
            payload = _project_record_from_row(project)
            connection.execute(
                f"""
                INSERT OR REPLACE INTO projects ({", ".join(PROJECT_RECORD_FIELDS)})
                VALUES ({", ".join("?" for _ in PROJECT_RECORD_FIELDS)})
                """,
                tuple(payload.get(field) for field in PROJECT_RECORD_FIELDS),
            )


def list_audit_projects() -> list[dict[str, object]]:
    return read_projects_index()


def get_project_record(project_id: str) -> dict[str, object] | None:
    ensure_workspace_store()
    with _project_db_connection() as connection:
        row = connection.execute(
            f"SELECT {', '.join(PROJECT_RECORD_FIELDS)} FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
    return _project_record_from_row(row) if row else None


def project_quality_paths(project: dict[str, object]) -> tuple[Path, Path]:
    ledger_path = Path(str(project.get("ledger_path", "")))
    project_dir = ledger_path.parent
    return project_dir / "data_quality_report.json", project_dir / "quarantined_rows.csv"


def load_project_quality(project_id: str) -> tuple[dict[str, object] | None, pd.DataFrame]:
    project = get_project_record(project_id)
    if project is None:
        return None, pd.DataFrame()
    quality_path, quarantine_path = project_quality_paths(project)
    quality: dict[str, object] | None = None
    if quality_path.exists():
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            quality = None
    quarantine = pd.DataFrame()
    if quarantine_path.exists():
        try:
            quarantine = pd.read_csv(quarantine_path)
        except Exception:
            quarantine = pd.DataFrame()
    return quality, quarantine


def _file_mtime_ns(file_path: Path) -> int:
    file_stat = file_path.stat()
    return int(getattr(file_stat, "st_mtime_ns", int(file_stat.st_mtime * 1_000_000_000)))


def _project_analysis_cache_path(ledger_path: Path) -> Path:
    return ledger_path.parent / PROJECT_ANALYSIS_CACHE_FILE


def _project_analysis_cache_key(ledger_path: Path) -> str | None:
    if not ledger_path.exists():
        return None

    try:
        ledger_size = int(ledger_path.stat().st_size)
        ledger_mtime_ns = _file_mtime_ns(ledger_path)
    except OSError:
        return None

    model_size = 0
    model_mtime_ns = 0
    if MODEL_PATH.exists():
        try:
            model_size = int(MODEL_PATH.stat().st_size)
            model_mtime_ns = _file_mtime_ns(MODEL_PATH)
        except OSError:
            model_size = 0
            model_mtime_ns = 0

    signature_payload = {
        "cache_version": PROJECT_ANALYSIS_CACHE_VERSION,
        "ledger_size": ledger_size,
        "ledger_mtime_ns": ledger_mtime_ns,
        "model_size": model_size,
        "model_mtime_ns": model_mtime_ns,
    }
    serialized_payload = json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized_payload).hexdigest()


def _looks_like_analysis_bundle(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    required_keys = {"scored", "flagged", "features", "contributions", "base_values", "model_metadata"}
    return required_keys.issubset(set(payload.keys()))


def read_project_analysis_cache(ledger_path: Path) -> dict[str, object] | None:
    cache_key = _project_analysis_cache_key(ledger_path)
    if not cache_key:
        return None

    cache_path = _project_analysis_cache_path(ledger_path)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("rb") as cache_file:
            payload = pickle.load(cache_file)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if int(payload.get("cache_version", -1)) != PROJECT_ANALYSIS_CACHE_VERSION:
        return None
    if payload.get("cache_key") != cache_key:
        return None

    analysis = payload.get("analysis")
    if not _looks_like_analysis_bundle(analysis):
        return None
    return analysis


def write_project_analysis_cache(ledger_path: Path, analysis: dict[str, object]) -> None:
    cache_key = _project_analysis_cache_key(ledger_path)
    if not cache_key or not _looks_like_analysis_bundle(analysis):
        return

    cache_path = _project_analysis_cache_path(ledger_path)
    cache_payload = {
        "cache_version": PROJECT_ANALYSIS_CACHE_VERSION,
        "cache_key": cache_key,
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis": analysis,
    }
    try:
        with cache_path.open("wb") as cache_file:
            pickle.dump(cache_payload, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return


def case_feedback_map(project_id: str) -> dict[str, dict[str, str]]:
    ensure_workspace_store()
    with _project_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT transaction_id, status, note, updated_at_utc
            FROM case_feedback
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchall()
    return {
        str(row["transaction_id"]): {
            "status": str(row["status"]),
            "note": str(row["note"] or ""),
            "updated_at_utc": str(row["updated_at_utc"]),
        }
        for row in rows
    }


def get_case_feedback(project_id: str, transaction_id: str) -> dict[str, str] | None:
    if not project_id or not transaction_id:
        return None
    feedback = case_feedback_map(project_id).get(str(transaction_id))
    return feedback


def upsert_case_feedback(project_id: str, transaction_id: str, status: str, note: str = "") -> None:
    if status not in FEEDBACK_STATUS_OPTIONS:
        raise ValueError("Invalid feedback status.")
    ensure_workspace_store()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _project_db_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO case_feedback (project_id, transaction_id, status, note, updated_at_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            (project_id, transaction_id, status, note.strip(), timestamp),
        )


def apply_case_feedback_to_analysis(
    analysis: dict[str, object] | None,
    project_id: str | None,
) -> dict[str, object] | None:
    if analysis is None or not project_id:
        return analysis

    feedback = case_feedback_map(project_id)
    if not feedback:
        scored = analysis["scored"].copy()
        scored["reviewer_status"] = "Needs review"
        scored["reviewer_note"] = ""
        scored["review_priority_multiplier"] = 1.0
        scored = scored.sort_values("blended_priority_score", ascending=False).reset_index(drop=True)
        flagged = scored[scored["fraud_prediction"] == 1].reset_index(drop=True)
        return {**analysis, "scored": scored, "flagged": flagged}

    scored = analysis["scored"].copy()
    scored["reviewer_status"] = scored["transaction_id"].astype(str).map(
        lambda transaction_id: feedback.get(transaction_id, {}).get("status", "Needs review")
    )
    scored["reviewer_note"] = scored["transaction_id"].astype(str).map(
        lambda transaction_id: feedback.get(transaction_id, {}).get("note", "")
    )
    multiplier_map = {"Escalated": 1.18, "Needs review": 1.0, "Cleared": 0.55}
    scored["review_priority_multiplier"] = scored["reviewer_status"].map(multiplier_map).fillna(1.0)
    scored["blended_priority_score"] = (
        scored["blended_risk_score"] * scored["review_priority_multiplier"]
    ).clip(0.0, 1.0)
    scored["review_status"] = np.select(
        [
            scored["reviewer_status"] == "Cleared",
            scored["reviewer_status"] == "Escalated",
            scored["fraud_prediction"] == 1,
        ],
        [
            "Cleared by reviewer",
            "Escalated by reviewer",
            "Needs review",
        ],
        default="Looks routine",
    )
    scored = scored.sort_values("blended_priority_score", ascending=False).reset_index(drop=True)
    flagged = scored[scored["fraud_prediction"] == 1].reset_index(drop=True)
    return {**analysis, "scored": scored, "flagged": flagged}


def sync_theme_widget(widget_key: str) -> None:
    st.session_state[widget_key] = current_theme_name()


def apply_theme_from_widget(widget_key: str) -> None:
    selected_theme = st.session_state.get(widget_key)
    if selected_theme in THEMES:
        set_theme(selected_theme)


def get_theme_palette() -> dict[str, str]:
    return THEMES[current_theme_name()]


def get_auth_config() -> dict[str, object]:
    user = os.getenv("AUDITR_AUTH_USER") or DEFAULT_AUTH_USER
    secret = os.getenv("AUDITR_TOTP_SECRET") or DEFAULT_AUTH_SECRET
    demo_mode = "AUDITR_AUTH_USER" not in os.environ or "AUDITR_TOTP_SECRET" not in os.environ
    return {
        "username": user,
        "secret": secret,
        "demo_mode": demo_mode,
        "window_seconds": 30,
    }


def _cookie_value(cookie_name: str) -> str | None:
    cookie_jar = getattr(st.context, "cookies", None)
    if cookie_jar is None:
        return None

    raw_value = cookie_jar.get(cookie_name)
    if raw_value is None:
        return None
    if hasattr(raw_value, "value"):
        raw_value = raw_value.value
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8", errors="ignore")
    return str(raw_value)


def _session_signing_key() -> bytes:
    auth = get_auth_config()
    seed = f"auditr-session|{auth['username']}|{auth['secret']}".encode("utf-8")
    return hashlib.sha256(seed).digest()


def _build_auth_cookie_value(username: str, issued_at: float) -> str:
    issued_at_int = int(issued_at)
    payload = f"{username}|{issued_at_int}"
    signature = hmac.new(_session_signing_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    token = f"{payload}|{signature}"
    return base64.urlsafe_b64encode(token.encode("utf-8")).decode("utf-8").rstrip("=")


def _read_auth_cookie_value(token: str | None) -> dict[str, object] | None:
    if not token:
        return None

    try:
        padding = "=" * ((4 - len(token) % 4) % 4)
        decoded = base64.urlsafe_b64decode(f"{token}{padding}".encode("utf-8")).decode("utf-8")
        username, issued_at_text, signature = decoded.split("|", 2)
        issued_at = int(issued_at_text)
    except (ValueError, TypeError, base64.binascii.Error):
        return None

    if username != get_auth_config()["username"]:
        return None

    payload = f"{username}|{issued_at}"
    expected_signature = hmac.new(
        _session_signing_key(),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        return None

    now = int(time.time())
    if issued_at > now or now - issued_at > AUTH_SESSION_MAX_AGE:
        return None

    return {
        "username": username,
        "issued_at": float(issued_at),
    }


def _restore_persisted_state() -> None:
    if not st.session_state.get("_persisted_state_bootstrapped"):
        theme_cookie = _cookie_value(THEME_COOKIE_NAME)
        if theme_cookie in THEMES:
            st.session_state["theme_name"] = theme_cookie

        if not st.session_state.get("clear_auth_cookie"):
            auth_cookie = _cookie_value(AUTH_COOKIE_NAME)
            auth_session = _read_auth_cookie_value(auth_cookie)
            if auth_session:
                st.session_state["authenticated"] = True
                st.session_state["auth_user"] = auth_session["username"]
                st.session_state["authenticated_at"] = auth_session["issued_at"]
            elif auth_cookie:
                st.session_state["clear_auth_cookie"] = True

        if (
            st.session_state.get("authenticated")
            and st.session_state.get("analysis") is None
            and not st.session_state.get("current_project_id")
        ):
            persisted_project_id = _cookie_value(PROJECT_COOKIE_NAME)
            if persisted_project_id:
                try:
                    open_audit_project(persisted_project_id)
                except Exception:
                    clear_active_project_context()

        st.session_state["_persisted_state_bootstrapped"] = True


def sync_client_persistence() -> None:
    auth_token = ""
    if st.session_state.get("authenticated"):
        auth_token = _build_auth_cookie_value(
            st.session_state.get("auth_user") or get_auth_config()["username"],
            st.session_state.get("authenticated_at") or time.time(),
        )

    bridge_payload = {
        "themeCookieName": THEME_COOKIE_NAME,
        "themeValue": current_theme_name(),
        "themeMaxAge": THEME_COOKIE_MAX_AGE,
        "authCookieName": AUTH_COOKIE_NAME,
        "authValue": auth_token,
        "authMaxAge": AUTH_SESSION_MAX_AGE,
        "projectCookieName": PROJECT_COOKIE_NAME,
        "projectValue": current_project_id() if st.session_state.get("authenticated") else "",
        "projectMaxAge": PROJECT_COOKIE_MAX_AGE,
        "clearAuth": bool(st.session_state.get("clear_auth_cookie")),
    }

    components.html(
        f"""
        <script>
        const payload = {json.dumps(bridge_payload)};
        const rootDoc = window.parent?.document || document;
        const rootWindow = window.parent || window;

        function writeCookie(name, value, maxAge) {{
            rootDoc.cookie = `${{name}}=${{encodeURIComponent(value)}}; path=/; max-age=${{maxAge}}; SameSite=Lax`;
        }}

        function deleteCookie(name) {{
            rootDoc.cookie = `${{name}}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax`;
        }}

        try {{
            writeCookie(payload.themeCookieName, payload.themeValue, payload.themeMaxAge);
            rootWindow.localStorage?.setItem(payload.themeCookieName, payload.themeValue);

            if (payload.clearAuth) {{
                deleteCookie(payload.authCookieName);
                rootWindow.localStorage?.removeItem(payload.authCookieName);
            }} else if (payload.authValue) {{
                writeCookie(payload.authCookieName, payload.authValue, payload.authMaxAge);
                rootWindow.localStorage?.setItem(payload.authCookieName, payload.authValue);
            }}

            if (payload.projectValue) {{
                writeCookie(payload.projectCookieName, payload.projectValue, payload.projectMaxAge);
                rootWindow.localStorage?.setItem(payload.projectCookieName, payload.projectValue);
            }} else {{
                deleteCookie(payload.projectCookieName);
                rootWindow.localStorage?.removeItem(payload.projectCookieName);
            }}
        }} catch (error) {{
            console.warn("Auditr persistence sync failed", error);
        }}
        </script>
        """,
        height=0,
        width=0,
    )

    if st.session_state.get("clear_auth_cookie"):
        st.session_state["clear_auth_cookie"] = False


def format_manual_key(secret: str) -> str:
    normalized = _normalize_totp_secret(secret).replace("=", "")
    return " ".join(
        normalized[index : index + 4]
        for index in range(0, len(normalized), 4)
    )


def _normalize_totp_secret(secret: str) -> str:
    cleaned = secret.strip().replace(" ", "").upper()
    padding = "=" * ((8 - len(cleaned) % 8) % 8)
    return cleaned + padding


def generate_totp(secret: str, timestamp: float | None = None, digits: int = 6, step: int = 30) -> str:
    key = base64.b32decode(_normalize_totp_secret(secret), casefold=True)
    counter = int((timestamp or time.time()) // step)
    message = struct.pack(">Q", counter)
    digest = hmac.new(key, message, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    binary = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return f"{binary % (10 ** digits):0{digits}d}"


def verify_totp(candidate: str, secret: str, window: int = 1) -> bool:
    token = candidate.strip()
    if not token.isdigit() or len(token) != 6:
        return False

    now = time.time()
    for step_offset in range(-window, window + 1):
        comparison_time = now + (step_offset * 30)
        if generate_totp(secret, timestamp=comparison_time) == token:
            return True
    return False


def current_user_label() -> str:
    return st.session_state.get("auth_user") or get_auth_config()["username"]


def current_user_initials() -> str:
    user = current_user_label().strip() or "AU"
    parts = [part[0].upper() for part in user.replace("_", " ").split()[:2] if part]
    return "".join(parts) or user[:2].upper()


def session_age_label() -> str:
    authenticated_at = st.session_state.get("authenticated_at")
    if not authenticated_at:
        return "Just now"
    elapsed = max(int(time.time() - authenticated_at), 0)
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


def logout_user() -> None:
    st.session_state["authenticated"] = False
    st.session_state["auth_user"] = None
    st.session_state["authenticated_at"] = None
    st.session_state["current_page"] = DEFAULT_PAGE
    st.session_state["clear_auth_cookie"] = True


def render_auth_gate() -> None:
    if st.session_state.get("authenticated"):
        return

    auth = get_auth_config()
    demo_code = generate_totp(auth["secret"])
    manual_key = format_manual_key(auth["secret"])

    left, center, right = st.columns([0.28, 0.44, 0.28])
    with center:
        st.markdown(
            hero_panel(
                "Secure entry",
                "Enter your 6-digit TOTP code",
                (
                    f"You are signing in as `{auth['username']}`. "
                    "Use the code from your authenticator app. "
                    "For this local demo, the manual key and current rotating code are shown below."
                ),
            ),
            unsafe_allow_html=True,
        )

        with st.form("auditr_auth"):
            st.caption(f"Fixed account: `{auth['username']}`")
            code = st.text_input("6-digit TOTP code", max_chars=6, placeholder="123456")
            submitted = st.form_submit_button("Enter workspace", width="stretch")

        if auth["demo_mode"]:
            st.markdown(
                notice_panel(
                    "Demo authentication is active",
                    (
                        f"Manual key: `{manual_key}`. "
                        f"The code rotates every {auth['window_seconds']} seconds. "
                        f"Current demo code: `{demo_code}`."
                    ),
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                notice_panel(
                    "What the manual key means",
                    (
                        "The manual key is the secret you paste into Google Authenticator, Microsoft Authenticator, "
                        "or any TOTP app if you are not using a QR code."
                    ),
                ),
                unsafe_allow_html=True,
            )

        if submitted:
            if not verify_totp(code, auth["secret"]):
                st.error("The TOTP code is invalid or expired. Wait for the next 30-second window and try again.")
            else:
                st.session_state["authenticated"] = True
                st.session_state["auth_user"] = auth["username"]
                st.session_state["authenticated_at"] = time.time()
                st.session_state["clear_auth_cookie"] = False
                st.rerun()

    st.stop()


def inject_app_styles() -> None:
    palette = get_theme_palette()
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {{
            --bg-top: {palette["bg_top"]};
            --bg-bottom: {palette["bg_bottom"]};
            --surface: {palette["surface"]};
            --surface-strong: {palette["surface_strong"]};
            --surface-alt: {palette["surface_alt"]};
            --text: {palette["text"]};
            --muted: {palette["muted"]};
            --line: {palette["line"]};
            --accent: {palette["accent"]};
            --accent-soft: {palette["accent_soft"]};
            --amber: {palette["amber"]};
            --danger: {palette["danger"]};
            --success: {palette["success"]};
            --shadow: {palette["shadow"]};
        }}

        [data-testid="stSidebar"], [data-testid="collapsedControl"] {{
            display: none !important;
        }}

        #MainMenu,
        [data-testid="stToolbar"],
        [data-testid="stStatusWidget"],
        [data-testid="stHeaderActionElements"],
        div[data-testid="stDecoration"] {{
            display: none !important;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 78% 0%, color-mix(in srgb, var(--accent) 5%, transparent), transparent 22%),
                linear-gradient(180deg, color-mix(in srgb, var(--bg-top) 94%, white) 0%, var(--bg-bottom) 100%);
            color: var(--text);
        }}

        .stApp, .stMarkdown, label, p, span, div {{
            font-family: "Manrope", sans-serif;
        }}

        code, pre {{
            font-family: "IBM Plex Mono", monospace !important;
        }}

        header[data-testid="stHeader"] {{
            display: none !important;
        }}

        .stApp .block-container {{
            max-width: 1320px;
            padding-top: 1.35rem;
            padding-bottom: 2.4rem;
        }}

        .stApp [data-testid="stHorizontalBlock"] {{
            align-items: flex-start;
        }}

        .stApp [data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {{
            gap: 0.92rem !important;
        }}

        .stApp .element-container,
        .stApp div[data-testid="stElementContainer"] {{
            margin-bottom: 0.16rem !important;
        }}

        .hero-panel, .card-panel, .notice-panel, .guide-card, .field-card {{
            background: linear-gradient(180deg, var(--surface), color-mix(in srgb, var(--surface-strong) 78%, transparent));
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: 0 16px 34px var(--shadow);
            backdrop-filter: blur(12px);
        }}

        .shell-brand {{
            padding: 0.4rem 0.1rem 0.7rem 0.1rem;
            margin-left: 0.55rem;
            min-height: 0;
            background: transparent;
            border: none;
            box-shadow: none;
            backdrop-filter: none;
        }}

        .shell-row {{
            display: flex;
            align-items: baseline;
            gap: 0.85rem;
            flex-wrap: wrap;
        }}

        .shell-product {{
            display: inline-block;
            padding: 0;
            color: var(--text);
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: 0.04em;
            line-height: 1;
            text-transform: uppercase;
        }}

        .shell-page {{
            color: var(--accent);
            font-size: 0.9rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .hero-panel {{
            padding: 1.2rem 1.35rem;
            margin-bottom: 0.88rem;
        }}

        .hero-eyebrow, .section-eyebrow {{
            margin: 0;
            color: var(--accent);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }}

        .hero-title {{
            margin: 0.55rem 0 0.62rem 0;
            color: var(--text);
            font-size: 2.08rem;
            font-weight: 800;
            line-height: 1.1;
            max-width: 18ch;
        }}

        .hero-copy, .section-copy {{
            margin: 0;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.68;
            max-width: 68ch;
        }}

        .card-panel {{
            min-height: 0;
            height: 100%;
            display: flex;
            flex-direction: column;
            padding: 0.98rem 1.04rem;
        }}

        .card-panel.danger {{
            background: linear-gradient(180deg, color-mix(in srgb, var(--danger) 8%, white), var(--surface));
        }}

        .card-panel.accent {{
            background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 9%, white), var(--surface));
        }}

        .card-panel.amber {{
            background: linear-gradient(180deg, color-mix(in srgb, var(--amber) 10%, white), var(--surface));
        }}

        .card-label {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .card-value {{
            margin-top: 0.56rem;
            color: var(--text);
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.05;
        }}

        .card-value.compact {{
            font-size: 1.52rem;
            line-height: 1.08;
            letter-spacing: -0.02em;
        }}

        .card-caption {{
            margin-top: auto;
            padding-top: 0.68rem;
            color: var(--muted);
            font-size: 0.87rem;
            line-height: 1.5;
        }}

        .notice-panel {{
            padding: 0.94rem 1.02rem;
            margin-top: 0.62rem;
        }}

        .notice-title {{
            margin: 0 0 0.3rem 0;
            font-size: 0.92rem;
            font-weight: 800;
            color: var(--text);
        }}

        .notice-copy {{
            margin: 0;
            color: var(--muted);
            line-height: 1.58;
            font-size: 0.92rem;
        }}

        .meta-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: color-mix(in srgb, var(--surface-strong) 86%, transparent);
            color: var(--text);
            font-size: 0.84rem;
            font-weight: 700;
        }}

        .glossary-card {{
            background: linear-gradient(180deg, var(--surface), color-mix(in srgb, var(--surface-alt) 82%, transparent));
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.94rem 1rem;
            height: 100%;
            box-shadow: 0 10px 24px var(--shadow);
        }}

        .glossary-term {{
            margin: 0 0 0.3rem 0;
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 800;
        }}

        .glossary-body {{
            margin: 0;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.58;
        }}

        .guide-card, .field-card {{
            padding: 1rem 1.04rem;
            height: 100%;
        }}

        .section-gap {{
            height: 0.4rem;
        }}

        .project-browser-card {{
            background: linear-gradient(180deg, var(--surface), color-mix(in srgb, var(--surface-strong) 86%, transparent));
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 0.95rem 1rem 0.98rem 1rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 14px 28px var(--shadow);
            min-height: 188px;
        }}

        .project-browser-card.selected {{
            border-color: color-mix(in srgb, var(--accent) 26%, var(--line));
            box-shadow: 0 18px 32px color-mix(in srgb, var(--accent) 10%, transparent);
        }}

        .project-browser-rule {{
            height: 4px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), color-mix(in srgb, var(--accent) 46%, white));
            margin-bottom: 0.9rem;
        }}

        .project-browser-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.9rem;
        }}

        .project-browser-title {{
            color: var(--text);
            font-size: 1.1rem;
            font-weight: 800;
            line-height: 1.24;
            margin: 0;
        }}

        .project-browser-client {{
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: 0.22rem;
        }}

        .project-status-pill {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.36rem 0.72rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 800;
            white-space: nowrap;
            border: 1px solid color-mix(in srgb, var(--accent) 14%, var(--line));
            color: var(--accent);
            background: color-mix(in srgb, var(--accent-soft) 58%, white);
        }}

        .project-status-pill.active {{
            color: var(--success);
            border-color: color-mix(in srgb, var(--success) 16%, var(--line));
            background: color-mix(in srgb, var(--success) 12%, white);
        }}

        .project-status-pill.closed {{
            color: var(--muted);
            background: color-mix(in srgb, var(--surface-alt) 86%, white);
        }}

        .project-browser-metrics {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            margin-top: 0.95rem;
        }}

        .project-browser-metric {{
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.62rem 0.72rem;
            background: color-mix(in srgb, var(--surface-strong) 84%, transparent);
        }}

        .project-browser-metric-label {{
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .project-browser-metric-value {{
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
            margin-top: 0.24rem;
        }}

        .project-browser-footer {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
            margin-top: 1.05rem;
            padding-top: 0.95rem;
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.84rem;
        }}

        .drawer-current {{
            display: flex;
            align-items: center;
            padding: 0.76rem 0.9rem;
            border-radius: 14px;
            background: color-mix(in srgb, var(--accent-soft) 58%, var(--surface-strong));
            border: 1px solid color-mix(in srgb, var(--accent) 22%, var(--line));
            color: var(--text);
            font-size: 0.92rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }}

        .guide-title, .field-title {{
            margin: 0;
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
        }}

        .guide-copy, .field-copy {{
            margin: 0.3rem 0 0 0;
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.55;
        }}

        .guide-step {{
            display: flex;
            gap: 0.8rem;
            padding: 0.8rem 0 0 0;
            margin-top: 0.8rem;
            border-top: 1px solid var(--line);
        }}

        .guide-step-index {{
            width: 1.9rem;
            height: 1.9rem;
            border-radius: 999px;
            background: color-mix(in srgb, var(--accent) 14%, white);
            color: var(--accent);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.84rem;
            font-weight: 800;
            flex: 0 0 auto;
        }}

        .guide-step-title {{
            color: var(--text);
            font-size: 0.92rem;
            font-weight: 800;
        }}

        .guide-step-copy {{
            margin-top: 0.18rem;
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.5;
        }}

        .field-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.95rem;
        }}

        .field-chip {{
            padding: 0.75rem 0.82rem;
            border-radius: 16px;
            background: color-mix(in srgb, var(--surface-strong) 94%, transparent);
            border: 1px solid color-mix(in srgb, var(--accent) 18%, var(--line));
        }}

        .field-chip-title {{
            display: block;
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .field-chip-code {{
            display: block;
            margin-top: 0.28rem;
            color: var(--text);
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.82rem;
            line-height: 1.35;
        }}

        .summary-panel {{
            background: linear-gradient(180deg, var(--surface), color-mix(in srgb, var(--surface-alt) 84%, transparent));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.02rem 1.08rem;
            box-shadow: 0 14px 30px var(--shadow);
        }}

        .summary-heading {{
            margin: 0.3rem 0 0.45rem 0;
            color: var(--text);
            font-size: 1.15rem;
            font-weight: 800;
        }}

        .summary-text {{
            margin: 0;
            color: var(--muted);
            line-height: 1.65;
            font-size: 0.95rem;
        }}

        [data-baseweb="base-input"] > div,
        [data-baseweb="select"] > div,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stNumberInputContainer"] > div {{
            background: var(--surface-strong) !important;
            border: 1px solid var(--line) !important;
            border-radius: 18px !important;
            color: var(--text) !important;
            box-shadow: 0 8px 18px color-mix(in srgb, var(--shadow) 55%, transparent);
        }}

        [data-testid="stWidgetLabel"] *,
        [data-testid="stWidgetLabel"],
        label,
        .stCaptionContainer *,
        .stTextInput label,
        .stSelectbox label {{
            color: var(--text) !important;
        }}

        [data-baseweb="base-input"] input,
        [data-baseweb="select"] input,
        [data-baseweb="select"] span,
        [data-baseweb="select"] svg,
        [data-testid="stFileUploaderDropzone"] * {{
            color: var(--text) !important;
            fill: var(--text) !important;
        }}

        [data-testid="stFileUploader"] {{
            color: var(--text) !important;
        }}

        [data-testid="stFileUploader"] *,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] svg {{
            color: var(--text) !important;
            fill: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
        }}

        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {{
            color: var(--text) !important;
            background: transparent !important;
        }}

        [data-testid="stFileUploader"] button[kind="icon"] {{
            color: var(--text) !important;
            border-color: color-mix(in srgb, var(--accent) 20%, var(--line)) !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            min-height: 150px !important;
            padding: 1.1rem 1.18rem !important;
            background: linear-gradient(180deg, color-mix(in srgb, var(--surface-strong) 98%, transparent), color-mix(in srgb, var(--accent-soft) 34%, white)) !important;
            border: 1px dashed color-mix(in srgb, var(--accent) 35%, var(--line)) !important;
            box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 8%, transparent) !important;
        }}

        [data-testid="stFileUploaderDropzone"] section {{
            padding: 0 !important;
            align-items: center !important;
        }}

        [data-testid="stFileUploaderDropzone"] button {{
            background: var(--surface-strong) !important;
            color: var(--text) !important;
            border: 1px solid color-mix(in srgb, var(--accent) 24%, var(--line)) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            font-weight: 800 !important;
        }}

        [data-testid="stFileUploaderDropzone"] small {{
            color: var(--muted) !important;
        }}

        [data-testid="stFileUploaderDropzone"] svg {{
            display: none !important;
        }}

        input,
        textarea {{
            background: var(--surface-strong) !important;
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
            caret-color: var(--text) !important;
        }}

        input::placeholder,
        textarea::placeholder {{
            color: var(--muted) !important;
            opacity: 1 !important;
        }}

        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {{
            background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 92%, white), var(--accent));
            color: white;
            border: 1px solid color-mix(in srgb, var(--accent) 64%, black);
            border-radius: 14px;
            box-shadow: 0 10px 22px color-mix(in srgb, var(--accent) 18%, transparent);
            font-weight: 800;
            min-height: 46px;
        }}

        .stButton > button p,
        .stDownloadButton > button p,
        .stFormSubmitButton > button p {{
            color: inherit !important;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {{
            border-color: color-mix(in srgb, var(--accent) 55%, black);
            color: white;
            transform: translateY(-1px);
        }}

        div[role="radiogroup"] {{
            gap: 0.35rem !important;
        }}

        div[role="radiogroup"] button {{
            background: color-mix(in srgb, var(--surface-strong) 96%, white) !important;
            color: var(--muted) !important;
            border: 1px solid var(--line) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            min-height: 42px !important;
        }}

        div[role="radiogroup"] button p,
        div[role="radiogroup"] button span {{
            color: inherit !important;
        }}

        div[role="radiogroup"] button[aria-checked="true"] {{
            background: color-mix(in srgb, var(--accent-soft) 66%, white) !important;
            color: var(--accent) !important;
            border-color: color-mix(in srgb, var(--accent) 24%, var(--line)) !important;
        }}

        div[role="radiogroup"] button:hover {{
            background: color-mix(in srgb, var(--accent-soft) 40%, white) !important;
            color: var(--text) !important;
            transform: none !important;
        }}

        .stPopover > button,
        button[data-testid="stPopoverButton"] {{
            background: rgba(255, 255, 255, 0.96) !important;
            background-color: rgba(255, 255, 255, 0.96) !important;
            background-image: none !important;
            color: var(--accent) !important;
            -webkit-text-fill-color: var(--accent) !important;
            border: 1px solid color-mix(in srgb, var(--accent) 20%, var(--line)) !important;
            border-radius: 14px;
            box-shadow: 0 8px 18px var(--shadow) !important;
            font-weight: 800;
            min-height: 42px;
            min-width: 42px;
            padding: 0.32rem 0.72rem !important;
        }}

        .stPopover > button p,
        .stPopover > button span,
        .stPopover > button svg,
        button[data-testid="stPopoverButton"] p,
        button[data-testid="stPopoverButton"] span,
        button[data-testid="stPopoverButton"] svg {{
            color: var(--accent) !important;
            fill: var(--accent) !important;
        }}

        .stPopover > button:hover,
        .stPopover > button:focus,
        .stPopover > button:active,
        button[data-testid="stPopoverButton"]:hover,
        button[data-testid="stPopoverButton"]:focus,
        button[data-testid="stPopoverButton"]:active {{
            border-color: color-mix(in srgb, var(--accent) 28%, var(--line));
            color: var(--accent) !important;
            -webkit-text-fill-color: var(--accent) !important;
            background: color-mix(in srgb, var(--accent-soft) 54%, white) !important;
            background-color: color-mix(in srgb, var(--accent-soft) 54%, white) !important;
            background-image: none !important;
        }}

        button[data-testid="stPopoverButton"] svg {{
            display: none !important;
        }}

        button[data-testid="stPopoverButton"] span {{
            display: none !important;
        }}

        button[data-testid="stPopoverButton"]::before {{
            content: "\\2630";
            display: inline-block;
            color: var(--accent);
            font-size: 1rem;
            font-weight: 800;
            line-height: 1;
        }}

        [data-testid="stPopoverBody"] {{
            min-width: 270px !important;
            border-radius: 18px !important;
            border: 1px solid var(--line) !important;
            box-shadow: 0 18px 32px var(--shadow) !important;
            background: color-mix(in srgb, var(--surface-strong) 97%, transparent) !important;
            background-color: color-mix(in srgb, var(--surface-strong) 97%, transparent) !important;
            background-image: none !important;
            padding: 0.38rem !important;
        }}

        [data-testid="stPopoverBody"] > div {{
            background: color-mix(in srgb, var(--surface-strong) 97%, transparent) !important;
            background-color: color-mix(in srgb, var(--surface-strong) 97%, transparent) !important;
            background-image: none !important;
            border-radius: 16px !important;
            color: var(--text) !important;
        }}

        [data-testid="stPopoverBody"] [data-testid="stVerticalBlock"],
        [data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] > div,
        [data-testid="stPopoverBody"] [data-testid="stVerticalBlockBorderWrapper"] {{
            background: transparent !important;
            background-color: transparent !important;
            background-image: none !important;
        }}

        [data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {{
            gap: 0.3rem !important;
        }}

        [data-testid="stPopoverBody"] .stButton > button {{
            width: 100%;
            justify-content: flex-start;
            background: transparent !important;
            color: var(--text) !important;
            border: 1px solid transparent !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            padding: 0.62rem 0.82rem !important;
            min-height: 0 !important;
        }}

        [data-testid="stPopoverBody"] .stButton > button:hover {{
            background: color-mix(in srgb, var(--accent-soft) 48%, var(--surface-strong)) !important;
            border-color: color-mix(in srgb, var(--accent) 20%, var(--line)) !important;
            color: var(--text) !important;
            transform: none;
        }}

        [data-testid="stPopoverBody"] .stButton > button p {{
            color: var(--text) !important;
        }}

        [data-testid="stPopoverBody"] .stCaptionContainer * {{
            color: var(--muted) !important;
        }}

        [data-testid="stPopoverBody"] .stCaptionContainer {{
            margin: 0.04rem 0 0.16rem 0 !important;
        }}

        [data-testid="stPopoverBody"] hr {{
            margin: 0.4rem 0 0.36rem 0 !important;
        }}

        [data-testid="stPopoverBody"] .stMarkdown,
        [data-testid="stPopoverBody"] .stMarkdown *,
        [data-testid="stPopoverBody"] .stText,
        [data-testid="stPopoverBody"] p,
        [data-testid="stPopoverBody"] span,
        [data-testid="stPopoverBody"] div {{
            color: var(--text) !important;
        }}

        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"] {{
            background: linear-gradient(180deg, var(--surface), color-mix(in srgb, var(--surface-alt) 82%, transparent));
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: 0 12px 26px var(--shadow);
            padding: 0.4rem;
        }}

        [data-testid="stDataFrame"] {{
            overflow: hidden;
        }}

        .stAlert {{
            border-radius: 18px;
            border: 1px solid var(--line);
        }}

        h3 {{
            color: var(--text) !important;
            font-size: 1.28rem !important;
            font-weight: 800 !important;
            margin: 0.28rem 0 0.2rem 0 !important;
        }}

        .stMarkdown a[href^="#"] {{
            display: none !important;
        }}

        @media (max-width: 900px) {{
            .hero-title {{
                font-size: 1.82rem;
            }}

            .field-grid {{
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }}
        }}

        @media (max-width: 640px) {{
            .hero-title {{
                font-size: 1.42rem;
            }}

            .hero-copy,
            .section-copy,
            .notice-copy,
            .glossary-body {{
                font-size: 0.92rem;
            }}

            .shell-row {{
                gap: 0.45rem;
            }}

            .shell-brand {{
                padding: 0.1rem 0 0.35rem 0;
                margin-top: -3.15rem;
                margin-left: 4.25rem;
            }}

            .shell-product {{
                font-size: 1.22rem;
            }}

            .shell-page {{
                font-size: 0.78rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_panel(eyebrow: str, title: str, body: str) -> str:
    return f"""
    <section class="hero-panel">
        <p class="hero-eyebrow">{eyebrow}</p>
        <h1 class="hero-title">{title}</h1>
        <p class="hero-copy">{body}</p>
    </section>
    """


def shell_brand(page_label: str) -> str:
    return f"""
    <div class="shell-brand">
        <div class="shell-row">
            <span class="shell-product">Auditr</span>
            <span class="shell-page">{page_label}</span>
        </div>
    </div>
    """


def metric_card(title: str, value: str, caption: str, tone: str = "accent") -> str:
    value_text = str(value)
    value_class = "card-value compact" if len(value_text) > 16 else "card-value"
    return f"""
    <div class="card-panel {tone}">
        <div class="card-label">{title}</div>
        <div class="{value_class}">{value_text}</div>
        <div class="card-caption">{caption}</div>
    </div>
    """


def notice_panel(title: str, body: str) -> str:
    return f"""
    <div class="notice-panel">
        <div class="notice-title">{title}</div>
        <div class="notice-copy">{body}</div>
    </div>
    """


def section_gap() -> str:
    return '<div class="section-gap"></div>'


def project_browser_card(project: dict[str, object], *, active: bool = False, selected: bool = False) -> str:
    project_name = html.escape(str(project.get("name", "")))
    client_name = html.escape(str(project.get("client", "")))
    source_file = html.escape(str(project.get("source_file", "")))
    updated_at = html.escape(str(project.get("updated_at_utc", ""))[:16].replace("T", " "))
    status_text = str(project.get("status", ""))
    review_rate = float(project.get("review_rate", 0.0))
    rows = int(project.get("rows", 0))
    flagged = int(project.get("flagged_count", 0))

    if active:
        pill_text = "Active in workspace"
        pill_class = "project-status-pill active"
    elif status_text == "Closed":
        pill_text = status_text
        pill_class = "project-status-pill closed"
    else:
        pill_text = status_text
        pill_class = "project-status-pill"

    selected_class = " selected" if selected else ""
    return f"""
    <div class="project-browser-card{selected_class}">
        <div class="project-browser-rule"></div>
        <div class="project-browser-header">
            <div>
                <div class="project-browser-title">{project_name}</div>
                <div class="project-browser-client">{client_name}</div>
            </div>
            <div class="{pill_class}">{html.escape(pill_text)}</div>
        </div>
        <div class="project-browser-metrics">
            <div class="project-browser-metric">
                <div class="project-browser-metric-label">Rows</div>
                <div class="project-browser-metric-value">{rows:,}</div>
            </div>
            <div class="project-browser-metric">
                <div class="project-browser-metric-label">Flagged</div>
                <div class="project-browser-metric-value">{flagged:,}</div>
            </div>
            <div class="project-browser-metric">
                <div class="project-browser-metric-label">Review rate</div>
                <div class="project-browser-metric-value">{review_rate:.1%}</div>
            </div>
        </div>
        <div class="project-browser-footer">
            <span>{updated_at}</span>
            <span>{source_file}</span>
        </div>
    </div>
    """


def glossary_card(term: str, body: str) -> str:
    return f"""
    <div class="glossary-card">
        <div class="glossary-term">{term}</div>
        <div class="glossary-body">{body}</div>
    </div>
    """


def summary_panel(title: str, summary: str, next_step: str) -> str:
    return f"""
    <div class="summary-panel">
        <p class="section-eyebrow">What this means</p>
        <div class="summary-heading">{title}</div>
        <p class="summary-text">{summary}</p>
        <p class="section-eyebrow" style="margin-top: 1rem;">What the auditor should do next</p>
        <p class="summary-text">{next_step}</p>
    </div>
    """


def glossary_entries() -> list[tuple[str, str]]:
    return [
        (
            "Risk score",
            "A percentage that shows how strongly the model suspects a transaction. Higher means more suspicious.",
        ),
        (
            "Needs review",
            "The model believes a human auditor should look at this transaction before trusting it.",
        ),
        (
            "Amount deviation",
            "How far a payment is from the vendor's historical average before that transaction happened.",
        ),
        (
            "Duplicate invoice signal",
            "The invoice had already appeared earlier in the ledger, which can indicate a duplicate payment attempt.",
        ),
        (
            "Rounded amount pattern",
            "Amounts like 2500.00 or 4100.00 can be a manual-entry signal because they look unnaturally neat.",
        ),
        (
            "Same-day vendor activity",
            "How many earlier payments had already hit the same vendor on that day. Sudden bursts can be suspicious.",
        ),
        (
            "Split-payment pattern",
            "Several same-day payments or an amount just below a common approval limit can indicate an attempt to bypass controls.",
        ),
    ]


def validate_ledger(
    dataframe: pd.DataFrame,
    explicit_mapping: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    prepared_bundle = prepare_ledger_for_analysis(dataframe, explicit_mapping=explicit_mapping)
    return prepared_bundle["prepared"]


def set_active_dataset(
    dataframe: pd.DataFrame,
    label: str,
    source_key: str,
    explicit_mapping: dict[str, str | None] | None = None,
) -> None:
    prepared_bundle = prepare_ledger_for_analysis(dataframe, explicit_mapping=explicit_mapping)
    st.session_state["analysis"] = run_audit_analysis(prepared_bundle["prepared"])
    st.session_state["dataset_label"] = label
    st.session_state["active_source_key"] = source_key
    st.session_state["current_project_id"] = None
    st.session_state["current_project_name"] = None
    st.session_state["last_data_quality_report"] = prepared_bundle["quality_report"]
    st.session_state["last_quarantined_rows"] = prepared_bundle["quarantine"]


def clear_active_project_context() -> None:
    st.session_state["analysis"] = None
    st.session_state["dataset_label"] = None
    st.session_state["active_source_key"] = None
    st.session_state["current_project_id"] = None
    st.session_state["current_project_name"] = None
    st.session_state["last_data_quality_report"] = None
    st.session_state["last_quarantined_rows"] = None


def project_metrics_from_analysis(analysis: dict[str, object]) -> dict[str, object]:
    scored = analysis["scored"]
    flagged_count = int(scored["fraud_prediction"].sum())
    review_rate = flagged_count / max(scored.shape[0], 1)
    return {
        "rows": int(scored.shape[0]),
        "flagged_count": flagged_count,
        "review_rate": float(review_rate),
    }


def set_project_context(project: dict[str, object]) -> None:
    st.session_state["current_project_id"] = project.get("id")
    st.session_state["current_project_name"] = project.get("name")


def create_audit_project(
    project_name: str,
    client_name: str,
    raw_bytes: bytes,
    file_name: str,
    status: str = "In progress",
    notes: str = "",
    explicit_mapping: dict[str, str | None] | None = None,
    activate: bool = True,
) -> dict[str, object]:
    ensure_workspace_store()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    project_id = f"{timestamp}-{slugify_project_name(project_name)}"
    project_dir = PROJECTS_ROOT / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    stored_file_name = safe_file_name(file_name)
    ledger_path = project_dir / stored_file_name
    ledger_path.write_bytes(raw_bytes)

    dataframe = read_ledger_csv(raw_bytes)
    prepared_bundle = prepare_ledger_for_analysis(dataframe, explicit_mapping=explicit_mapping)
    prepared = prepared_bundle["prepared"]
    quarantine = prepared_bundle["quarantine"]
    quality_report = prepared_bundle["quality_report"]
    analysis = run_audit_analysis(prepared)
    metrics = project_metrics_from_analysis(analysis)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    quality_report_path = project_dir / "data_quality_report.json"
    quarantine_path = project_dir / "quarantined_rows.csv"
    quality_report_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
    quarantine.to_csv(quarantine_path, index=False)
    project_record = {
        "id": project_id,
        "name": project_name.strip(),
        "client": client_name.strip() or "Unspecified client",
        "status": status if status in PROJECT_STATUS_OPTIONS else PROJECT_STATUS_OPTIONS[0],
        "notes": notes.strip(),
        "created_at_utc": created_at,
        "updated_at_utc": created_at,
        "source_file": stored_file_name,
        "ledger_path": str(ledger_path),
        **metrics,
    }
    write_project_analysis_cache(ledger_path, analysis)

    _upsert_project_record(project_record)

    if activate:
        st.session_state["analysis"] = apply_case_feedback_to_analysis(analysis, project_id=project_id)
        st.session_state["dataset_label"] = f"Project: {project_record['name']}"
        st.session_state["active_source_key"] = f"project:{project_id}"
        st.session_state["last_data_quality_report"] = quality_report
        st.session_state["last_quarantined_rows"] = quarantine
        set_project_context(project_record)
    return project_record


def open_audit_project(project_id: str) -> dict[str, object]:
    project = get_project_record(project_id)
    if project is None:
        raise ValueError("That audit project could not be found.")

    ledger_path = Path(str(project["ledger_path"]))
    if not ledger_path.exists():
        raise ValueError("The saved project ledger file is missing.")

    analysis = read_project_analysis_cache(ledger_path)
    if analysis is None:
        dataframe = read_ledger_csv_file(ledger_path)
        prepared_bundle = prepare_ledger_for_analysis(dataframe)
        analysis = run_audit_analysis(prepared_bundle["prepared"])
        write_project_analysis_cache(ledger_path, analysis)

    st.session_state["analysis"] = apply_case_feedback_to_analysis(analysis, project_id=project_id)
    st.session_state["dataset_label"] = f"Project: {project['name']}"
    st.session_state["active_source_key"] = f"project:{project_id}"
    quality_report, quarantine_rows = load_project_quality(project_id)
    st.session_state["last_data_quality_report"] = quality_report
    st.session_state["last_quarantined_rows"] = quarantine_rows

    analysis = current_analysis()
    if analysis is not None:
        metrics = project_metrics_from_analysis(analysis)
        update_audit_project(project_id, **metrics)

    set_project_context(project)
    refreshed = get_project_record(project_id) or project
    return refreshed


def update_audit_project(project_id: str, **changes: object) -> dict[str, object]:
    existing_record = get_project_record(project_id)
    if existing_record is None:
        raise ValueError("That audit project could not be updated.")

    updated_record = {
        **existing_record,
        **changes,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _upsert_project_record(updated_record)
    if st.session_state.get("current_project_id") == project_id:
        st.session_state["current_project_name"] = updated_record.get("name")
    return updated_record


def delete_audit_project(project_id: str) -> None:
    project_to_remove = get_project_record(project_id)
    if project_to_remove is None:
        raise ValueError("That audit project could not be deleted.")

    ledger_path = Path(str(project_to_remove.get("ledger_path", "")))
    project_dir = ledger_path.parent if str(ledger_path) else PROJECTS_ROOT / project_id
    if project_dir.exists() and project_dir.is_dir():
        shutil.rmtree(project_dir, ignore_errors=True)

    with _project_db_connection() as connection:
        connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        connection.execute("DELETE FROM case_feedback WHERE project_id = ?", (project_id,))

    if st.session_state.get("current_project_id") == project_id:
        clear_active_project_context()


def reset_all_projects() -> None:
    ensure_workspace_store()
    if PROJECTS_INDEX_PATH.exists():
        PROJECTS_INDEX_PATH.unlink(missing_ok=True)
    if PROJECTS_ROOT.exists() and PROJECTS_ROOT.is_dir():
        for project_dir in PROJECTS_ROOT.iterdir():
            if project_dir.is_dir():
                shutil.rmtree(project_dir, ignore_errors=True)
    with _project_db_connection() as connection:
        connection.execute("DELETE FROM case_feedback")
        connection.execute("DELETE FROM projects")
    clear_active_project_context()


def demo_pack_csv_files() -> list[Path]:
    demo_root = DEMO_DATASETS_ROOT
    if demo_root.exists():
        paths = sorted(demo_root.glob("*.csv"))
        if paths:
            return paths
    return [path for path in SAMPLE_DATASETS.values() if path.exists()]


def load_demo_pack_into_projects() -> list[dict[str, object]]:
    created_projects: list[dict[str, object]] = []
    demo_files = demo_pack_csv_files()
    for demo_file in demo_files:
        project_name = demo_file.stem.replace("_", " ").replace("-", " ").title()
        project = create_audit_project(
            project_name=project_name,
            client_name="Demo Client",
            raw_bytes=demo_file.read_bytes(),
            file_name=demo_file.name,
            status="In progress",
            notes="Generated from bundled demo ledger pack.",
            activate=False,
        )
        created_projects.append(project)
    if created_projects:
        latest = created_projects[0]
        open_audit_project(str(latest["id"]))
    return created_projects


def audit_projects_dataframe() -> pd.DataFrame:
    projects = list_audit_projects()
    if not projects:
        return pd.DataFrame(
            columns=["Project", "Client", "Status", "Rows", "Flagged", "Review rate", "Updated", "File"]
        )
    rows = []
    for project in projects:
        rows.append(
            {
                "Project": str(project.get("name", "")),
                "Client": str(project.get("client", "")),
                "Status": str(project.get("status", "")),
                "Rows": int(project.get("rows", 0)),
                "Flagged": int(project.get("flagged_count", 0)),
                "Review rate": f"{float(project.get('review_rate', 0.0)):.1%}",
                "Updated": str(project.get("updated_at_utc", ""))[:16].replace("T", " "),
                "File": str(project.get("source_file", "")),
            }
        )
    return pd.DataFrame(rows)


def set_uploaded_dataset(raw_bytes: bytes, file_name: str) -> None:
    source_key = hashlib.md5(raw_bytes).hexdigest()
    if st.session_state.get("active_source_key") == source_key:
        return
    dataframe = read_ledger_csv(raw_bytes)
    set_active_dataset(dataframe, f"Uploaded: {file_name}", source_key)


def preprocess_ledger(dataframe: pd.DataFrame) -> pd.DataFrame:
    ledger = dataframe.copy()
    ledger.columns = [column.strip() for column in ledger.columns]
    ledger["date"] = pd.to_datetime(ledger["date"], errors="coerce")
    ledger["amount"] = pd.to_numeric(ledger["amount"], errors="coerce")

    if ledger["date"].isna().any():
        raise ValueError("One or more transaction dates could not be parsed.")
    if ledger["amount"].isna().any():
        raise ValueError("One or more transaction amounts are not numeric.")

    for column in [
        "transaction_id",
        "vendor",
        "department",
        "account_type",
        "payment_method",
        "employee",
        "invoice_id",
    ]:
        ledger[column] = ledger[column].astype(str).str.strip()

    ledger = ledger.sort_values(["date", "transaction_id"]).reset_index(drop=True)
    ledger["txn_day"] = ledger["date"].dt.date
    ledger["posting_hour"] = ledger["date"].dt.hour
    ledger["posting_weekday"] = ledger["date"].dt.weekday
    ledger["is_weekend"] = (ledger["posting_weekday"] >= 5).astype(int)
    ledger["invoice_id_normalized"] = ledger["invoice_id"].map(normalize_invoice_id)

    global_avg_prior = ledger["amount"].expanding().mean().shift(1)
    global_std_prior = ledger["amount"].expanding().std(ddof=0).shift(1)
    department_avg_prior = ledger.groupby("department")["amount"].transform(
        lambda series: series.expanding().mean().shift(1)
    )
    vendor_avg_prior = ledger.groupby("vendor")["amount"].transform(
        lambda series: series.expanding().mean().shift(1)
    )
    vendor_std_prior = ledger.groupby("vendor")["amount"].transform(
        lambda series: series.expanding().std(ddof=0).shift(1)
    )

    ledger["vendor_transaction_count"] = ledger.groupby("vendor").cumcount()
    ledger["vendor_avg_amount"] = (
        vendor_avg_prior.fillna(department_avg_prior).fillna(global_avg_prior).fillna(ledger["amount"])
    )
    fallback_std = float(ledger["amount"].std(ddof=0) or 1.0)
    ledger["vendor_std_amount"] = vendor_std_prior.fillna(global_std_prior).fillna(fallback_std)
    ledger["department_avg_amount"] = department_avg_prior.fillna(global_avg_prior).fillna(ledger["amount"])

    ledger["amount_deviation_signed"] = ledger["amount"] - ledger["vendor_avg_amount"]
    ledger["amount_deviation_abs"] = ledger["amount_deviation_signed"].abs()
    safe_vendor_avg = ledger["vendor_avg_amount"].replace(0, np.nan)
    ledger["amount_to_vendor_avg_ratio"] = (
        ledger["amount"] / safe_vendor_avg
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 25.0)
    safe_vendor_std = ledger["vendor_std_amount"].replace(0, np.nan)
    ledger["vendor_amount_zscore"] = (
        ledger["amount_deviation_signed"] / safe_vendor_std
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-15.0, 15.0)
    ledger["transactions_per_day_vendor"] = ledger.groupby(["vendor", "txn_day"]).cumcount()
    ledger["invoice_count"] = ledger.groupby("invoice_id").cumcount()
    invoice_last_date = ledger.groupby("invoice_id")["date"].shift(1)
    ledger["invoice_gap_days"] = (
        (ledger["date"] - invoice_last_date).dt.total_seconds() / 86400
    ).fillna(999.0).clip(0.0, 999.0)
    ledger = compute_fuzzy_invoice_features(ledger)
    ledger["vendor_department_count"] = ledger.groupby(["vendor", "department"]).cumcount()
    ledger["employee_vendor_count"] = ledger.groupby(["employee", "vendor"]).cumcount()
    ledger["employee_department_count"] = ledger.groupby(["employee", "department"]).cumcount()
    ledger["department_amount_deviation"] = ledger["amount"] - ledger["department_avg_amount"]
    ledger["vendor_payment_method_count"] = ledger.groupby(["vendor", "payment_method"]).cumcount()
    ledger["vendor_account_type_count"] = ledger.groupby(["vendor", "account_type"]).cumcount()
    vendor_day_running_total = ledger.groupby(["vendor", "txn_day"])["amount"].cumsum()
    ledger["vendor_day_running_total_with_current"] = vendor_day_running_total
    ledger["vendor_day_running_total_prior"] = (vendor_day_running_total - ledger["amount"]).clip(lower=0.0)
    nearest_approval_gap = np.minimum.reduce([np.abs(ledger["amount"] - threshold) for threshold in APPROVAL_THRESHOLDS])
    below_threshold_gaps = np.column_stack([(threshold - ledger["amount"]).clip(lower=0.0) for threshold in APPROVAL_THRESHOLDS])
    ledger["is_just_below_approval"] = (
        (below_threshold_gaps <= np.array([150.0, 250.0, 400.0])).any(axis=1)
    ).astype(int)
    crosses_threshold = np.zeros(len(ledger), dtype=int)
    for threshold in APPROVAL_THRESHOLDS:
        crosses_threshold |= (
            (ledger["vendor_day_running_total_prior"] < threshold)
            & (ledger["vendor_day_running_total_with_current"] >= threshold)
            & (ledger["transactions_per_day_vendor"] > 0)
        ).astype(int)
    ledger["crosses_approval_with_same_day_vendor"] = crosses_threshold.astype(int)
    ledger["is_new_vendor_department_pair"] = (
        (ledger["vendor_transaction_count"] > 0) & (ledger["vendor_department_count"] == 0)
    ).astype(int)
    ledger["is_new_vendor_payment_method"] = (
        (ledger["vendor_transaction_count"] > 0) & (ledger["vendor_payment_method_count"] == 0)
    ).astype(int)
    ledger["is_new_vendor_account_type"] = (
        (ledger["vendor_transaction_count"] > 0) & (ledger["vendor_account_type_count"] == 0)
    ).astype(int)
    vendor_last_date = ledger.groupby("vendor")["date"].shift(1)
    employee_last_date = ledger.groupby("employee")["date"].shift(1)
    ledger["days_since_vendor_last_payment"] = (
        (ledger["date"] - vendor_last_date).dt.total_seconds() / 86400
    ).fillna(999.0).clip(0.0, 999.0)
    ledger["days_since_employee_last_payment"] = (
        (ledger["date"] - employee_last_date).dt.total_seconds() / 86400
    ).fillna(999.0).clip(0.0, 999.0)
    ledger["is_duplicate_invoice"] = (ledger["invoice_count"] > 0).astype(int)

    amount_cents = (ledger["amount"] * 100).round().astype("int64")
    amount_dollars = amount_cents // 100
    ledger["is_round_number"] = (
        (amount_cents % 100 == 0)
        & ((amount_dollars % 10 == 0) | (amount_dollars % 100 == 0))
    ).astype(int)
    return ledger


def build_feature_matrix(
    ledger: pd.DataFrame,
    feature_names: list[str],
    category_columns: dict[str, list[str]],
) -> pd.DataFrame:
    base_frame = ledger[BASE_FEATURES].copy()
    categorical_frames = []
    for source_column, encoded_columns in category_columns.items():
        prefix = CATEGORICAL_PREFIXES[source_column]
        values = ledger[source_column].astype(str)
        encoded_frames = pd.DataFrame(
            {
                column: (values == column[len(prefix) :]).astype(int)
                for column in encoded_columns
            },
            index=ledger.index,
        )
        categorical_frames.append(encoded_frames)

    feature_matrix = pd.concat([base_frame, *categorical_frames], axis=1)
    return feature_matrix.reindex(columns=feature_names, fill_value=0)


def review_status(
    probability: float,
    prediction: int,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    priority_threshold: float = DEFAULT_PRIORITY_THRESHOLD,
    watchlist_threshold: float = DEFAULT_WATCHLIST_THRESHOLD,
) -> str:
    if prediction == 1:
        return "Needs review"
    return "Looks routine"


def risk_level(probability: float) -> str:
    if probability >= 0.85:
        return "Critical"
    if probability >= 0.65:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


def anomaly_level(score: float) -> str:
    if score >= 0.82:
        return "Critical"
    if score >= 0.62:
        return "High"
    if score >= 0.38:
        return "Medium"
    return "Low"


def _normalize_score(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    low = float(values.quantile(0.05))
    high = float(values.quantile(0.95))
    if abs(high - low) < 1e-9:
        return pd.Series(np.zeros(len(values)), index=values.index)
    normalized = (values - low) / (high - low)
    return normalized.clip(0.0, 1.0)


def compute_anomaly_scores(ledger: pd.DataFrame) -> pd.Series:
    anomaly_features = [
        "amount",
        "vendor_transaction_count",
        "vendor_avg_amount",
        "amount_deviation_abs",
        "transactions_per_day_vendor",
        "is_duplicate_invoice",
        "is_round_number",
    ]
    working = ledger[anomaly_features].replace([np.inf, -np.inf], np.nan).copy()
    for column in working.columns:
        median_value = float(working[column].median()) if not working[column].dropna().empty else 0.0
        working[column] = working[column].fillna(median_value).astype(float)

    if IsolationForest is not None and len(working) >= 40:
        try:
            detector = IsolationForest(
                n_estimators=120,
                contamination=0.08,
                random_state=42,
                n_jobs=1,
            )
            detector.fit(working)
            raw_scores = -detector.decision_function(working)
            return _normalize_score(pd.Series(raw_scores, index=working.index).astype(float))
        except Exception:
            pass

    zscore_frame = working.copy()
    for column in zscore_frame.columns:
        std = float(zscore_frame[column].std(ddof=0))
        if std <= 0:
            zscore_frame[column] = 0.0
        else:
            zscore_frame[column] = ((zscore_frame[column] - zscore_frame[column].mean()) / std).abs()
    fallback_scores = zscore_frame.mean(axis=1).astype(float)
    return _normalize_score(fallback_scores)


def select_primary_driver(impact_series: pd.Series) -> str:
    positive_impact = impact_series[impact_series > 0].sort_values(ascending=False)
    if not positive_impact.empty:
        return positive_impact.index[0]
    return impact_series.abs().sort_values(ascending=False).index[0]


def build_rule_based_summary(row: pd.Series, feature_name: str) -> dict[str, str]:
    if feature_name == "is_duplicate_invoice":
        return {
            "title": "The invoice looks duplicated",
            "summary": (
                f"Invoice `{row['invoice_id']}` had already appeared earlier in the ledger. "
                "That is one of the strongest classic warning signs for duplicate payments."
            ),
            "next_step": "Compare every occurrence of this invoice ID, then match each one to its approval trail and support documents.",
        }
    if feature_name == "invoice_count":
        return {
            "title": "The invoice appears more than once",
            "summary": (
                f"Invoice `{row['invoice_id']}` had already appeared {int(row['invoice_count'])} time(s) before this payment. "
                "That reuse pattern is strong evidence that the payment may need duplicate-payment review."
            ),
            "next_step": "Pull every line with this invoice ID and verify whether each payment is supported by a separate legitimate document.",
        }
    if feature_name == "invoice_gap_days":
        return {
            "title": "The same invoice returned after a short gap",
            "summary": (
                f"Invoice `{row['invoice_id']}` was last seen {row['invoice_gap_days']:.1f} day(s) before this payment. "
                "Short gaps between identical invoice references can point to resubmission or duplicate-processing risk."
            ),
            "next_step": "Check the earlier invoice occurrence and confirm whether this is a legitimate follow-up payment or a resubmitted duplicate.",
        }
    if feature_name in {"fuzzy_invoice_match_count", "max_invoice_similarity"}:
        return {
            "title": "The invoice looks very similar to an earlier one",
            "summary": (
                f"Auditr found {int(row['fuzzy_invoice_match_count'])} earlier invoice(s) for vendor `{row['vendor']}` with a near-match invoice pattern. "
                f"The strongest similarity score is {row['max_invoice_similarity']:.2f}, which can indicate a typo-ed duplicate or a lightly edited re-submission."
            ),
            "next_step": "Compare this invoice against prior vendor invoices with similar IDs, then confirm whether the reference was re-keyed or duplicated.",
        }
    if feature_name == "transactions_per_day_vendor":
        return {
            "title": "The vendor was unusually active on one day",
            "summary": (
                f"Vendor `{row['vendor']}` already had {int(row['transactions_per_day_vendor'])} earlier payment(s) on {row['date'].date()}. "
                "That burst pattern is stronger than the model expects for normal activity."
            ),
            "next_step": "Check whether the payments should have been combined, delayed, or routed through a different approval path.",
        }
    if feature_name in {"vendor_day_running_total_prior", "vendor_day_running_total_with_current"}:
        return {
            "title": "Several same-day payments stacked together",
            "summary": (
                f"Before this payment, same-day payments to `{row['vendor']}` already totaled {format_currency(row['vendor_day_running_total_prior'])}. "
                f"With the current payment included, the same-day total reached {format_currency(row['vendor_day_running_total_with_current'])}."
            ),
            "next_step": "Review whether the payments were intentionally split across the day and whether they should have been reviewed together.",
        }
    if feature_name == "is_just_below_approval":
        return {
            "title": "The amount sits just below a common approval limit",
            "summary": (
                f"The payment of {format_currency(row['amount'])} lands unusually close to a common approval threshold. "
                "That pattern can indicate an attempt to avoid a higher approval level."
            ),
            "next_step": "Check the approval matrix and confirm whether the amount was intentionally kept just under an escalation threshold.",
        }
    if feature_name == "crosses_approval_with_same_day_vendor":
        return {
            "title": "Same-day payments crossed an approval threshold together",
            "summary": (
                f"When combined with earlier same-day payments to `{row['vendor']}`, this transaction pushed the total to {format_currency(row['vendor_day_running_total_with_current'])}. "
                "That stacked total can indicate split payments designed to bypass controls."
            ),
            "next_step": "Review all same-day payments to this vendor together and compare the combined total against approval policy.",
        }
    if feature_name in {"amount_deviation_abs", "vendor_amount_zscore", "amount_to_vendor_avg_ratio"}:
        return {
            "title": "The payment amount is unusual for this vendor",
            "summary": (
                f"This payment is {format_currency(row['amount_deviation_abs'])} away from `{row['vendor']}`'s historical average "
                f"of {format_currency(row['vendor_avg_amount'])} before this transaction."
            ),
            "next_step": "Review the invoice and contract to confirm the amount is justified and not a one-off spike.",
        }
    if feature_name == "amount_deviation_signed":
        direction = "above" if row["amount_deviation_signed"] >= 0 else "below"
        return {
            "title": "The amount sits away from the vendor norm",
            "summary": (
                f"This payment is {format_currency(abs(row['amount_deviation_signed']))} {direction} "
                f"`{row['vendor']}`'s historical average payment of {format_currency(row['vendor_avg_amount'])}."
            ),
            "next_step": "Check whether the amount difference is expected for this vendor or whether it represents an exception that needs approval evidence.",
        }
    if feature_name == "vendor_transaction_count":
        return {
            "title": "The vendor history level mattered",
            "summary": (
                f"Before this payment, vendor `{row['vendor']}` had {int(row['vendor_transaction_count'])} earlier transaction(s) in the ledger. "
                "That history level shaped how the model interpreted the transaction."
            ),
            "next_step": "Inspect the full vendor history, then compare approval quality across repeated payments.",
        }
    if feature_name == "vendor_department_count":
        return {
            "title": "This vendor is heavily tied to one department",
            "summary": (
                f"Before this payment, vendor `{row['vendor']}` had {int(row['vendor_department_count'])} earlier transaction(s) with department `{row['department']}`. "
                "That repeated pairing shaped the model score for this transaction."
            ),
            "next_step": "Check whether this department-vendor relationship is normal business activity or whether it hides repeated exception handling.",
        }
    if feature_name == "is_new_vendor_department_pair":
        return {
            "title": "The vendor appeared in a new department context",
            "summary": (
                f"Vendor `{row['vendor']}` had prior history, but not with department `{row['department']}`. "
                "That new pairing can be legitimate, but it also shows up in fraud cases where a familiar vendor is routed through a different cost center."
            ),
            "next_step": "Confirm the department should legitimately be paying this vendor and review the approval trail for the new pairing.",
        }
    if feature_name == "employee_vendor_count":
        return {
            "title": "The employee-vendor relationship mattered",
            "summary": (
                f"Before this payment, employee `{row['employee']}` had {int(row['employee_vendor_count'])} earlier transaction(s) with vendor `{row['vendor']}`. "
                "That relationship pattern influenced the model decision."
            ),
            "next_step": "Review whether the employee should normally initiate or approve payments to this vendor.",
        }
    if feature_name == "employee_department_count":
        return {
            "title": "The employee's department history mattered",
            "summary": (
                f"Before this payment, employee `{row['employee']}` had {int(row['employee_department_count'])} earlier transaction(s) in department `{row['department']}`. "
                "That historical pattern influenced the model score."
            ),
            "next_step": "Check whether the employee normally works with this department and whether the assignment is supported by policy.",
        }
    if feature_name == "is_round_number":
        return {
            "title": "The amount is suspiciously neat",
            "summary": (
                f"The payment value {format_currency(row['amount'])} is a very rounded number. "
                "Fraudulent or manually adjusted payments often look cleaner than real invoice totals."
            ),
            "next_step": "Check price detail, tax detail, and whether the payment was manually edited before posting.",
        }
    if feature_name == "amount":
        return {
            "title": "The payment amount is large on its own",
            "summary": (
                f"The transaction total of {format_currency(row['amount'])} is materially large, "
                "so the model treats it as important risk context."
            ),
            "next_step": "Confirm the amount sits within approval limits and matches the source invoice exactly.",
        }
    if feature_name == "vendor_avg_amount":
        return {
            "title": "The vendor's historical average mattered",
            "summary": (
                f"Before this payment, `{row['vendor']}` had a historical average payment value of {format_currency(row['vendor_avg_amount'])}. "
                "That vendor baseline shaped the risk score for this transaction."
            ),
            "next_step": "Compare this transaction against the vendor's previous invoices to see whether the pattern is still reasonable.",
        }
    if feature_name == "department_avg_amount":
        return {
            "title": "The department baseline mattered",
            "summary": (
                f"Before this payment, the `{row['department']}` department had a historical average of {format_currency(row['department_avg_amount'])}. "
                "That departmental baseline influenced the model's interpretation of the amount."
            ),
            "next_step": "Check whether this payment fits the department's historical spending pattern and approval norms.",
        }
    if feature_name == "department_amount_deviation":
        direction = "above" if row["department_amount_deviation"] >= 0 else "below"
        return {
            "title": "The payment is unusual for the department",
            "summary": (
                f"This transaction is {format_currency(abs(row['department_amount_deviation']))} {direction} "
                f"the `{row['department']}` department average of {format_currency(row['department_avg_amount'])}."
            ),
            "next_step": "Check whether the spending level fits the department's normal operating pattern and approval limits.",
        }
    if feature_name == "vendor_payment_method_count":
        return {
            "title": "The payment method does not fit the vendor's history",
            "summary": (
                f"Before this transaction, vendor `{row['vendor']}` had used payment method `{row['payment_method']}` "
                f"{int(row['vendor_payment_method_count'])} time(s). That history level influenced the score."
            ),
            "next_step": "Confirm that this payment method is normal for the vendor and matches policy and prior invoice handling.",
        }
    if feature_name == "is_new_vendor_payment_method":
        return {
            "title": "The vendor is using a new payment method",
            "summary": (
                f"Vendor `{row['vendor']}` had prior history, but this is the first time the ledger shows payment method `{row['payment_method']}` for that vendor. "
                "That switch can be operationally valid, but it is also a known fraud pattern."
            ),
            "next_step": "Verify the payment method change, bank details, and whether the vendor master data was updated through the proper process.",
        }
    if feature_name == "vendor_account_type_count":
        return {
            "title": "The account coding looks unusual for the vendor",
            "summary": (
                f"Before this transaction, vendor `{row['vendor']}` had used account type `{row['account_type']}` "
                f"{int(row['vendor_account_type_count'])} time(s). That made the coding pattern relevant to the score."
            ),
            "next_step": "Verify that the account classification matches the actual purpose of the payment and is not a recoding workaround.",
        }
    if feature_name == "is_new_vendor_account_type":
        return {
            "title": "The vendor is hitting a new account type",
            "summary": (
                f"Vendor `{row['vendor']}` had prior history, but this is the first time it appears under account type `{row['account_type']}`. "
                "That change can indicate miscoding or an attempt to reroute the expense."
            ),
            "next_step": "Check whether the account type is appropriate for the invoice and whether the classification changed from the vendor's normal pattern.",
        }
    if feature_name == "days_since_vendor_last_payment":
        return {
            "title": "The vendor was paid again very quickly",
            "summary": (
                f"The gap since the previous payment to `{row['vendor']}` was {row['days_since_vendor_last_payment']:.2f} day(s). "
                "Short gaps can point to payment splitting, duplicate processing, or accelerated approval."
            ),
            "next_step": "Check the earlier vendor payment, then confirm whether this rapid follow-up is operationally justified.",
        }
    if feature_name == "days_since_employee_last_payment":
        return {
            "title": "The employee posted payments in quick succession",
            "summary": (
                f"The gap since employee `{row['employee']}` last posted a payment was {row['days_since_employee_last_payment']:.2f} day(s). "
                "That rapid sequence influenced the risk score."
            ),
            "next_step": "Review whether the employee is working through a normal batch or whether the rapid sequence indicates exception handling.",
        }
    if feature_name == "posting_hour":
        return {
            "title": "The posting time influenced the score",
            "summary": (
                f"This transaction was posted at hour {int(row['posting_hour']):02d}:00. "
                "Timing can matter when the model sees a payment pattern outside the usual operating rhythm."
            ),
            "next_step": "Check whether the posting time is normal for the finance process or whether it was entered late, rushed, or outside normal controls.",
        }
    if feature_name == "is_weekend":
        return {
            "title": "The payment timing looks unusual",
            "summary": "The transaction landed on a weekend pattern, which the model treats as a higher-risk operating context.",
            "next_step": "Confirm whether weekend posting is normal for this business process and supported by the approval trail.",
        }
    if feature_name.startswith("vendor_"):
        return {
            "title": "The vendor identity pushed the score upward",
            "summary": (
                f"The model has learned a risk pattern around vendor `{row['vendor']}`, so the vendor itself raised this case."
            ),
            "next_step": "Review vendor onboarding, history, and any previous exceptions tied to this supplier.",
        }
    if feature_name.startswith("department_"):
        return {
            "title": "The department context mattered",
            "summary": f"The `{row['department']}` department pattern contributed to the risk score for this payment.",
            "next_step": "Check whether this department has unusual approval patterns or recent exception activity.",
        }
    if feature_name.startswith("payment_method_"):
        return {
            "title": "The payment method raised concern",
            "summary": (
                f"The use of `{row['payment_method']}` matches a payment pattern that the model treats as riskier."
            ),
            "next_step": "Confirm this payment method is normal for the vendor and supported by policy.",
        }
    if feature_name.startswith("employee_"):
        return {
            "title": "The employee context mattered",
            "summary": (
                f"The employee attribute for `{row['employee']}` was one of the reasons the transaction moved toward review."
            ),
            "next_step": "Check whether the employee, vendor, department, and account type align with normal business practice.",
        }
    if feature_name.startswith("account_type_"):
        return {
            "title": "The account type raised the score",
            "summary": (
                f"The `{row['account_type']}` account classification contributed to the risk estimate for this payment."
            ),
            "next_step": "Verify the account coding and make sure the ledger classification matches the actual spend purpose.",
        }
    return {
        "title": "Several smaller signals combined",
        "summary": "No single factor dominated. The model reached this score because multiple moderate signals stacked together.",
        "next_step": "Review the top contributors, then validate the invoice, approval chain, and vendor history.",
    }


@st.cache_data(show_spinner=False)
def run_audit_analysis(dataframe: pd.DataFrame) -> dict[str, object]:
    bundle = load_model_bundle()
    ledger = preprocess_ledger(dataframe)
    feature_matrix = build_feature_matrix(
        ledger,
        bundle["feature_names"],
        bundle["category_columns"],
    )

    model = bundle["model"]
    booster = bundle["booster"]
    metadata = bundle["metadata"]
    fraud_probability = model.predict_proba(feature_matrix)[:, 1]
    decision_threshold = float(metadata.get("decision_threshold", DEFAULT_DECISION_THRESHOLD))
    priority_threshold = float(metadata.get("priority_threshold", DEFAULT_PRIORITY_THRESHOLD))
    watchlist_threshold = float(metadata.get("watchlist_threshold", DEFAULT_WATCHLIST_THRESHOLD))
    fraud_prediction = (fraud_probability >= decision_threshold).astype(int)
    anomaly_score_values = compute_anomaly_scores(ledger)

    dmatrix = xgb.DMatrix(feature_matrix, feature_names=list(feature_matrix.columns))
    raw_contributions = booster.predict(dmatrix, pred_contribs=True)
    contribution_frame = pd.DataFrame(
        raw_contributions[:, :-1],
        columns=feature_matrix.columns,
        index=feature_matrix.index,
    )
    base_values = raw_contributions[:, -1]

    scored = ledger.copy()
    scored["fraud_probability"] = fraud_probability
    scored["fraud_prediction"] = fraud_prediction.astype(int)
    scored["risk_level"] = scored["fraud_probability"].map(risk_level)
    scored["anomaly_score"] = anomaly_score_values.values
    scored["anomaly_level"] = scored["anomaly_score"].map(anomaly_level)
    control_intensity = (
        (
            (scored["is_duplicate_invoice"] > 0).astype(int)
            + (scored["fuzzy_invoice_match_count"] > 0).astype(int)
            + (scored["transactions_per_day_vendor"] > 0).astype(int)
            + (scored["is_just_below_approval"] > 0).astype(int)
            + (scored["crosses_approval_with_same_day_vendor"] > 0).astype(int)
        )
        / 5.0
    )
    scored["blended_risk_score"] = (
        0.66 * scored["fraud_probability"]
        + 0.26 * scored["anomaly_score"]
        + 0.08 * control_intensity
    ).clip(0.0, 1.0)
    scored["blended_priority_score"] = scored["blended_risk_score"]
    scored["review_status"] = [
        review_status(
            probability,
            prediction,
            decision_threshold=decision_threshold,
            priority_threshold=priority_threshold,
            watchlist_threshold=watchlist_threshold,
        )
        for probability, prediction in zip(scored["fraud_probability"], scored["fraud_prediction"])
    ]
    scored["fraud_probability_pct"] = scored["fraud_probability"] * 100
    scored["date_only"] = scored["date"].dt.date
    scored["month_period"] = scored["date"].dt.to_period("M").astype(str)

    drivers = []
    summary_titles = []
    summary_text = []
    next_steps = []
    for row_index in scored.index:
        row = scored.loc[row_index]
        impact_series = contribution_frame.loc[row_index]
        driver = select_primary_driver(impact_series)
        insight = build_rule_based_summary(row, driver)
        drivers.append(driver)
        summary_titles.append(insight["title"])
        summary_text.append(insight["summary"])
        next_steps.append(insight["next_step"])

    scored["primary_driver"] = drivers
    scored["primary_driver_label"] = scored["primary_driver"].map(format_feature_label)
    scored["summary_title"] = summary_titles
    scored["summary_text"] = summary_text
    scored["next_step"] = next_steps

    scored["reviewer_status"] = "Needs review"
    scored["reviewer_note"] = ""
    scored["review_priority_multiplier"] = 1.0

    sorted_index = scored.sort_values("blended_priority_score", ascending=False).index
    scored = scored.loc[sorted_index].reset_index(drop=True)
    feature_matrix = feature_matrix.loc[sorted_index].reset_index(drop=True)
    contribution_frame = contribution_frame.loc[sorted_index].reset_index(drop=True)
    base_values = pd.Series(base_values).loc[sorted_index].reset_index(drop=True)
    flagged = scored[scored["fraud_prediction"] == 1].reset_index(drop=True)

    return {
        "scored": scored,
        "flagged": flagged,
        "features": feature_matrix,
        "contributions": contribution_frame,
        "base_values": base_values,
        "model_metadata": metadata,
    }


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_feature_label(feature_name: str) -> str:
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name].title()
    for source_column, prefix in CATEGORICAL_PREFIXES.items():
        if feature_name.startswith(prefix):
            value = feature_name[len(prefix) :]
            label = source_column.replace("_", " ").title()
            return f"{label}: {value}"
    return feature_name.replace("_", " ").title()


def explain_transaction(analysis: dict[str, object], transaction_id: str) -> dict[str, object]:
    scored = analysis["scored"]
    contributions = analysis["contributions"]
    row = scored.loc[scored["transaction_id"] == transaction_id].iloc[0]
    contribution_row = contributions.loc[row.name]
    top_contributors = contribution_row.reindex(
        contribution_row.abs().sort_values(ascending=False).index
    ).head(8)
    positive_driver = select_primary_driver(contribution_row)
    return {
        "row": row,
        "contributions": contribution_row,
        "top_contributors": top_contributors,
        "positive_driver": positive_driver,
        "summary": build_rule_based_summary(row, positive_driver),
    }


def chart_layout_defaults(title: str | None = None) -> dict[str, object]:
    palette = get_theme_palette()
    return {
        "template": "plotly_white",
        "paper_bgcolor": palette["plot_bg"],
        "plot_bgcolor": palette["plot_bg"],
        "font": {"color": palette["text"], "family": "Manrope, sans-serif"},
        "margin": {"l": 28, "r": 24, "t": 30 if title else 18, "b": 24},
        "title": {"text": title or "", "font": {"size": 16}},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11, "color": palette["muted"]},
        },
        "xaxis": {
            "showgrid": True,
            "gridcolor": palette["line"],
            "zeroline": False,
            "tickfont": {"color": palette["muted"]},
            "title_font": {"color": palette["muted"], "size": 12},
        },
        "yaxis": {
            "showgrid": False,
            "zeroline": False,
            "automargin": True,
            "tickfont": {"color": palette["muted"]},
            "title_font": {"color": palette["muted"], "size": 12},
        },
    }


def plotly_chart_config() -> dict[str, object]:
    return {
        "displayModeBar": False,
        "responsive": True,
    }


def build_review_queue_chart(scored: pd.DataFrame) -> go.Figure:
    palette = get_theme_palette()
    needs_review = int(scored["fraud_prediction"].sum())
    looks_routine = int(scored.shape[0] - needs_review)

    figure = go.Figure(
        go.Pie(
            labels=["Looks routine", "Flagged as risky"],
            values=[looks_routine, needs_review],
            hole=0.68,
            sort=False,
            marker={"colors": [palette["accent"], palette["danger"]]},
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        )
    )
    layout = chart_layout_defaults()
    layout["showlegend"] = False
    layout["margin"] = {"l": 12, "r": 12, "t": 12, "b": 12}
    figure.update_layout(**layout)
    return figure


def build_reason_chart(scored: pd.DataFrame) -> go.Figure:
    palette = get_theme_palette()
    reasons = (
        scored[scored["fraud_prediction"] == 1]["primary_driver_label"]
        .value_counts()
        .head(6)
        .sort_values(ascending=True)
    )
    figure = go.Figure(
        go.Bar(
            x=reasons.values,
            y=reasons.index,
            orientation="h",
            marker_color=palette["amber"],
            text=[f"{value:,}" for value in reasons.values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        )
    )
    left_margin = int(min(max(220, reasons.index.map(len).max() * 11), 380)) if not reasons.empty else 220
    layout = chart_layout_defaults()
    layout["margin"] = {"l": left_margin, "r": 28, "t": 16, "b": 38}
    layout["xaxis"] = {**layout["xaxis"], "title": "Flagged transactions"}
    layout["yaxis"] = {**layout["yaxis"], "title": "", "automargin": True, "ticklabelposition": "outside"}
    figure.update_layout(**layout)
    return figure


def build_department_chart(scored: pd.DataFrame) -> go.Figure:
    palette = get_theme_palette()
    department_view = (
        scored.groupby("department", as_index=False)
        .agg(
            flagged_cases=("fraud_prediction", "sum"),
            avg_probability=("fraud_probability", "mean"),
        )
        .sort_values(["flagged_cases", "avg_probability"], ascending=[False, False])
    )
    top_departments = department_view.head(8).sort_values("flagged_cases", ascending=True)
    figure = go.Figure(
        go.Bar(
            x=top_departments["flagged_cases"],
            y=top_departments["department"],
            orientation="h",
            marker_color=palette["accent"],
            text=[f"{value:,}" for value in top_departments["flagged_cases"]],
            textposition="outside",
            cliponaxis=False,
            customdata=(top_departments["avg_probability"] * 100).round(1),
            hovertemplate="%{y}: %{x:,} flagged transactions<br>Average risk: %{customdata:.1f}%<extra></extra>",
        )
    )
    left_margin = int(min(max(200, top_departments["department"].astype(str).map(len).max() * 11), 320)) if not top_departments.empty else 200
    layout = chart_layout_defaults()
    layout["margin"] = {"l": left_margin, "r": 28, "t": 16, "b": 38}
    layout["xaxis"] = {**layout["xaxis"], "title": "Flagged transactions"}
    layout["yaxis"] = {**layout["yaxis"], "title": "", "automargin": True, "ticklabelposition": "outside"}
    figure.update_layout(**layout)
    return figure


def build_timeline_chart(scored: pd.DataFrame) -> go.Figure:
    palette = get_theme_palette()
    trend = (
        scored.groupby("date_only", as_index=False)
        .agg(
            flagged_cases=("fraud_prediction", "sum"),
            avg_probability=("fraud_probability", "mean"),
        )
        .sort_values("date_only")
    )
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=trend["date_only"],
            y=trend["flagged_cases"],
            name="Flagged as risky",
            marker_color=palette["amber"],
            opacity=0.70,
            yaxis="y2",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trend["date_only"],
            y=trend["avg_probability"],
            mode="lines+markers",
            name="Average risk score",
            line={"color": palette["accent"], "width": 3},
            marker={"size": 7},
            yaxis="y1",
        )
    )
    layout = chart_layout_defaults()
    layout["yaxis"] = {"title": "Average risk score", "tickformat": ".0%", "showgrid": False}
    layout["yaxis2"] = {
        "title": "Flagged as risky",
        "overlaying": "y",
        "side": "right",
        "showgrid": False,
        "tickfont": {"color": palette["muted"]},
        "title_font": {"color": palette["muted"], "size": 12},
    }
    figure.update_layout(**layout)
    return figure


def build_department_risk_heatmap(scored: pd.DataFrame) -> go.Figure:
    palette = get_theme_palette()
    working = scored.copy()
    working["priority_level"] = pd.cut(
        working["blended_priority_score"],
        bins=[-0.001, 0.35, 0.6, 0.82, 1.001],
        labels=RISK_LEVEL_ORDER,
    )
    heatmap_frame = (
        working.groupby(["department", "priority_level"], observed=False)
        .size()
        .reset_index(name="count")
    )
    if heatmap_frame.empty:
        figure = go.Figure()
        figure.update_layout(**chart_layout_defaults())
        figure.add_annotation(
            text="No rows available for heatmap.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": palette["muted"], "size": 13},
        )
        return figure

    pivot = heatmap_frame.pivot(index="department", columns="priority_level", values="count").fillna(0)
    pivot = pivot.reindex(columns=RISK_LEVEL_ORDER, fill_value=0)
    pivot = pivot.sort_values(by=list(pivot.columns), ascending=False).head(12)

    figure = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(column) for column in pivot.columns],
            y=[str(index) for index in pivot.index],
            colorscale=[
                [0.0, "rgba(123, 108, 244, 0.08)"],
                [0.35, "rgba(123, 108, 244, 0.35)"],
                [0.65, "rgba(123, 108, 244, 0.62)"],
                [1.0, "rgba(123, 108, 244, 0.92)"],
            ],
            colorbar={"title": "Transactions"},
            hovertemplate="Department: %{y}<br>Risk level: %{x}<br>Transactions: %{z:,}<extra></extra>",
        )
    )
    layout = chart_layout_defaults()
    layout["margin"] = {"l": 180, "r": 24, "t": 18, "b": 40}
    layout["xaxis"] = {**layout["xaxis"], "title": "Risk level"}
    layout["yaxis"] = {**layout["yaxis"], "title": "Department", "automargin": True}
    figure.update_layout(**layout)
    return figure


def build_risk_trend_chart(
    scored: pd.DataFrame,
    *,
    department: str = "All",
) -> go.Figure:
    palette = get_theme_palette()
    working = scored.copy()
    if department != "All":
        working = working[working["department"] == department].copy()
    if working.empty:
        figure = go.Figure()
        figure.update_layout(**chart_layout_defaults())
        figure.add_annotation(
            text="No rows match the selected trend filter.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": palette["muted"], "size": 13},
        )
        return figure

    trend = (
        working.assign(date_day=working["date"].dt.date)
        .groupby("date_day", as_index=False)
        .agg(
            transactions=("transaction_id", "size"),
            flagged=("fraud_prediction", "sum"),
            avg_model_risk=("fraud_probability", "mean"),
            avg_priority=("blended_priority_score", "mean"),
        )
        .sort_values("date_day")
    )
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=trend["date_day"],
            y=trend["flagged"],
            name="Flagged",
            marker_color=palette["amber"],
            opacity=0.68,
            yaxis="y2",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trend["date_day"],
            y=trend["avg_model_risk"],
            name="Model risk",
            mode="lines+markers",
            line={"color": palette["accent"], "width": 3},
            marker={"size": 6},
            yaxis="y1",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trend["date_day"],
            y=trend["avg_priority"],
            name="Priority score",
            mode="lines",
            line={"color": palette["danger"], "width": 2, "dash": "dot"},
            yaxis="y1",
        )
    )
    layout = chart_layout_defaults()
    layout["yaxis"] = {"title": "Average score", "tickformat": ".0%", "showgrid": False}
    layout["yaxis2"] = {
        "title": "Flagged cases",
        "overlaying": "y",
        "side": "right",
        "showgrid": False,
        "tickfont": {"color": palette["muted"]},
        "title_font": {"color": palette["muted"], "size": 12},
    }
    figure.update_layout(**layout)
    return figure


def build_relationship_network_chart(scored: pd.DataFrame, top_n: int = 180) -> go.Figure:
    palette = get_theme_palette()
    if scored.empty:
        figure = go.Figure()
        figure.update_layout(**chart_layout_defaults())
        return figure

    subset = scored.nlargest(min(top_n, len(scored)), "blended_priority_score").copy()
    subset["employee"] = subset["employee"].astype(str)
    subset["vendor"] = subset["vendor"].astype(str)
    subset["department"] = subset["department"].astype(str)
    subset["invoice_id"] = subset["invoice_id"].astype(str)

    edge_counter: dict[tuple[str, str], dict[str, float]] = {}

    def add_edge(left: str, right: str, score: float) -> None:
        key = (left, right) if left <= right else (right, left)
        if key not in edge_counter:
            edge_counter[key] = {"weight": 0.0, "risk_sum": 0.0}
        edge_counter[key]["weight"] += 1.0
        edge_counter[key]["risk_sum"] += float(score)

    for row in subset.itertuples():
        employee_node = f"employee::{row.employee}"
        vendor_node = f"vendor::{row.vendor}"
        department_node = f"department::{row.department}"
        invoice_node = f"invoice::{row.invoice_id}"
        add_edge(employee_node, vendor_node, float(row.blended_priority_score))
        add_edge(vendor_node, department_node, float(row.blended_priority_score))
        add_edge(vendor_node, invoice_node, float(row.blended_priority_score))

    ranked_edges = sorted(
        edge_counter.items(),
        key=lambda item: (item[1]["risk_sum"], item[1]["weight"]),
        reverse=True,
    )[:180]
    if not ranked_edges:
        figure = go.Figure()
        figure.update_layout(**chart_layout_defaults())
        return figure

    nodes: dict[str, dict[str, object]] = {}
    for (left, right), payload in ranked_edges:
        for node in (left, right):
            node_type, node_label = node.split("::", 1)
            if node not in nodes:
                nodes[node] = {"type": node_type, "label": node_label, "weight": 0.0, "risk_sum": 0.0}
            nodes[node]["weight"] = float(nodes[node]["weight"]) + float(payload["weight"])
            nodes[node]["risk_sum"] = float(nodes[node]["risk_sum"]) + float(payload["risk_sum"])

    if nx is not None:
        graph = nx.Graph()
        for node, node_data in nodes.items():
            graph.add_node(node, **node_data)
        for (left, right), payload in ranked_edges:
            graph.add_edge(left, right, weight=float(payload["weight"]))
        positions = nx.spring_layout(graph, seed=42, weight="weight", k=0.85)
    else:
        nodes_by_type: dict[str, list[str]] = {"employee": [], "vendor": [], "department": [], "invoice": []}
        for node, node_data in nodes.items():
            nodes_by_type[str(node_data["type"])].append(node)
        x_map = {"employee": 0.08, "vendor": 0.36, "department": 0.66, "invoice": 0.92}
        positions = {}
        for node_type, node_list in nodes_by_type.items():
            node_list = sorted(node_list, key=lambda key: float(nodes[key]["risk_sum"]), reverse=True)
            count = max(len(node_list), 1)
            for index, node in enumerate(node_list):
                y_value = 0.1 + 0.8 * (index / max(count - 1, 1))
                positions[node] = np.array([x_map[node_type], y_value])

    edge_x: list[float] = []
    edge_y: list[float] = []
    for (left, right), _payload in ranked_edges:
        if left not in positions or right not in positions:
            continue
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x: list[float] = []
    node_y: list[float] = []
    node_size: list[float] = []
    node_color: list[str] = []
    node_text: list[str] = []
    node_color_map = {
        "employee": palette["accent"],
        "vendor": palette["danger"],
        "department": palette["amber"],
        "invoice": palette["muted"],
    }
    for node, node_data in nodes.items():
        if node not in positions:
            continue
        x_value, y_value = positions[node]
        node_x.append(float(x_value))
        node_y.append(float(y_value))
        weight_value = float(node_data["weight"])
        node_size.append(10 + min(weight_value, 20) * 1.15)
        node_color.append(node_color_map.get(str(node_data["type"]), palette["accent"]))
        node_text.append(
            f"{node_data['type'].title()}: {node_data['label']}<br>"
            f"Linked transactions: {int(weight_value):,}<br>"
            f"Risk weight: {float(node_data['risk_sum']):.2f}"
        )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 0.9, "color": palette["line"]},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker={"size": node_size, "color": node_color, "line": {"width": 1.0, "color": palette["surface_strong"]}},
            hovertemplate="%{text}<extra></extra>",
            text=node_text,
            showlegend=False,
        )
    )
    layout = chart_layout_defaults()
    layout["margin"] = {"l": 18, "r": 18, "t": 18, "b": 18}
    layout["xaxis"] = {"visible": False}
    layout["yaxis"] = {"visible": False}
    figure.update_layout(**layout)
    return figure


def drilldown_transactions_for_risk_trend(
    scored: pd.DataFrame,
    *,
    date_value: str | None = None,
    department: str = "All",
    limit: int = 25,
) -> pd.DataFrame:
    working = scored.copy()
    if department != "All":
        working = working[working["department"] == department].copy()
    if date_value:
        working = working[working["date"].dt.strftime("%Y-%m-%d") == date_value].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "Transaction ID",
                "Date",
                "Vendor",
                "Department",
                "Amount",
                "Model risk",
                "Anomaly",
                "Priority",
                "Main reason",
            ]
        )
    table = working.nlargest(limit, "blended_priority_score")[
        [
            "transaction_id",
            "date",
            "vendor",
            "department",
            "amount",
            "fraud_probability",
            "anomaly_score",
            "blended_priority_score",
            "primary_driver_label",
        ]
    ].copy()
    table.columns = [
        "Transaction ID",
        "Date",
        "Vendor",
        "Department",
        "Amount",
        "Model risk",
        "Anomaly",
        "Priority",
        "Main reason",
    ]
    return table


def build_contribution_chart(top_contributors: pd.Series) -> go.Figure:
    palette = get_theme_palette()
    chart_frame = pd.DataFrame(
        {
            "feature": [format_feature_label(name) for name in top_contributors.index[::-1]],
            "impact": top_contributors.values[::-1],
        }
    )
    chart_frame["direction"] = np.where(chart_frame["impact"] >= 0, "Raised suspicion", "Reduced suspicion")

    figure = px.bar(
        chart_frame,
        x="impact",
        y="feature",
        color="direction",
        orientation="h",
        color_discrete_map={
            "Raised suspicion": palette["danger"],
            "Reduced suspicion": palette["accent"],
        },
        labels={"impact": "Feature impact on the score", "feature": ""},
    )
    figure.update_layout(**chart_layout_defaults())
    figure.add_vline(x=0, line_width=1, line_dash="dash", line_color=palette["muted"])
    return figure


def build_top_risk_table(scored: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    top_risk = scored.nlargest(limit, "blended_priority_score")[
        [
            "transaction_id",
            "date",
            "vendor",
            "department",
            "amount",
            "fraud_probability",
            "anomaly_score",
            "blended_priority_score",
            "primary_driver_label",
            "next_step",
        ]
    ].copy()
    top_risk.columns = [
        "Transaction ID",
        "Date",
        "Vendor",
        "Department",
        "Amount",
        "Risk score",
        "Anomaly score",
        "Priority score",
        "Why it was flagged",
        "What to check next",
    ]
    return top_risk


def build_audit_briefing(scored: pd.DataFrame) -> dict[str, str]:
    flagged = scored[scored["fraud_prediction"] == 1].copy()
    if flagged.empty:
        return {
            "headline": "No transactions were pushed into manual review.",
            "body": "This ledger looks routine under the current review threshold. The auditor can still sample transactions from Transactions if desired.",
        }

    top_reason = flagged["primary_driver_label"].value_counts().idxmax()
    top_department = flagged["department"].value_counts().idxmax()
    top_vendor = flagged["vendor"].value_counts().idxmax()
    headline = (
        f"{flagged.shape[0]:,} transactions need manual review. "
        f"The strongest repeating pattern is {top_reason.lower()}."
    )
    body = (
        f"The heaviest concentration is in {top_department}, and vendor {top_vendor} appears most often in the flagged queue. "
        "Start with the manual review queue below, then open Transactions for a full ledger search."
    )
    return {"headline": headline, "body": body}


def build_vendor_watchlist_table(scored: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    flagged = scored[scored["fraud_prediction"] == 1].copy()
    if flagged.empty:
        return pd.DataFrame(
            columns=["Vendor", "Flagged transactions", "Average risk", "Flagged amount", "Main department"]
        )

    watchlist = (
        flagged.groupby("vendor", as_index=False)
        .agg(
            flagged_transactions=("transaction_id", "size"),
            average_risk=("fraud_probability", "mean"),
            flagged_amount=("amount", "sum"),
            main_department=("department", lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]),
        )
        .sort_values(["flagged_transactions", "average_risk", "flagged_amount"], ascending=[False, False, False])
        .head(limit)
    )
    watchlist.columns = ["Vendor", "Flagged transactions", "Average risk", "Flagged amount", "Main department"]
    return watchlist


def build_vendor_exposure_chart(scored: pd.DataFrame) -> go.Figure:
    palette = get_theme_palette()
    flagged = scored[scored["fraud_prediction"] == 1].copy()
    vendor_view = (
        flagged.groupby("vendor", as_index=False)
        .agg(
            flagged_amount=("amount", "sum"),
            flagged_cases=("transaction_id", "size"),
            average_risk=("fraud_probability", "mean"),
        )
        .sort_values(["flagged_amount", "flagged_cases"], ascending=[False, False])
        .head(8)
        .sort_values("flagged_amount", ascending=True)
    )
    figure = go.Figure(
        go.Bar(
            x=vendor_view["flagged_amount"],
            y=vendor_view["vendor"],
            orientation="h",
            marker_color=palette["danger"],
            text=[format_currency(value) for value in vendor_view["flagged_amount"]],
            textposition="outside",
            customdata=np.column_stack(
                [
                    vendor_view["flagged_cases"].to_numpy(),
                    (vendor_view["average_risk"] * 100).round(1).to_numpy(),
                ]
            ),
            hovertemplate="%{y}<br>Flagged amount: %{text}<br>Flagged cases: %{customdata[0]:,.0f}<br>Average risk: %{customdata[1]:.1f}%<extra></extra>",
        )
    )
    left_margin = int(min(max(120, vendor_view["vendor"].astype(str).map(len).max() * 8), 200)) if not vendor_view.empty else 120
    layout = chart_layout_defaults()
    layout["margin"] = {"l": left_margin, "r": 34, "t": 16, "b": 38}
    layout["xaxis"] = {**layout["xaxis"], "title": "Flagged amount", "tickprefix": "$"}
    layout["yaxis"] = {**layout["yaxis"], "title": "", "automargin": True}
    figure.update_layout(**layout)
    return figure


def build_control_signal_chart(scored: pd.DataFrame) -> go.Figure:
    palette = get_theme_palette()
    flagged = scored[scored["fraud_prediction"] == 1].copy()
    chart_frame = pd.DataFrame(
        [
            {"Signal": "Exact duplicate invoice", "Count": int(flagged["is_duplicate_invoice"].sum())},
            {"Signal": "Near-duplicate invoice", "Count": int((flagged["fuzzy_invoice_match_count"] > 0).sum())},
            {"Signal": "Same-day vendor burst", "Count": int((flagged["transactions_per_day_vendor"] > 0).sum())},
            {"Signal": "Just below approval limit", "Count": int(flagged["is_just_below_approval"].sum())},
            {"Signal": "Split-payment threshold", "Count": int(flagged["crosses_approval_with_same_day_vendor"].sum())},
        ]
    )
    chart_frame = chart_frame.sort_values("Count", ascending=True)
    figure = go.Figure(
        go.Bar(
            x=chart_frame["Count"],
            y=chart_frame["Signal"],
            orientation="h",
            marker_color=palette["amber"],
            text=[f"{value:,}" for value in chart_frame["Count"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:,} flagged cases<extra></extra>",
        )
    )
    left_margin = int(min(max(160, chart_frame["Signal"].astype(str).map(len).max() * 8), 240))
    layout = chart_layout_defaults()
    layout["margin"] = {"l": left_margin, "r": 28, "t": 16, "b": 38}
    layout["xaxis"] = {**layout["xaxis"], "title": "Flagged cases"}
    layout["yaxis"] = {**layout["yaxis"], "title": "", "automargin": True}
    figure.update_layout(**layout)
    return figure


def build_review_queue_export(flagged: pd.DataFrame) -> pd.DataFrame:
    queue = flagged[
        [
            "transaction_id",
            "date",
            "vendor",
            "department",
            "amount",
            "fraud_probability_pct",
            "anomaly_score",
            "blended_priority_score",
            "review_status",
            "reviewer_status",
            "reviewer_note",
            "primary_driver_label",
            "summary_title",
            "next_step",
        ]
    ].copy()
    queue.columns = [
        "Transaction ID",
        "Date",
        "Vendor",
        "Department",
        "Amount",
        "Risk score (%)",
        "Anomaly score",
        "Priority score",
        "Review status",
        "Reviewer status",
        "Reviewer note",
        "Main reason",
        "Summary",
        "What to check next",
    ]
    return queue.sort_values("Priority score", ascending=False)


def describe_feature_observation(row: pd.Series, feature_name: str) -> str:
    if feature_name == "amount_to_vendor_avg_ratio":
        return (
            f"Payment is {row['amount_to_vendor_avg_ratio']:.2f}x the vendor's historical average "
            f"of {format_currency(row['vendor_avg_amount'])}."
        )
    if feature_name == "amount_deviation_abs":
        return f"Payment differs from the vendor's usual pattern by {format_currency(row['amount_deviation_abs'])}."
    if feature_name == "invoice_count":
        return f"Invoice `{row['invoice_id']}` already appeared {int(row['invoice_count'])} time(s) earlier in the ledger."
    if feature_name == "invoice_gap_days":
        return f"The same invoice appeared {row['invoice_gap_days']:.1f} day(s) before this payment."
    if feature_name == "fuzzy_invoice_match_count":
        return f"Auditr found {int(row['fuzzy_invoice_match_count'])} near-match invoice(s) for the same vendor."
    if feature_name == "max_invoice_similarity":
        return f"Closest earlier invoice ID has a similarity score of {row['max_invoice_similarity']:.2f}."
    if feature_name == "transactions_per_day_vendor":
        return f"This vendor already had {int(row['transactions_per_day_vendor'])} earlier payment(s) on the same day."
    if feature_name in {"vendor_day_running_total_prior", "vendor_day_running_total_with_current"}:
        return (
            f"Same-day payments to this vendor reached {format_currency(row['vendor_day_running_total_with_current'])} "
            "once this payment is included."
        )
    if feature_name == "is_just_below_approval":
        return "The amount sits just below a common approval threshold."
    if feature_name == "crosses_approval_with_same_day_vendor":
        return "Multiple same-day payments together cross a common approval threshold."
    if feature_name == "is_duplicate_invoice":
        return f"Invoice `{row['invoice_id']}` is an exact duplicate of an earlier invoice in the file."
    if feature_name == "is_round_number":
        return f"The amount {format_currency(row['amount'])} is unusually neat and rounded."
    if feature_name == "is_new_vendor_payment_method":
        return f"Vendor `{row['vendor']}` is being paid with a new payment method in this file."
    if feature_name == "is_new_vendor_department_pair":
        return f"Vendor `{row['vendor']}` appears in a new department pairing: {row['department']}."
    if feature_name == "days_since_vendor_last_payment":
        return f"The last payment to this vendor was {row['days_since_vendor_last_payment']:.1f} day(s) ago."
    if feature_name == "posting_hour":
        return f"The payment was posted at {int(row['posting_hour']):02d}:00."
    return "This signal contributed to the score based on how unusual the transaction looked against the ledger history."


def build_case_signal_table(row: pd.Series, top_contributors: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for feature_name, impact in top_contributors.items():
        rows.append(
            {
                "Signal": format_feature_label(feature_name),
                "Direction": "Raised risk" if impact >= 0 else "Lowered risk",
                "What Auditr noticed": describe_feature_observation(row, feature_name),
            }
        )
    return pd.DataFrame(rows)


def build_vendor_risk_snapshot(row: pd.Series, scored: pd.DataFrame, flagged: pd.DataFrame) -> dict[str, object]:
    vendor_rows = scored[scored["vendor"] == row["vendor"]].copy()
    vendor_flagged = flagged[flagged["vendor"] == row["vendor"]].copy()
    top_reason = (
        vendor_flagged["primary_driver_label"].value_counts().idxmax()
        if not vendor_flagged.empty
        else row["primary_driver_label"]
    )
    summary = (
        f"Vendor `{row['vendor']}` appears {vendor_rows.shape[0]:,} time(s) in this engagement. "
        f"{vendor_flagged.shape[0]:,} payment(s) were flagged, totaling {format_currency(vendor_flagged['amount'].sum() if not vendor_flagged.empty else 0.0)}. "
        f"The most common reason is {top_reason.lower()}."
    )
    evidence = []
    if int((vendor_rows["is_duplicate_invoice"] > 0).sum()) > 0:
        evidence.append(f"{int((vendor_rows['is_duplicate_invoice'] > 0).sum())} exact duplicate invoice hit(s)")
    if int((vendor_rows["fuzzy_invoice_match_count"] > 0).sum()) > 0:
        evidence.append(f"{int((vendor_rows['fuzzy_invoice_match_count'] > 0).sum())} near-duplicate invoice hit(s)")
    if int((vendor_rows["crosses_approval_with_same_day_vendor"] > 0).sum()) > 0:
        evidence.append(f"{int((vendor_rows['crosses_approval_with_same_day_vendor'] > 0).sum())} split-payment threshold event(s)")
    if int((vendor_rows["transactions_per_day_vendor"] > 0).sum()) > 0:
        evidence.append(f"{int((vendor_rows['transactions_per_day_vendor'] > 0).sum())} same-day repeat payment(s)")

    return {
        "summary": summary,
        "top_reason": top_reason,
        "vendor_rows": vendor_rows,
        "vendor_flagged": vendor_flagged,
        "evidence_text": "; ".join(evidence) if evidence else "This vendor stands out mainly because of the selected case rather than repeated control-pattern hits.",
    }


def build_vendor_case_evidence_table(vendor_flagged: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if vendor_flagged.empty:
        return pd.DataFrame(columns=["Transaction ID", "Date", "Invoice ID", "Amount", "Risk score", "Main reason"])
    table = vendor_flagged[
        [
            "transaction_id",
            "date",
            "invoice_id",
            "amount",
            "fraud_probability",
            "primary_driver_label",
        ]
    ].copy()
    table = table.sort_values("fraud_probability", ascending=False).head(limit)
    table.columns = ["Transaction ID", "Date", "Invoice ID", "Amount", "Risk score", "Main reason"]
    table["Date"] = table["Date"].map(lambda value: value.strftime("%Y-%m-%d %H:%M"))
    table["Amount"] = table["Amount"].map(format_currency)
    table["Risk score"] = table["Risk score"].map(lambda value: f"{value:.1%}")
    return table


def build_case_checklist(row: pd.Series, primary_driver: str) -> list[str]:
    if primary_driver in {"is_duplicate_invoice", "invoice_count", "invoice_gap_days", "fuzzy_invoice_match_count", "max_invoice_similarity"}:
        return [
            "Compare this invoice number against earlier transactions for the same vendor.",
            "Check whether the invoice date, amount, and support documents match a previously paid item.",
            "Confirm whether the duplicate or near-duplicate invoice was approved more than once.",
        ]
    if primary_driver in {"transactions_per_day_vendor", "vendor_day_running_total_prior", "vendor_day_running_total_with_current", "is_just_below_approval", "crosses_approval_with_same_day_vendor"}:
        return [
            "Review all same-day payments to this vendor together, not in isolation.",
            "Check whether the payments were split to stay below an approval limit.",
            "Confirm whether the approvals were meant to cover one payment or several separate ones.",
        ]
    if primary_driver in {"amount_to_vendor_avg_ratio", "amount_deviation_abs", "vendor_amount_zscore"}:
        return [
            "Compare the amount against the vendor's normal payment history.",
            "Check the contract, invoice, and approval note for a reason the amount is unusually high or low.",
            "Confirm the payment is coded to the correct vendor and department.",
        ]
    if primary_driver in {"is_new_vendor_payment_method", "is_new_vendor_department_pair", "is_new_vendor_account_type"}:
        return [
            "Check whether the vendor normally appears with this payment method, account type, or department.",
            "Confirm the master-data setup and supporting documents for this transaction.",
            "Make sure the transaction was not misrouted or coded to the wrong place.",
        ]
    return [
        "Review the invoice, contract, and approval trail together.",
        "Compare this transaction with earlier payments to the same vendor.",
        "Document whether the unusual pattern has a valid business explanation.",
    ]


def build_fraud_type_recall_table(scored: pd.DataFrame) -> pd.DataFrame:
    if "fraud_label" not in scored.columns or "fraud_type" not in scored.columns:
        return pd.DataFrame()

    fraud_only = scored[scored["fraud_label"].astype(int) == 1].copy()
    if fraud_only.empty:
        return pd.DataFrame()

    fraud_type_view = (
        fraud_only.groupby("fraud_type", as_index=False)
        .agg(
            cases=("transaction_id", "size"),
            detected=("fraud_prediction", "sum"),
            avg_score=("fraud_probability", "mean"),
        )
        .sort_values(["cases", "avg_score"], ascending=[False, False])
    )
    fraud_type_view["recall"] = fraud_type_view["detected"] / fraud_type_view["cases"].clip(lower=1)
    fraud_type_view.columns = ["Fraud type", "Cases", "Detected", "Average score", "Recall"]
    return fraud_type_view


def model_health_rows(metadata: dict[str, object]) -> list[dict[str, str]]:
    validation = metadata.get("validation_metrics", {}) or {}
    feature_names = metadata.get("feature_names", []) or []
    core_features = ", ".join(str(name) for name in feature_names[:8]) + ("..." if len(feature_names) > 8 else "")
    rows = [
        {"Metric": "Model version", "Value": str(metadata.get("model_version", "unknown"))},
        {"Metric": "Model type", "Value": "XGBoost classifier"},
        {"Metric": "Output", "Value": "Risk score used for prioritisation, not a final fraud verdict"},
        {"Metric": "Explanation method", "Value": "Contribution-based local explanations (per transaction feature impacts)"},
        {"Metric": "Feature highlights", "Value": core_features or "Not available"},
        {
            "Metric": "Limitations",
            "Value": "Score quality depends on ledger quality, column mapping, date parsing, amount parsing, and completeness.",
        },
        {"Metric": "Feature set", "Value": str(metadata.get("feature_set_version", "unknown"))},
        {"Metric": "Decision threshold", "Value": f"{float(metadata.get('decision_threshold', DEFAULT_DECISION_THRESHOLD)):.2f}"},
        {"Metric": "Validation strategy", "Value": str(metadata.get("validation_strategy", "unknown"))},
    ]
    if validation:
        rows.extend(
            [
                {"Metric": "Test precision", "Value": f"{float(validation.get('test_precision', 0.0)):.1%}"},
                {"Metric": "Test recall", "Value": f"{float(validation.get('test_recall', 0.0)):.1%}"},
                {"Metric": "Test F1", "Value": f"{float(validation.get('test_f1', 0.0)):.3f}"},
                {"Metric": "Test PR-AUC", "Value": f"{float(validation.get('test_pr_auc', 0.0)):.3f}"},
            ]
        )
    return rows


def dataframe_for_transactions(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored[
        [
            "transaction_id",
            "date",
            "vendor",
            "department",
            "payment_method",
            "employee",
            "invoice_id",
            "amount",
            "fraud_probability_pct",
            "anomaly_score",
            "blended_priority_score",
            "review_status",
            "reviewer_status",
            "primary_driver_label",
            "summary_title",
        ]
    ].copy()
    frame.columns = [
        "Transaction ID",
        "Date",
        "Vendor",
        "Department",
        "Payment Method",
        "Employee",
        "Invoice ID",
        "Amount",
        "Risk score (%)",
        "Anomaly score",
        "Priority score",
        "Review status",
        "Reviewer status",
        "Main reason",
        "Why Auditr flagged it",
    ]
    frame["Payment Method"] = frame["Payment Method"].map(friendly_payment_method)
    return frame.sort_values("Priority score", ascending=False)


def audit_report_dataframe(scored: pd.DataFrame) -> pd.DataFrame:
    report = scored.copy()
    report["risk_score_pct"] = report["fraud_probability_pct"].round(2)
    report = report.drop(columns=["fraud_probability_pct"])
    return report


def safe_dataframe_for_streamlit(dataframe: pd.DataFrame) -> pd.DataFrame:
    safe = dataframe.copy()
    for column in safe.columns:
        series = safe[column]
        if pd.api.types.is_object_dtype(series):
            observed_types = {type(value).__name__ for value in series.dropna().tolist()}
            if len(observed_types) > 1:
                safe[column] = series.map(lambda value: "" if pd.isna(value) else str(value))
            elif any(isinstance(value, (dict, list, tuple, set)) for value in series.dropna().head(50)):
                safe[column] = series.map(lambda value: "" if pd.isna(value) else str(value))
    return safe


def build_audit_memo_markdown(
    scored: pd.DataFrame,
    *,
    project_name: str | None,
    dataset_label: str | None,
) -> str:
    flagged = scored[scored["fraud_prediction"] == 1].copy()
    top_reasons = (
        flagged["primary_driver_label"].value_counts().head(5).to_dict() if not flagged.empty else {}
    )
    top_vendors = (
        flagged.groupby("vendor")["amount"].sum().sort_values(ascending=False).head(5).to_dict()
        if not flagged.empty
        else {}
    )
    top_cases = scored.nlargest(10, "blended_priority_score")[
        ["transaction_id", "vendor", "department", "amount", "fraud_probability", "anomaly_score", "primary_driver_label"]
    ].copy()

    lines = [
        "# Auditr Audit Memo",
        "",
        f"- Project: {project_name or 'N/A'}",
        f"- Dataset: {dataset_label or 'N/A'}",
        f"- Generated at (UTC): {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}",
        "",
        "## Executive Summary",
        f"- Total transactions reviewed: {len(scored):,}",
        f"- Flagged transactions: {len(flagged):,}",
        f"- Flagged rate: {len(flagged) / max(len(scored), 1):.1%}",
        "",
        "## Top Suspicious Reasons",
    ]
    if top_reasons:
        lines.extend([f"- {reason}: {count:,}" for reason, count in top_reasons.items()])
    else:
        lines.append("- No flagged reasons in the current dataset.")

    lines.extend(["", "## Top Risky Vendors (flagged amount)"])
    if top_vendors:
        lines.extend([f"- {vendor}: {format_currency(amount)}" for vendor, amount in top_vendors.items()])
    else:
        lines.append("- No flagged vendors in the current dataset.")

    lines.extend(["", "## Top 10 Priority Cases", "", "| Transaction ID | Vendor | Department | Amount | Model Risk | Anomaly | Main Reason |", "|---|---|---|---:|---:|---:|---|"])
    for _, row in top_cases.iterrows():
        lines.append(
            "| "
            + f"{row['transaction_id']} | {row['vendor']} | {row['department']} | "
            + f"{format_currency(float(row['amount']))} | {float(row['fraud_probability']):.1%} | {float(row['anomaly_score']):.1%} | "
            + f"{row['primary_driver_label']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- Auditr risk score is a prioritisation signal, not a final fraud verdict.",
            "- Reviewer outcomes (Cleared/Escalated) should be captured in Case Review and reflected in queue ordering.",
        ]
    )
    return "\n".join(lines)


def render_empty_state(message: str) -> None:
    st.markdown(notice_panel("Nothing to show yet", message), unsafe_allow_html=True)
