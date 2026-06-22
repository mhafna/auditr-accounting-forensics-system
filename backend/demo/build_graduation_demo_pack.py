from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET = PROJECT_ROOT / "accounting_fraud_dataset_v2.csv"
OUTPUT_DIR = PROJECT_ROOT / "Demo Datasets"


def _save_csv(dataframe: pd.DataFrame, name: str) -> None:
    output_path = OUTPUT_DIR / name
    dataframe.to_csv(output_path, index=False)


def _to_euro_amount(value: float) -> str:
    return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _build_base_frame(source: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    return source.sample(sample_size, random_state=seed).sort_values("date").reset_index(drop=True)


def build_demo_pack() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_DATASET)
    source["date"] = pd.to_datetime(source["date"])

    files: list[Path] = []

    high_risk = _build_base_frame(source[source["fraud_label"] == 1], 450, 11)
    _save_csv(high_risk, "01_high_risk_focus.csv")
    files.append(OUTPUT_DIR / "01_high_risk_focus.csv")

    subtle_mix = _build_base_frame(source, 600, 17)
    subtle_mix["amount"] = subtle_mix["amount"].round(2)
    _save_csv(subtle_mix, "02_subtle_mixed_risk.csv")
    files.append(OUTPUT_DIR / "02_subtle_mixed_risk.csv")

    vendor_burst = _build_base_frame(source[source["vendor"].isin(source["vendor"].value_counts().head(8).index)], 700, 23)
    burst_day = pd.Timestamp("2024-03-20")
    vendor_burst.loc[:180, "date"] = burst_day + pd.to_timedelta(np.random.default_rng(2).integers(0, 6, 181), unit="h")
    _save_csv(vendor_burst, "03_vendor_burst_same_day.csv")
    files.append(OUTPUT_DIR / "03_vendor_burst_same_day.csv")

    split_threshold = _build_base_frame(source, 650, 29)
    split_threshold.loc[:140, "amount"] = np.random.default_rng(4).choice([2485.0, 2490.0, 4990.0, 9990.0], 141)
    _save_csv(split_threshold, "04_split_threshold_pattern.csv")
    files.append(OUTPUT_DIR / "04_split_threshold_pattern.csv")

    weekend_after_hours = _build_base_frame(source, 620, 31)
    weekend_after_hours.loc[:180, "date"] = pd.to_datetime("2024-02-10 21:00:00") + pd.to_timedelta(
        np.random.default_rng(8).integers(0, 72, 181), unit="h"
    )
    _save_csv(weekend_after_hours, "05_weekend_after_hours.csv")
    files.append(OUTPUT_DIR / "05_weekend_after_hours.csv")

    locale_amounts = _build_base_frame(source, 500, 37)
    locale_amounts["amount"] = locale_amounts["amount"].map(_to_euro_amount)
    locale_amounts["date"] = pd.to_datetime(locale_amounts["date"]).dt.strftime("%d/%m/%Y %H:%M")
    _save_csv(locale_amounts, "06_locale_amounts_dates.csv")
    files.append(OUTPUT_DIR / "06_locale_amounts_dates.csv")

    alias_headers = _build_base_frame(source, 560, 41)
    alias_headers = alias_headers.rename(
        columns={
            "transaction_id": "Transaction Number",
            "date": "Posting Date",
            "amount": "Payment Amount",
            "vendor": "Supplier Name",
            "department": "Cost Center",
            "account_type": "Expense Category",
            "payment_method": "Payment Mode",
            "employee": "Prepared By",
            "invoice_id": "Invoice Number",
        }
    )
    _save_csv(alias_headers, "07_alias_headers_mapping.csv")
    files.append(OUTPUT_DIR / "07_alias_headers_mapping.csv")

    quarantine_demo = _build_base_frame(source, 540, 43)
    quarantine_demo["amount"] = quarantine_demo["amount"].astype(object)
    quarantine_demo["date"] = quarantine_demo["date"].astype(object)
    quarantine_demo.loc[:35, "amount"] = "INVALID_AMOUNT"
    quarantine_demo.loc[36:70, "date"] = "31/02/2024 99:99"
    quarantine_demo.loc[71:95, "vendor"] = ""
    _save_csv(quarantine_demo, "08_quarantine_bad_rows.csv")
    files.append(OUTPUT_DIR / "08_quarantine_bad_rows.csv")

    duplicate_headers = _build_base_frame(source, 520, 47).copy()
    duplicate_headers.columns = [
        "transaction_id",
        "date",
        "amount",
        "vendor",
        "department",
        "department",
        "payment_method",
        "employee",
        "invoice_id",
        "fraud_label",
        "fraud_type",
    ]
    _save_csv(duplicate_headers, "09_duplicate_headers.csv")
    files.append(OUTPUT_DIR / "09_duplicate_headers.csv")

    executive_pack = _build_base_frame(source, 900, 53)
    executive_pack = executive_pack.sort_values(["fraud_label", "amount"], ascending=[False, False]).reset_index(drop=True)
    _save_csv(executive_pack, "10_executive_mix.csv")
    files.append(OUTPUT_DIR / "10_executive_mix.csv")

    return files


if __name__ == "__main__":
    created = build_demo_pack()
    print(f"Created {len(created)} demo ledgers:")
    for path in created:
        print(f"- {path.name}")
