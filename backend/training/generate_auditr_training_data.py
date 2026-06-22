from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEPARTMENTS = [
    "Finance",
    "HR",
    "IT",
    "Marketing",
    "Operations",
    "Legal",
    "Sales",
    "Support",
]
ACCOUNT_TYPES = ["Consulting", "IT", "Operations", "Payroll", "Supplies", "Travel"]
PAYMENT_METHODS = ["ACH", "Wire", "Card", "Check"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class VendorProfile:
    vendor: str
    home_department: str
    allowed_departments: list[str]
    allowed_account_types: list[str]
    preferred_methods: list[str]
    payment_method_weights: list[float]
    base_mean: float
    base_std: float
    approval_threshold: float
    activity_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a cleaner synthetic fraud training dataset for Auditr.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "accounting_fraud_dataset_v2.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--rows", type=int, default=6000, help="Target number of rows to generate.")
    parser.add_argument("--fraud-rate", type=float, default=0.12, help="Approximate fraud rate for the dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def create_vendor_profiles(rng: np.random.Generator, count: int = 120) -> dict[str, VendorProfile]:
    profiles: dict[str, VendorProfile] = {}
    for index in range(1, count + 1):
        vendor = f"Vendor_{index:03d}"
        home_department = str(rng.choice(DEPARTMENTS))
        other_departments = [value for value in DEPARTMENTS if value != home_department]
        extra_count = int(rng.integers(0, 3))
        extra_departments = rng.choice(other_departments, size=extra_count, replace=False).tolist() if extra_count else []
        allowed_departments = [home_department, *extra_departments]

        account_count = int(rng.integers(1, 4))
        allowed_account_types = rng.choice(ACCOUNT_TYPES, size=account_count, replace=False).tolist()

        preferred_methods = rng.choice(PAYMENT_METHODS, size=2, replace=False).tolist()
        method_weights = np.array([0.12, 0.10, 0.10, 0.08], dtype=float)
        method_weights[PAYMENT_METHODS.index(preferred_methods[0])] = 0.56
        method_weights[PAYMENT_METHODS.index(preferred_methods[1])] = 0.22
        method_weights = (method_weights / method_weights.sum()).tolist()

        base_mean = float(np.clip(rng.lognormal(mean=np.log(2200), sigma=0.62), 220.0, 12000.0))
        base_std = float(max(base_mean * rng.uniform(0.10, 0.34), 45.0))
        approval_threshold = float(rng.choice([2500.0, 5000.0, 10000.0], p=[0.35, 0.45, 0.20]))
        activity_weight = float(rng.uniform(0.35, 2.30))

        profiles[vendor] = VendorProfile(
            vendor=vendor,
            home_department=home_department,
            allowed_departments=allowed_departments,
            allowed_account_types=allowed_account_types,
            preferred_methods=preferred_methods,
            payment_method_weights=method_weights,
            base_mean=base_mean,
            base_std=base_std,
            approval_threshold=approval_threshold,
            activity_weight=activity_weight,
        )
    return profiles


def create_employees() -> tuple[list[str], dict[str, list[str]]]:
    employees_by_department: dict[str, list[str]] = {department: [] for department in DEPARTMENTS}
    employee_names: list[str] = []
    employee_index = 1
    for department in DEPARTMENTS:
        for _ in range(15):
            employee = f"Emp_{employee_index:03d}"
            employee_names.append(employee)
            employees_by_department[department].append(employee)
            employee_index += 1
    return employee_names, employees_by_department


def sample_business_hour(rng: np.random.Generator) -> int:
    hours = np.arange(7, 19)
    weights = np.array([0.03, 0.05, 0.08, 0.12, 0.13, 0.14, 0.13, 0.11, 0.09, 0.06, 0.04, 0.02])
    return int(rng.choice(hours, p=weights / weights.sum()))


def sample_riskier_hour(rng: np.random.Generator) -> int:
    hours = np.array([0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 8, 9, 18, 19])
    weights = np.array([0.08, 0.08, 0.08, 0.07, 0.07, 0.06, 0.05, 0.08, 0.08, 0.08, 0.08, 0.05, 0.04, 0.05, 0.05])
    return int(rng.choice(hours, p=weights / weights.sum()))


def clamp_timestamp(timestamp: pd.Timestamp, max_timestamp: pd.Timestamp) -> pd.Timestamp:
    if timestamp > max_timestamp:
        return max_timestamp - pd.Timedelta(minutes=1)
    return timestamp


def next_invoice_id(counter: list[int]) -> str:
    counter[0] += 1
    return f"INV-2024-{counter[0]:07d}"


def normal_amount(profile: VendorProfile, rng: np.random.Generator) -> float:
    amount = float(np.clip(rng.normal(profile.base_mean, profile.base_std), 25.0, profile.base_mean * 3.2))
    if rng.random() < 0.02:
        amount = round(round(amount / 50.0) * 50.0, 2)
    return round(amount, 2)


def build_normal_ledger(
    row_count: int,
    rng: np.random.Generator,
    profiles: dict[str, VendorProfile],
    employees_by_department: dict[str, list[str]],
) -> list[dict[str, object]]:
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-06-30 23:59:00")
    days = pd.date_range(start_date, end_date, freq="D")
    vendor_names = list(profiles.keys())
    vendor_weights = np.array([profiles[vendor].activity_weight for vendor in vendor_names], dtype=float)
    vendor_weights = vendor_weights / vendor_weights.sum()
    invoice_counter = [0]
    records: list[dict[str, object]] = []

    for day in days:
        if len(records) >= row_count:
            break
        expected = 33 if day.weekday() < 5 else 16
        transactions_today = max(8, int(rng.poisson(expected)))
        for _ in range(transactions_today):
            if len(records) >= row_count:
                break

            vendor = str(rng.choice(vendor_names, p=vendor_weights))
            profile = profiles[vendor]
            if rng.random() < 0.93:
                department = profile.home_department if rng.random() < 0.72 else str(rng.choice(profile.allowed_departments))
            else:
                department = str(rng.choice(DEPARTMENTS))

            employee = str(rng.choice(employees_by_department[department]))
            if rng.random() < 0.90:
                account_type = str(rng.choice(profile.allowed_account_types))
            else:
                account_type = str(rng.choice(ACCOUNT_TYPES))
            payment_method = str(rng.choice(PAYMENT_METHODS, p=profile.payment_method_weights))

            timestamp = day + pd.Timedelta(
                hours=sample_business_hour(rng),
                minutes=int(rng.integers(0, 60)),
            )
            amount = normal_amount(profile, rng)
            records.append(
                {
                    "date": timestamp,
                    "amount": amount,
                    "vendor": vendor,
                    "department": department,
                    "account_type": account_type,
                    "payment_method": payment_method,
                    "employee": employee,
                    "invoice_id": next_invoice_id(invoice_counter),
                    "fraud_label": 0,
                    "fraud_type": "normal",
                }
            )

    while len(records) < row_count:
        day = pd.Timestamp(rng.choice(days))
        vendor = str(rng.choice(vendor_names, p=vendor_weights))
        profile = profiles[vendor]
        department = profile.home_department if rng.random() < 0.75 else str(rng.choice(profile.allowed_departments))
        employee = str(rng.choice(employees_by_department[department]))
        account_type = str(rng.choice(profile.allowed_account_types))
        payment_method = str(rng.choice(PAYMENT_METHODS, p=profile.payment_method_weights))
        timestamp = day + pd.Timedelta(
            hours=sample_business_hour(rng),
            minutes=int(rng.integers(0, 60)),
        )
        amount = normal_amount(profile, rng)
        records.append(
            {
                "date": timestamp,
                "amount": amount,
                "vendor": vendor,
                "department": department,
                "account_type": account_type,
                "payment_method": payment_method,
                "employee": employee,
                "invoice_id": next_invoice_id(invoice_counter),
                "fraud_label": 0,
                "fraud_type": "normal",
            }
        )

    return records


def fraud_reference_frame(records: list[dict[str, object]], profiles: dict[str, VendorProfile]) -> pd.DataFrame:
    frame = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    frame["vendor_prior_count"] = frame.groupby("vendor").cumcount()
    frame["vendor_day_prior_count"] = frame.groupby(["vendor", frame["date"].dt.date]).cumcount()
    frame["department_history"] = frame.groupby(["vendor", "department"]).cumcount()
    frame["method_history"] = frame.groupby(["vendor", "payment_method"]).cumcount()
    frame["account_history"] = frame.groupby(["vendor", "account_type"]).cumcount()
    frame["profile_threshold"] = frame["vendor"].map(lambda value: profiles[value].approval_threshold)
    return frame


def choose_unseen_department(profile: VendorProfile, current_department: str, rng: np.random.Generator) -> str:
    choices = [value for value in DEPARTMENTS if value not in profile.allowed_departments and value != current_department]
    if not choices:
        choices = [value for value in DEPARTMENTS if value != current_department]
    return str(rng.choice(choices))


def choose_unusual_method(profile: VendorProfile, current_method: str, rng: np.random.Generator) -> str:
    choices = [value for value in PAYMENT_METHODS if value not in profile.preferred_methods and value != current_method]
    if not choices:
        choices = [value for value in PAYMENT_METHODS if value != current_method]
    return str(rng.choice(choices))


def mutate_invoice_id(invoice_id: str, rng: np.random.Generator) -> str:
    characters = list(str(invoice_id))
    mutable_positions = [index for index, character in enumerate(characters) if character.isdigit()]
    if not mutable_positions:
        mutable_positions = [index for index, character in enumerate(characters) if character.isalpha()]
    if not mutable_positions:
        return f"{invoice_id}A"

    edit_count = int(rng.integers(1, 3))
    for _ in range(edit_count):
        position = int(rng.choice(mutable_positions))
        current = characters[position]
        if current.isdigit():
            replacement_choices = [value for value in "0123456789" if value != current]
        else:
            replacement_choices = [value for value in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if value != current.upper()]
        characters[position] = str(rng.choice(replacement_choices))
    return "".join(characters)


def make_fraud_record(
    fraud_type: str,
    reference: pd.Series,
    rng: np.random.Generator,
    profiles: dict[str, VendorProfile],
    employees_by_department: dict[str, list[str]],
    invoice_counter: list[int],
    max_timestamp: pd.Timestamp,
) -> dict[str, object]:
    profile = profiles[str(reference["vendor"])]
    base_timestamp = pd.Timestamp(reference["date"])
    date = base_timestamp
    amount = float(reference["amount"])
    vendor = str(reference["vendor"])
    department = str(reference["department"])
    account_type = str(reference["account_type"])
    payment_method = str(reference["payment_method"])
    employee = str(reference["employee"])
    invoice_id = str(reference["invoice_id"])

    if fraud_type == "duplicate_invoice":
        date = clamp_timestamp(base_timestamp + pd.Timedelta(hours=int(rng.integers(2, 72))), max_timestamp)
        amount = round(float(reference["amount"]) * rng.uniform(0.98, 1.03), 2)
        payment_method = choose_unusual_method(profile, payment_method, rng) if rng.random() < 0.55 else payment_method
        employee = str(rng.choice(employees_by_department[department]))
    elif fraud_type == "fuzzy_duplicate":
        date = clamp_timestamp(base_timestamp + pd.Timedelta(hours=int(rng.integers(2, 60))), max_timestamp)
        amount = round(float(reference["amount"]) * rng.uniform(0.99, 1.02), 2)
        invoice_id = mutate_invoice_id(invoice_id, rng)
        payment_method = choose_unusual_method(profile, payment_method, rng) if rng.random() < 0.45 else payment_method
        employee = str(rng.choice(employees_by_department[department]))
    elif fraud_type == "high_amount":
        date = clamp_timestamp(
            base_timestamp + pd.Timedelta(days=int(rng.integers(1, 12)), hours=sample_riskier_hour(rng), minutes=int(rng.integers(0, 60))),
            max_timestamp,
        )
        amount = round(max(profile.base_mean * rng.uniform(3.2, 6.1), float(reference["amount"]) * rng.uniform(2.0, 4.5)), 2)
        payment_method = choose_unusual_method(profile, payment_method, rng)
        invoice_id = next_invoice_id(invoice_counter)
    elif fraud_type == "same_day_burst":
        late_hour = min(23, max(base_timestamp.hour + int(rng.integers(1, 6)), sample_riskier_hour(rng)))
        date = clamp_timestamp(
            pd.Timestamp(base_timestamp.date()) + pd.Timedelta(hours=late_hour, minutes=int(rng.integers(0, 60))),
            max_timestamp,
        )
        amount = round(float(reference["amount"]) * rng.uniform(0.82, 1.25), 2)
        invoice_id = next_invoice_id(invoice_counter)
    elif fraud_type == "split_payment":
        late_hour = min(23, max(base_timestamp.hour + int(rng.integers(1, 4)), sample_business_hour(rng)))
        date = clamp_timestamp(
            pd.Timestamp(base_timestamp.date()) + pd.Timedelta(hours=late_hour, minutes=int(rng.integers(0, 60))),
            max_timestamp,
        )
        amount = round((profile.approval_threshold - rng.uniform(25.0, 240.0)) / 10.0) * 10.0
        amount = round(max(amount, 150.0), 2)
        payment_method = choose_unusual_method(profile, payment_method, rng)
        invoice_id = next_invoice_id(invoice_counter)
    elif fraud_type == "vendor_department_shift":
        date = clamp_timestamp(
            base_timestamp + pd.Timedelta(days=int(rng.integers(1, 18)), hours=sample_riskier_hour(rng), minutes=int(rng.integers(0, 60))),
            max_timestamp,
        )
        department = choose_unseen_department(profile, department, rng)
        employee = str(rng.choice(employees_by_department[department]))
        amount = round(float(reference["amount"]) * rng.uniform(1.35, 2.40), 2)
        account_type = str(rng.choice([value for value in ACCOUNT_TYPES if value not in profile.allowed_account_types] or ACCOUNT_TYPES))
        payment_method = choose_unusual_method(profile, payment_method, rng) if rng.random() < 0.6 else payment_method
        invoice_id = next_invoice_id(invoice_counter)
    elif fraud_type == "payment_method_switch":
        date = clamp_timestamp(
            base_timestamp + pd.Timedelta(days=int(rng.integers(1, 10)), hours=sample_riskier_hour(rng), minutes=int(rng.integers(0, 60))),
            max_timestamp,
        )
        payment_method = choose_unusual_method(profile, payment_method, rng)
        amount = round(max(float(reference["amount"]) * rng.uniform(1.35, 2.35), profile.base_mean * 1.4), 2)
        account_type = str(rng.choice([value for value in ACCOUNT_TYPES if value not in profile.allowed_account_types] or ACCOUNT_TYPES))
        invoice_id = next_invoice_id(invoice_counter)
    else:
        date = clamp_timestamp(
            base_timestamp + pd.Timedelta(hours=int(rng.integers(1, 40))),
            max_timestamp,
        )
        amount = round(max(profile.base_mean * rng.uniform(2.4, 4.6), float(reference["amount"]) * rng.uniform(1.8, 3.2)), 2)
        payment_method = choose_unusual_method(profile, payment_method, rng)
        invoice_id = next_invoice_id(invoice_counter)

    return {
        "date": date,
        "amount": amount,
        "vendor": vendor,
        "department": department,
        "account_type": account_type,
        "payment_method": payment_method,
        "employee": employee,
        "invoice_id": invoice_id,
        "fraud_label": 1,
        "fraud_type": fraud_type,
    }


def inject_fraud_records(
    normal_records: list[dict[str, object]],
    fraud_count: int,
    rng: np.random.Generator,
    profiles: dict[str, VendorProfile],
    employees_by_department: dict[str, list[str]],
) -> list[dict[str, object]]:
    frame = fraud_reference_frame(normal_records, profiles)
    max_timestamp = pd.Timestamp("2024-06-30 23:59:00")
    invoice_counter = [int(frame["invoice_id"].str.extract(r"(\d+)$").astype(int).max().iloc[0])]
    fraud_records: list[dict[str, object]] = []
    fraud_types = [
        "duplicate_invoice",
        "fuzzy_duplicate",
        "high_amount",
        "same_day_burst",
        "split_payment",
        "vendor_department_shift",
        "payment_method_switch",
        "low_history_high_amount",
    ]
    fraud_weights = np.array([0.20, 0.08, 0.19, 0.15, 0.14, 0.12, 0.07, 0.05], dtype=float)
    fraud_weights = fraud_weights / fraud_weights.sum()

    low_history_candidates = frame[frame["vendor_prior_count"] <= 1]
    same_day_candidates = frame[frame["vendor_day_prior_count"] >= 1]
    mature_vendor_candidates = frame[frame["vendor_prior_count"] >= 3]
    department_shift_candidates = frame[frame["department_history"] >= 2]
    method_switch_candidates = frame[frame["method_history"] >= 1]

    for _ in range(fraud_count):
        fraud_type = str(rng.choice(fraud_types, p=fraud_weights))
        if fraud_type in {"duplicate_invoice", "fuzzy_duplicate"}:
            reference_pool = frame
        elif fraud_type == "same_day_burst":
            reference_pool = same_day_candidates if not same_day_candidates.empty else mature_vendor_candidates
        elif fraud_type == "split_payment":
            reference_pool = mature_vendor_candidates
        elif fraud_type == "vendor_department_shift":
            reference_pool = department_shift_candidates if not department_shift_candidates.empty else mature_vendor_candidates
        elif fraud_type == "payment_method_switch":
            reference_pool = method_switch_candidates if not method_switch_candidates.empty else mature_vendor_candidates
        elif fraud_type == "low_history_high_amount":
            reference_pool = low_history_candidates if not low_history_candidates.empty else frame.head(max(50, frame.shape[0] // 5))
        else:
            reference_pool = mature_vendor_candidates if not mature_vendor_candidates.empty else frame

        reference = reference_pool.sample(n=1, random_state=int(rng.integers(0, 2_000_000_000))).iloc[0]
        fraud_records.append(
            make_fraud_record(
                fraud_type=fraud_type,
                reference=reference,
                rng=rng,
                profiles=profiles,
                employees_by_department=employees_by_department,
                invoice_counter=invoice_counter,
                max_timestamp=max_timestamp,
            )
        )

    return fraud_records


def finalize_dataset(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    frame.insert(0, "transaction_id", [f"TX-{index:06d}" for index in range(1, frame.shape[0] + 1)])
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    ordered_columns = [
        "transaction_id",
        "date",
        "amount",
        "vendor",
        "department",
        "account_type",
        "payment_method",
        "employee",
        "invoice_id",
        "fraud_label",
        "fraud_type",
    ]
    return frame[ordered_columns]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    rng = np.random.default_rng(args.seed)

    target_rows = max(2000, int(args.rows))
    fraud_rows = max(120, int(round(target_rows * args.fraud_rate)))
    normal_rows = target_rows - fraud_rows

    profiles = create_vendor_profiles(rng)
    _employees, employees_by_department = create_employees()
    normal_records = build_normal_ledger(normal_rows, rng, profiles, employees_by_department)
    fraud_records = inject_fraud_records(normal_records, fraud_rows, rng, profiles, employees_by_department)
    dataset = finalize_dataset([*normal_records, *fraud_records])

    output_path.write_text(dataset.to_csv(index=False), encoding="utf-8")

    fraud_counts = Counter(dataset["fraud_type"])
    print(f"Saved dataset to {output_path}")
    print(f"Rows: {dataset.shape[0]:,}")
    print(f"Fraud rate: {dataset['fraud_label'].mean():.2%}")
    print("Fraud types:", dict(sorted(fraud_counts.items())))


if __name__ == "__main__":
    main()
