from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.utils import (
    COLUMN_ALIAS_EXAMPLES,
    REQUIRED_COLUMNS,
    SAMPLE_DATASETS,
    hero_panel,
    notice_panel,
)


def render() -> None:
    st.markdown(
        hero_panel(
            "Help and support",
            "How to prepare a ledger and use Auditr properly",
            (
                "This page is the quick guide for the final-year-project demo. "
                "It explains what Auditr does, what file structure works best, what payment terms mean, "
                "and what to do when a transaction is flagged."
            ),
        ),
        unsafe_allow_html=True,
    )

    intro_cols = st.columns(2, gap="medium")
    with intro_cols[0]:
        st.markdown(
            notice_panel(
                "What Auditr does",
                (
                    "Auditr screens an accounting ledger for unusual payment patterns. It highlights transactions that "
                    "look risky, groups them into a manual review queue, and summarises why they were flagged."
                ),
            ),
            unsafe_allow_html=True,
        )
    with intro_cols[1]:
        st.markdown(
            notice_panel(
                "What Auditr does not do",
                (
                    "A flagged case is not confirmed fraud. It means the transaction needs human review before it is cleared. "
                    "The auditor still checks the invoice, approval path, amount, and vendor history."
                ),
            ),
            unsafe_allow_html=True,
        )

    st.subheader("CSV template")
    template = pd.DataFrame(
        [
            {
                "transaction_id": "TX-000001",
                "date": "2024-01-15 09:30",
                "amount": 1250.00,
                "vendor": "Vendor_001",
                "department": "Finance",
                "account_type": "Operations",
                "payment_method": "ACH",
                "employee": "Emp_001",
                "invoice_id": "INV-001",
            }
        ],
        columns=REQUIRED_COLUMNS,
    )
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="auditr_ledger_template.csv",
        mime="text/csv",
    )
    st.caption("Auditr accepts these canonical column names. It also tries to recognise common aliases automatically.")

    alias_rows: list[dict[str, str]] = []
    for canonical, aliases in COLUMN_ALIAS_EXAMPLES.items():
        alias_rows.append(
            {
                "Auditr column": canonical,
                "Examples Auditr can usually recognise": ", ".join(aliases),
            }
        )
    st.dataframe(pd.DataFrame(alias_rows), width="stretch", hide_index=True, height=360)

    st.subheader("Payment method terms")
    payment_terms = pd.DataFrame(
        [
            {"Term": "ACH", "Meaning": "Bank transfer through an automated clearing system."},
            {"Term": "Wire", "Meaning": "Direct bank-to-bank transfer, often used for urgent or higher-value payments."},
            {"Term": "Card", "Meaning": "Company card payment."},
            {"Term": "Check", "Meaning": "Cheque / paper check payment."},
        ]
    )
    st.dataframe(payment_terms, width="stretch", hide_index=True)

    st.subheader("How to use Auditr")
    usage_cols = st.columns(3, gap="medium")
    usage_cols[0].markdown(
        notice_panel(
            "1. Create or open a project",
            "Use Projects to create the engagement or reopen an earlier one. Auditr will clean the CSV, map recognised aliases, and score the full file.",
        ),
        unsafe_allow_html=True,
    )
    usage_cols[1].markdown(
        notice_panel(
            "2. Open Dashboard",
            "Use Dashboard to see how many cases need manual review, what pattern is repeating, and which departments are affected.",
        ),
        unsafe_allow_html=True,
    )
    usage_cols[2].markdown(
        notice_panel(
            "3. Investigate",
            "Use Transactions for full-ledger search and Explainability for a case brief on one flagged transaction.",
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Troubleshooting")
    st.markdown(
        notice_panel(
            "If upload fails",
            (
                "Check that the file is a CSV, that the date and amount columns are valid, and that the ledger includes a vendor field. "
                "If your headings use different names, Auditr will try common aliases automatically."
            ),
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        notice_panel(
            "Demo files",
            f"Demo ledgers are still available in Settings for rehearsal only: {', '.join(SAMPLE_DATASETS.keys())}.",
        ),
        unsafe_allow_html=True,
    )
