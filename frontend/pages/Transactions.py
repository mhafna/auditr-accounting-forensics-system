from __future__ import annotations

import streamlit as st

from backend.utils import (
    audit_report_dataframe,
    current_analysis,
    dataframe_for_transactions,
    hero_panel,
    render_empty_state,
)


def render() -> None:
    analysis = current_analysis()
    if not analysis:
        render_empty_state("No ledger is loaded yet. Open Projects, load an engagement, then return here to search the ledger.")
        return

    scored = analysis["scored"]
    transaction_frame = dataframe_for_transactions(scored)

    st.markdown(
        hero_panel(
            "Ledger explorer",
            "Search the full ledger and isolate review cases",
            "Use the filters below to focus on the rows you want to inspect. The table is intentionally simplified so it reads like an audit workpaper, not a machine-learning debug view.",
        ),
        unsafe_allow_html=True,
    )

    filter_cols = st.columns(4)
    status_filter = filter_cols[0].selectbox(
        "Review status",
        options=["Needs review", "Looks routine"],
    )
    department_filter = filter_cols[1].selectbox(
        "Department",
        options=["All"] + sorted(transaction_frame["Department"].unique().tolist()),
    )
    payment_filter = filter_cols[2].selectbox(
        "Payment method",
        options=["All"] + sorted(transaction_frame["Payment Method"].unique().tolist()),
    )
    risk_threshold = filter_cols[3].slider(
        "Minimum risk score",
        min_value=0,
        max_value=100,
        value=st.session_state.get("review_threshold", 50),
        step=5,
        key="review_threshold",
    )

    search_text = st.text_input(
        "Search by transaction ID, vendor, employee, or invoice ID",
        placeholder="Example: TX-001232 or Vendor_073",
    ).strip()

    filtered = transaction_frame.copy()
    filtered = filtered[filtered["Review status"] == status_filter]
    if department_filter != "All":
        filtered = filtered[filtered["Department"] == department_filter]
    if payment_filter != "All":
        filtered = filtered[filtered["Payment Method"] == payment_filter]
    filtered = filtered[filtered["Risk score (%)"] >= risk_threshold]

    if search_text:
        matches = (
            filtered["Transaction ID"].str.contains(search_text, case=False, na=False)
            | filtered["Vendor"].str.contains(search_text, case=False, na=False)
            | filtered["Employee"].str.contains(search_text, case=False, na=False)
            | filtered["Invoice ID"].str.contains(search_text, case=False, na=False)
        )
        filtered = filtered[matches]

    st.caption(
        f"{filtered.shape[0]:,} rows match the current filters. "
        "Use the download button if you want the cleaned audit report with model outputs included."
    )

    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
        height=420,
        column_config={
            "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
            "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "Risk score (%)": st.column_config.ProgressColumn(
                "Risk score (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )

    report = audit_report_dataframe(scored)
    flagged_report = report[report["fraud_prediction"].astype(int) == 1].copy()
    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download cleaned audit report",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="auditr_audit_report.csv",
        mime="text/csv",
        width="stretch",
    )
    download_cols[1].download_button(
        "Download flagged cases only",
        data=flagged_report.to_csv(index=False).encode("utf-8"),
        file_name="auditr_flagged_cases.csv",
        mime="text/csv",
        width="stretch",
    )
