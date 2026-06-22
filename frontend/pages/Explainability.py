from __future__ import annotations

import streamlit as st

from backend.utils import (
    build_case_checklist,
    build_case_signal_table,
    build_contribution_chart,
    build_vendor_case_evidence_table,
    build_vendor_risk_snapshot,
    current_analysis,
    explain_transaction,
    format_currency,
    friendly_payment_method,
    hero_panel,
    metric_card,
    notice_panel,
    plotly_chart_config,
    render_empty_state,
)


def render() -> None:
    analysis = current_analysis()
    if not analysis:
        render_empty_state("No ledger is loaded yet. Open Projects, load an engagement, then return here for case review.")
        return

    flagged = analysis["flagged"]
    if flagged.empty:
        render_empty_state("The current dataset does not contain any flagged transactions.")
        return

    st.markdown(
        hero_panel(
            "Case review",
            "Review one case",
            (
                "Pick a flagged case, review why it was pushed into the queue, and see what evidence the auditor should check next."
            ),
        ),
        unsafe_allow_html=True,
    )

    selector_cols = st.columns([0.30, 0.70], gap="medium")
    vendor_choice = selector_cols[0].selectbox(
        "Vendor",
        options=["All flagged vendors"] + sorted(flagged["vendor"].unique().tolist()),
        index=0,
    )
    filtered_flagged = flagged if vendor_choice == "All flagged vendors" else flagged[flagged["vendor"] == vendor_choice].copy()
    transaction_id = selector_cols[1].selectbox(
        "Pick case",
        options=filtered_flagged["transaction_id"].tolist(),
        format_func=lambda value: (
            f"{value} | "
            f"{filtered_flagged.loc[filtered_flagged['transaction_id'] == value, 'vendor'].iloc[0]} | "
            f"{filtered_flagged.loc[filtered_flagged['transaction_id'] == value, 'fraud_probability'].iloc[0]:.1%}"
        ),
    )
    st.caption(
        f"Showing {filtered_flagged.shape[0]:,} flagged case(s)"
        + (f" for vendor `{vendor_choice}`." if vendor_choice != "All flagged vendors" else " across all flagged vendors.")
    )

    explanation = explain_transaction(analysis, transaction_id)
    row = explanation["row"]
    summary = explanation["summary"]
    top_contributors = explanation["top_contributors"]
    checklist = build_case_checklist(row, explanation["positive_driver"])
    signal_table = build_case_signal_table(row, top_contributors)
    vendor_snapshot = build_vendor_risk_snapshot(row, analysis["scored"], flagged)
    vendor_evidence = build_vendor_case_evidence_table(vendor_snapshot["vendor_flagged"])

    stat_cols = st.columns(3)
    stat_cols[0].markdown(
        metric_card(
            "Risk score",
            f"{row['fraud_probability']:.1%}",
            "Higher means Auditr found more unusual patterns in this transaction.",
            "danger",
        ),
        unsafe_allow_html=True,
    )
    stat_cols[1].markdown(
        metric_card(
            "Main reason",
            row["primary_driver_label"],
            f"Department: {row['department']}",
            "amber",
        ),
        unsafe_allow_html=True,
    )
    stat_cols[2].markdown(
        metric_card(
            "Payment amount",
            format_currency(row["amount"]),
            f"Vendor historical average before this payment: {format_currency(row['vendor_avg_amount'])}",
            "accent",
        ),
        unsafe_allow_html=True,
    )

    top_cols = st.columns([0.58, 0.42], gap="medium")
    with top_cols[0]:
        st.markdown(
            notice_panel(
                "What Auditr noticed",
                f"<strong>{summary['title']}</strong><br>{summary['summary']}",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            notice_panel(
                "What the auditor should check next",
                "<br>".join(f"{index}. {item}" for index, item in enumerate(checklist, start=1)),
            ),
            unsafe_allow_html=True,
        )

    with top_cols[1]:
        st.subheader("Case facts")
        case_facts = [
            {"Field": "Transaction ID", "Value": row["transaction_id"]},
            {"Field": "Date", "Value": row["date"].strftime("%Y-%m-%d %H:%M")},
            {"Field": "Vendor", "Value": row["vendor"]},
            {"Field": "Department", "Value": row["department"]},
            {"Field": "Payment method", "Value": friendly_payment_method(row["payment_method"])},
            {"Field": "Invoice ID", "Value": row["invoice_id"]},
            {"Field": "Vendor average before payment", "Value": format_currency(row["vendor_avg_amount"])},
            {"Field": "Amount difference from vendor average", "Value": format_currency(row["amount_deviation_abs"])},
            {"Field": "Same-day vendor total", "Value": format_currency(row["vendor_day_running_total_with_current"])},
            {"Field": "Near-duplicate invoice matches", "Value": str(int(row["fuzzy_invoice_match_count"]))},
        ]
        st.dataframe(case_facts, width="stretch", hide_index=True, height=350)

    st.subheader("Signals behind this case")
    st.caption("These are the strongest signals that pushed this case upward or downward in the risk score.")
    st.dataframe(signal_table.astype(str), width="stretch", hide_index=True, height=280)

    vendor_cols = st.columns([0.44, 0.56], gap="medium")
    with vendor_cols[0]:
        st.markdown(
            notice_panel(
                "Why this vendor stands out",
                f"{vendor_snapshot['summary']}<br><br><strong>Evidence:</strong> {vendor_snapshot['evidence_text']}",
            ),
            unsafe_allow_html=True,
        )
        vendor_stat_cols = st.columns(3, gap="medium")
        vendor_stat_cols[0].markdown(
            metric_card(
                "Vendor payments",
                f"{vendor_snapshot['vendor_rows'].shape[0]:,}",
                "All payments to this vendor inside the current engagement.",
                "accent",
            ),
            unsafe_allow_html=True,
        )
        vendor_stat_cols[1].markdown(
            metric_card(
                "Flagged for vendor",
                f"{vendor_snapshot['vendor_flagged'].shape[0]:,}",
                "How many of this vendor's payments entered the review queue.",
                "danger",
            ),
            unsafe_allow_html=True,
        )
        vendor_stat_cols[2].markdown(
            metric_card(
                "Flagged amount",
                format_currency(vendor_snapshot["vendor_flagged"]["amount"].sum() if not vendor_snapshot["vendor_flagged"].empty else 0.0),
                "Total flagged amount tied to this vendor in the project.",
                "amber",
            ),
            unsafe_allow_html=True,
        )

    with vendor_cols[1]:
        st.subheader("Other flagged payments for this vendor")
        st.caption("Use this as supporting evidence when the auditor wants to know whether the selected case is isolated or repeated.")
        st.dataframe(vendor_evidence.astype(str), width="stretch", hide_index=True, height=290)

    with st.expander("Advanced model view"):
        st.caption("This section keeps the raw model-facing contribution view available without cluttering the main audit page.")
        st.plotly_chart(build_contribution_chart(top_contributors), width="stretch", theme=None, config=plotly_chart_config())

    flagged_overview = flagged[
        [
            "transaction_id",
            "date",
            "vendor",
            "department",
            "invoice_id",
            "amount",
            "fraud_probability",
            "primary_driver_label",
            "summary_title",
            "next_step",
        ]
    ].copy()
    flagged_overview.columns = [
        "Transaction ID",
        "Date",
        "Vendor",
        "Department",
        "Invoice ID",
        "Amount",
        "Risk score",
        "Main reason",
        "Why Auditr flagged it",
        "What the auditor should do next",
    ]
    flagged_overview["Date"] = flagged_overview["Date"].map(lambda value: value.strftime("%Y-%m-%d %H:%M"))
    flagged_overview["Amount"] = flagged_overview["Amount"].map(format_currency)
    flagged_overview["Risk score"] = flagged_overview["Risk score"].map(lambda value: f"{value:.1%}")

    action_cols = st.columns([0.68, 0.32])
    action_cols[0].subheader("All flagged cases")
    action_cols[1].download_button(
        "Download flagged case brief",
        data=flagged_overview.to_csv(index=False).encode("utf-8"),
        file_name="auditr_flagged_case_brief.csv",
        mime="text/csv",
        width="stretch",
    )
    st.dataframe(flagged_overview.astype(str), width="stretch", hide_index=True, height=420)
