from __future__ import annotations

import streamlit as st

from backend.utils import (
    build_audit_briefing,
    build_control_signal_chart,
    build_department_chart,
    build_reason_chart,
    build_review_queue_export,
    build_vendor_exposure_chart,
    build_vendor_watchlist_table,
    current_analysis,
    current_dataset_label,
    format_currency,
    hero_panel,
    metric_card,
    notice_panel,
    plotly_chart_config,
    render_empty_state,
    section_gap,
    set_view,
)


def render() -> None:
    analysis = current_analysis()
    if not analysis:
        render_empty_state("No ledger is loaded yet. Start on Home, upload a company CSV, then return here for the overview.")
        return

    scored = analysis["scored"]
    flagged = analysis["flagged"]
    briefing = build_audit_briefing(scored)

    st.markdown(
        hero_panel(
            "Auditor overview",
            "Where the auditor should look first",
            (
                f"Current source: `{current_dataset_label()}`. "
                "This page is designed to answer only the practical review questions: how many items need attention, "
                "what pattern is repeating, which departments are affected, and what should be checked first."
            ),
        ),
        unsafe_allow_html=True,
    )

    fraud_rate = flagged.shape[0] / max(scored.shape[0], 1)
    stat_cols = st.columns(3)
    stat_cols[0].markdown(
        metric_card(
            "Transactions reviewed",
            f"{scored.shape[0]:,}",
            f"{scored['vendor'].nunique()} vendors appear in this ledger.",
            "accent",
        ),
        unsafe_allow_html=True,
    )
    stat_cols[1].markdown(
        metric_card(
            "Flagged as risky",
            f"{flagged.shape[0]:,}",
            "These are the transactions that should move into manual review.",
            "danger",
        ),
        unsafe_allow_html=True,
    )
    stat_cols[2].markdown(
        metric_card(
            "Review rate",
            f"{fraud_rate:.1%}",
            "Share of the ledger that Auditr believes should be checked by a human auditor.",
            "amber",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        notice_panel(
            "Audit briefing",
            f"{briefing['headline']} {briefing['body']}",
        ),
        unsafe_allow_html=True,
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Why flagged")
        st.caption("This highlights the main pattern driving the manual review queue.")
        st.markdown(section_gap(), unsafe_allow_html=True)
        st.plotly_chart(build_reason_chart(scored), width="stretch", theme=None, config=plotly_chart_config())
    with chart_cols[1]:
        st.subheader("Departments")
        st.caption("This shows where flagged transactions are concentrated across the business.")
        st.markdown(section_gap(), unsafe_allow_html=True)
        st.plotly_chart(build_department_chart(scored), width="stretch", theme=None, config=plotly_chart_config())

    extra_chart_cols = st.columns(2)
    with extra_chart_cols[0]:
        st.subheader("Vendor exposure")
        st.caption("These vendors carry the highest flagged dollar exposure in the current engagement.")
        st.markdown(section_gap(), unsafe_allow_html=True)
        st.plotly_chart(build_vendor_exposure_chart(scored), width="stretch", theme=None, config=plotly_chart_config())
    with extra_chart_cols[1]:
        st.subheader("Control signals")
        st.caption("These are the control-style patterns that appear most often inside the flagged queue.")
        st.markdown(section_gap(), unsafe_allow_html=True)
        st.plotly_chart(build_control_signal_chart(scored), width="stretch", theme=None, config=plotly_chart_config())

    queue_export = build_review_queue_export(flagged)
    queue_preview = queue_export.copy()
    queue_preview["Amount"] = queue_preview["Amount"].map(format_currency)
    queue_preview["Risk score (%)"] = queue_preview["Risk score (%)"].map(lambda value: f"{value:.1f}%")
    queue_preview["Date"] = queue_preview["Date"].map(lambda value: value.strftime("%Y-%m-%d %H:%M"))

    header_cols = st.columns([0.56, 0.22, 0.22])
    header_cols[0].subheader("Manual review queue")
    header_cols[1].download_button(
        "Download review queue",
        data=queue_export.to_csv(index=False).encode("utf-8"),
        file_name="auditr_manual_review_queue.csv",
        mime="text/csv",
        width="stretch",
    )
    if header_cols[2].button("Open Transactions", key="dashboard_to_transactions", width="stretch"):
        st.session_state["page_loading_label"] = "Transactions"
        set_view("transactions")
        st.rerun()

    st.caption("This queue contains the transactions Auditr believes should be reviewed manually. Scroll inside the table for more rows.")
    st.dataframe(queue_preview, width="stretch", hide_index=True, height=380)

    lower_cols = st.columns([0.56, 0.44], gap="medium")
    with lower_cols[0]:
        st.subheader("Vendors with repeated flags")
        st.caption("These vendors appear most often in the manual review queue.")
        vendor_watchlist = build_vendor_watchlist_table(scored)
        vendor_watchlist["Average risk"] = vendor_watchlist["Average risk"].map(lambda value: f"{value:.1%}")
        vendor_watchlist["Flagged amount"] = vendor_watchlist["Flagged amount"].map(format_currency)
        st.dataframe(vendor_watchlist, width="stretch", hide_index=True, height=320)

    with lower_cols[1]:
        st.subheader("Recommended next moves")
        st.markdown(
            notice_panel(
                "Start the review efficiently",
                (
                    "1. Begin with the top rows in the manual review queue.\n"
                    "2. Open Transactions if you need to search the full ledger for the same vendor or invoice.\n"
                    "3. Open Explainability when you want Auditr to summarise one flagged case in plain language."
                ).replace("\n", "<br>"),
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            notice_panel(
                "What Auditr means by manual review",
                (
                    "A flagged case is not confirmed fraud. It means the transaction looks unusual enough that the auditor "
                    "should compare the invoice, amount, approval path, and vendor history before clearing it."
                ),
            ),
            unsafe_allow_html=True,
        )
