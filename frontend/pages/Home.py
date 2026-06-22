from __future__ import annotations

import streamlit as st

from backend.utils import (
    build_audit_briefing,
    build_top_risk_table,
    build_vendor_watchlist_table,
    current_analysis,
    current_dataset_label,
    current_project_id,
    current_project_name,
    format_currency,
    get_project_record,
    hero_panel,
    list_audit_projects,
    metric_card,
    notice_panel,
    open_audit_project,
    project_browser_card,
    safe_dataframe_for_streamlit,
    section_gap,
    set_view,
)


def _open_project_from_home(project_id: str) -> None:
    with st.spinner("Opening project..."):
        open_audit_project(project_id)
    set_view("home")
    st.rerun()


def _render_recent_projects(projects: list[dict[str, object]], *, prefix: str) -> None:
    if not projects:
        return

    column_count = min(3, len(projects))
    project_columns = st.columns(column_count, gap="medium")
    for index, project in enumerate(projects[:3]):
        with project_columns[index % column_count]:
            st.markdown(project_browser_card(project), unsafe_allow_html=True)
            if st.button("Load project", key=f"{prefix}_load_{project['id']}", width="stretch"):
                _open_project_from_home(str(project["id"]))


def _render_empty_home(projects: list[dict[str, object]]) -> None:
    open_projects = sum(1 for project in projects if project.get("status") != "Closed")
    closed_projects = max(len(projects) - open_projects, 0)

    st.markdown(
        hero_panel(
            "Audit workspace",
            "No engagement loaded",
            (
                "Open an engagement from the project library and use Home as the control room for that audit."
            ),
        ),
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(3, gap="medium")
    stat_cols[0].markdown(
        metric_card("Saved engagements", f"{len(projects):,}", "Audit engagements ready to reopen or continue.", "accent"),
        unsafe_allow_html=True,
    )
    stat_cols[1].markdown(
        metric_card("Open audits", f"{open_projects:,}", "Engagements that are still active or waiting for review.", "amber"),
        unsafe_allow_html=True,
    )
    stat_cols[2].markdown(
        metric_card("Closed audits", f"{closed_projects:,}", "Engagements that have already been wrapped up.", "danger"),
        unsafe_allow_html=True,
    )

    st.markdown(
        notice_panel(
            "Start the next engagement",
            (
                "Projects is where the auditor creates, opens, and manages engagements. "
                "Once a project is loaded, Home becomes the engagement cockpit instead of a second project browser."
            ),
        ),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1.05rem;'></div>", unsafe_allow_html=True)
    if st.button("Open Projects", key="home_empty_projects", width="stretch"):
        set_view("projects")
        st.rerun()

    if projects:
        st.markdown(section_gap(), unsafe_allow_html=True)
        st.markdown("**Recent engagements**")
        _render_recent_projects(projects, prefix="home_empty_recent")


def render() -> None:
    analysis = current_analysis()
    project_id = current_project_id()
    project_label = current_project_name()
    has_active_project = analysis is not None and bool(project_label)
    projects = list_audit_projects()

    if not has_active_project:
        _render_empty_home(projects)
        return

    active_project = get_project_record(str(project_id)) if project_id else None
    project_status = str((active_project or {}).get("status", "In progress"))
    project_client = str((active_project or {}).get("client", "Unspecified client"))

    scored = analysis["scored"]
    flagged_rows = scored[scored["fraud_prediction"] == 1].copy()
    flagged = int(flagged_rows.shape[0])
    review_rate = flagged / max(scored.shape[0], 1)
    flagged_amount = float(flagged_rows["amount"].sum()) if not flagged_rows.empty else 0.0
    top_reason = flagged_rows["primary_driver_label"].value_counts().idxmax() if not flagged_rows.empty else "No flagged cases"
    top_department = flagged_rows["department"].value_counts().idxmax() if not flagged_rows.empty else "No concentration yet"
    top_vendor = flagged_rows["vendor"].value_counts().idxmax() if not flagged_rows.empty else "No concentration yet"
    briefing = build_audit_briefing(scored)

    st.markdown(
        hero_panel(
            "Active engagement",
            project_label,
            (
                f"{project_client} is loaded in the workspace. "
                "Use Home to understand what needs attention first, then move into the supporting pages only when the next review step is clear."
            ),
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            "<div style='display:flex; flex-wrap:wrap; gap:0.55rem; margin:0.15rem 0 0.9rem 0;'>"
            f"<span class='meta-chip'>Status {project_status}</span>"
            f"<span class='meta-chip'>Source {current_dataset_label()}</span>"
            f"<span class='meta-chip'>Client {project_client}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(4, gap="medium")
    stat_cols[0].markdown(
        metric_card("Transactions analyzed", f"{scored.shape[0]:,}", "Ledger rows screened in the active engagement.", "accent"),
        unsafe_allow_html=True,
    )
    stat_cols[1].markdown(
        metric_card("Flagged cases", f"{flagged:,}", "Transactions currently sitting in the manual review queue.", "danger"),
        unsafe_allow_html=True,
    )
    stat_cols[2].markdown(
        metric_card("Review rate", f"{review_rate:.1%}", "Share of the ledger pushed into manual review.", "amber"),
        unsafe_allow_html=True,
    )
    stat_cols[3].markdown(
        metric_card("Flagged value", format_currency(flagged_amount), "Total amount tied to the current flagged queue.", "accent"),
        unsafe_allow_html=True,
    )

    top_cols = st.columns([0.62, 0.38], gap="medium")
    with top_cols[0]:
        st.markdown(
            notice_panel(
                "Audit brief",
                f"{briefing['headline']} {briefing['body']}",
            ),
            unsafe_allow_html=True,
        )
        snapshot_cols = st.columns(3, gap="medium")
        snapshot_cols[0].markdown(notice_panel("Main reason", top_reason), unsafe_allow_html=True)
        snapshot_cols[1].markdown(notice_panel("Main department", top_department), unsafe_allow_html=True)
        snapshot_cols[2].markdown(notice_panel("Watch vendor", top_vendor), unsafe_allow_html=True)

    with top_cols[1]:
        st.markdown(
            notice_panel(
                "Next move",
                (
                    "Start with the queue below. If the pattern is repeated, open Transactions for the full ledger view. "
                    "Use Explainability only when the auditor needs a clean case memo for one flagged payment."
                ),
            ),
            unsafe_allow_html=True,
        )

    st.markdown(section_gap(), unsafe_allow_html=True)
    lower_cols = st.columns([0.56, 0.44], gap="medium")

    with lower_cols[0]:
        st.markdown("**Cases to open now**")
        top_cases = build_top_risk_table(scored, limit=6).copy()
        top_cases["Date"] = top_cases["Date"].map(lambda value: value.strftime("%Y-%m-%d %H:%M"))
        top_cases["Amount"] = top_cases["Amount"].map(format_currency)
        top_cases["Risk score"] = top_cases["Risk score"].map(lambda value: f"{value:.1%}")
        top_cases["Anomaly score"] = top_cases["Anomaly score"].map(lambda value: f"{float(value):.1%}")
        top_cases["Priority score"] = top_cases["Priority score"].map(lambda value: f"{float(value):.1%}")
        st.dataframe(
            safe_dataframe_for_streamlit(top_cases),
            width="stretch",
            hide_index=True,
            height=300,
        )

    with lower_cols[1]:
        st.markdown("**Vendors to review first**")
        watchlist = build_vendor_watchlist_table(scored, limit=5).copy()
        if watchlist.empty:
            st.markdown(
                notice_panel(
                    "No vendor watchlist",
                    "This engagement currently has no flagged transactions, so there is no vendor concentration to prioritise.",
                ),
                unsafe_allow_html=True,
            )
        else:
            watchlist["Average risk"] = watchlist["Average risk"].map(lambda value: f"{value:.1%}")
            watchlist["Flagged amount"] = watchlist["Flagged amount"].map(format_currency)
            st.dataframe(
                safe_dataframe_for_streamlit(watchlist),
                width="stretch",
                hide_index=True,
                height=300,
            )

    other_projects = [project for project in projects if str(project.get("id")) != str(project_id)]
    if other_projects:
        st.markdown(section_gap(), unsafe_allow_html=True)
        st.markdown("**Other engagements**")
        _render_recent_projects(other_projects, prefix="home_other_recent")
