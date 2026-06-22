from __future__ import annotations

import streamlit as st

from backend.utils import (
    PROJECT_STATUS_OPTIONS,
    audit_projects_dataframe,
    create_audit_project,
    current_project_id,
    current_project_name,
    delete_audit_project,
    hero_panel,
    list_audit_projects,
    notice_panel,
    open_audit_project,
    project_browser_card,
    section_gap,
    set_view,
    update_audit_project,
)


SORT_OPTIONS = [
    "Recent activity",
    "Highest flagged",
    "Largest ledger",
    "Highest review rate",
]


def _filter_projects(
    projects: list[dict[str, object]],
    *,
    scope: str,
    search_query: str,
    sort_by: str,
) -> list[dict[str, object]]:
    filtered = list(projects)

    if scope == "Open":
        filtered = [project for project in filtered if project.get("status") != "Closed"]
    elif scope == "Closed":
        filtered = [project for project in filtered if project.get("status") == "Closed"]

    if search_query:
        query = search_query.lower()
        filtered = [
            project
            for project in filtered
            if query in str(project.get("name", "")).lower()
            or query in str(project.get("client", "")).lower()
            or query in str(project.get("source_file", "")).lower()
        ]

    if sort_by == "Highest flagged":
        filtered.sort(key=lambda project: (int(project.get("flagged_count", 0)), float(project.get("review_rate", 0.0))), reverse=True)
    elif sort_by == "Largest ledger":
        filtered.sort(key=lambda project: int(project.get("rows", 0)), reverse=True)
    elif sort_by == "Highest review rate":
        filtered.sort(key=lambda project: (float(project.get("review_rate", 0.0)), int(project.get("flagged_count", 0))), reverse=True)
    else:
        filtered.sort(key=lambda project: str(project.get("updated_at_utc", "")), reverse=True)

    return filtered


def _open_project(project_id: str, destination: str) -> None:
    with st.spinner("Opening project..."):
        open_audit_project(project_id)
    set_view(destination)
    st.rerun()


def render() -> None:
    projects = list_audit_projects()
    active_project_id = current_project_id()
    active_project_name = current_project_name()

    st.markdown(
        hero_panel(
            "Project library",
            "Browse audit engagements",
            (
                "Create new engagements, reopen older ones, and keep the rest of the workspace pointed at the correct ledger. "
                "This page is the project browser, not the analysis page."
            ),
        ),
        unsafe_allow_html=True,
    )

    with st.expander("Create new project", expanded=not projects):
        with st.form("projects_create_form"):
            project_name = st.text_input("Project name", placeholder="Example: Q2 Vendor Payments Review")
            client_name = st.text_input("Client / company", placeholder="Example: Horizon Trading LLC")
            status = st.selectbox("Project status", options=PROJECT_STATUS_OPTIONS, index=0)
            notes = st.text_area("Project notes", placeholder="Optional context for the engagement", height=96)
            uploaded_file = st.file_uploader("Project ledger CSV", type=["csv"], key="projects_upload")
            submitted = st.form_submit_button("Create project and analyze", width="stretch")

        if submitted:
            if not project_name.strip():
                st.error("Enter a project name before creating the audit project.")
            elif uploaded_file is None:
                st.error("Upload a ledger CSV before creating the audit project.")
            else:
                with st.spinner("Creating project and analyzing ledger..."):
                    created = create_audit_project(
                        project_name=project_name,
                        client_name=client_name,
                        raw_bytes=uploaded_file.getvalue(),
                        file_name=uploaded_file.name,
                        status=status,
                        notes=notes,
                    )
                st.session_state["projects_selected_id"] = created["id"]
                st.success("Project created and loaded into the workspace.")
                st.rerun()

    st.markdown(section_gap(), unsafe_allow_html=True)
    if active_project_name:
        st.markdown(
            notice_panel(
                "Current workspace project",
                (
                    f"{active_project_name} is loaded right now. "
                    "Dashboard, Transactions, and Explainability will all follow this engagement automatically."
                ),
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            notice_panel(
                "No active project yet",
                "Create a project or open one from the library below. The rest of Auditr will follow that engagement automatically.",
            ),
            unsafe_allow_html=True,
        )

    if not projects:
        st.markdown(section_gap(), unsafe_allow_html=True)
        st.markdown(
            notice_panel(
                "No saved projects yet",
                "Create the first engagement above and it will appear here for reopening, updating, or deletion.",
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(section_gap(), unsafe_allow_html=True)
    open_count = sum(1 for project in projects if project.get("status") != "Closed")
    closed_count = len(projects) - open_count
    if "projects_scope" not in st.session_state:
        st.session_state["projects_scope"] = "All"

    scope_counts = {
        "All": len(projects),
        "Open": open_count,
        "Closed": closed_count,
    }
    scope = st.segmented_control(
        "Project scope",
        options=["All", "Open", "Closed"],
        format_func=lambda option: f"{option} ({scope_counts[option]})",
        key="projects_scope",
        width="stretch",
    )
    if scope is None:
        scope = "All"

    filter_cols = st.columns([0.56, 0.22, 0.22], gap="medium")
    search_query = filter_cols[0].text_input("Search projects or clients", placeholder="Filter projects or clients").strip()
    sort_by = filter_cols[1].selectbox("Sort by", options=SORT_OPTIONS, index=0)
    filter_cols[2].download_button(
        "Export register",
        data=audit_projects_dataframe().to_csv(index=False).encode("utf-8"),
        file_name="auditr_project_register.csv",
        mime="text/csv",
        width="stretch",
    )

    filtered_projects = _filter_projects(
        projects,
        scope=scope,
        search_query=search_query,
        sort_by=sort_by,
    )

    if not filtered_projects:
        st.markdown(section_gap(), unsafe_allow_html=True)
        st.markdown(
            notice_panel(
                "No projects match this filter",
                "Try a different status view or search term to see matching engagements.",
            ),
            unsafe_allow_html=True,
        )
        return

    current_selected = st.session_state.get("projects_selected_id")
    valid_ids = {str(project.get("id")) for project in filtered_projects}
    if current_selected not in valid_ids:
        st.session_state["projects_selected_id"] = active_project_id if active_project_id in valid_ids else str(filtered_projects[0].get("id"))

    selected_project_id = str(st.session_state["projects_selected_id"])

    st.markdown(section_gap(), unsafe_allow_html=True)
    column_count = 1 if len(filtered_projects) == 1 else 2 if len(filtered_projects) == 2 else 3
    grid_columns = st.columns(column_count, gap="medium")
    for index, project in enumerate(filtered_projects):
        project_id = str(project.get("id"))
        is_selected = project_id == selected_project_id
        with grid_columns[index % column_count]:
            st.markdown(
                project_browser_card(
                    project,
                    active=project_id == active_project_id,
                    selected=is_selected,
                ),
                unsafe_allow_html=True,
            )
            card_actions = st.columns([0.5, 0.5], gap="small")
            load_label = "Active in workspace" if project_id == active_project_id else "Load project"
            if card_actions[0].button(load_label, key=f"projects_load_{project_id}", width="stretch", disabled=project_id == active_project_id):
                _open_project(project_id, "home")
            if is_selected:
                card_actions[1].caption("Selected for editing")
            elif card_actions[1].button("Edit project", key=f"projects_select_{project_id}", width="stretch"):
                st.session_state["projects_selected_id"] = project_id
                st.rerun()

    selected_project = next(
        project for project in filtered_projects if str(project.get("id")) == str(st.session_state["projects_selected_id"])
    )

    st.markdown(section_gap(), unsafe_allow_html=True)
    st.subheader("Edit selected project")

    edit_left, edit_right = st.columns([0.62, 0.38], gap="large")
    with edit_left:
        st.markdown(
            notice_panel(
                "Editing",
                (
                    f"{selected_project.get('name', '')} for {selected_project.get('client', '')}. "
                    "The card above already shows the screening totals. Use this panel only for status, notes, or deletion."
                ),
            ),
            unsafe_allow_html=True,
        )
        current_status = str(selected_project.get("status", PROJECT_STATUS_OPTIONS[0]))
        if current_status not in PROJECT_STATUS_OPTIONS:
            current_status = PROJECT_STATUS_OPTIONS[0]
        updated_status = st.selectbox(
            "Project status",
            options=PROJECT_STATUS_OPTIONS,
            index=PROJECT_STATUS_OPTIONS.index(current_status),
            key=f"projects_status_{selected_project_id}",
        )
        notes_value = st.text_area(
            "Project notes",
            value=str(selected_project.get("notes", "")),
            key=f"projects_notes_editor_{selected_project_id}",
            height=140,
        )
        if st.button("Save project details", key="projects_save_details", width="stretch"):
            update_audit_project(
                str(selected_project.get("id")),
                status=updated_status,
                notes=notes_value.strip(),
            )
            st.success("Project details updated.")
            st.rerun()

    with edit_right:
        meta_rows = [
            f"Created {str(selected_project.get('created_at_utc', ''))[:16].replace('T', ' ')}",
            f"Updated {str(selected_project.get('updated_at_utc', ''))[:16].replace('T', ' ')}",
            f"Status {selected_project.get('status', '')}",
            f"Source {selected_project.get('source_file', '')}",
        ]
        st.markdown(
            notice_panel(
                "Project details",
                "<br>".join(meta_rows),
            ),
            unsafe_allow_html=True,
        )
        with st.expander("Delete this project", expanded=False):
            confirm_delete = st.checkbox(
                "I understand this permanently removes the saved engagement.",
                key=f"projects_delete_confirm_{selected_project_id}",
            )
            if st.button("Delete project", key="projects_delete_project", width="stretch"):
                if not confirm_delete:
                    st.error("Tick the confirmation checkbox before deleting the project.")
                else:
                    delete_audit_project(str(selected_project.get("id")))
                    st.session_state.pop("projects_selected_id", None)
                    st.success("Project deleted.")
                    st.rerun()
