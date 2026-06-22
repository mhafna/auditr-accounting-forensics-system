from __future__ import annotations

import streamlit as st

from backend.utils import (
    current_analysis,
    current_dataset_label,
    current_model_metadata,
    current_project_name,
    current_theme_name,
    current_user_label,
    get_auth_config,
    glossary_card,
    hero_panel,
    logout_user,
    metric_card,
    session_age_label,
)


def render() -> None:
    analysis = current_analysis()
    auth = get_auth_config()
    model_meta = current_model_metadata()
    flagged_count = int(analysis["scored"]["fraud_prediction"].sum()) if analysis else 0
    project_label = current_project_name() or "No active project"

    st.markdown(
        hero_panel(
            "Workspace identity",
            "Profile and session status",
            (
                "This page makes the app feel like a finished product instead of a notebook wrapper. "
                "It shows who is signed in, what dataset is active, and how the local TOTP gate is currently configured."
            ),
        ),
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(4)
    stat_cols[0].markdown(
        metric_card("Signed in user", current_user_label(), "Current local account for this session.", "accent"),
        unsafe_allow_html=True,
    )
    stat_cols[1].markdown(
        metric_card("Session age", session_age_label(), "How long the current authenticated session has been open.", "amber"),
        unsafe_allow_html=True,
    )
    stat_cols[2].markdown(
        metric_card("Active project", project_label, f"Current source: {current_dataset_label()}.", "accent"),
        unsafe_allow_html=True,
    )
    stat_cols[3].markdown(
        metric_card("Review queue", f"{flagged_count:,}", "Transactions currently marked for manual review.", "danger"),
        unsafe_allow_html=True,
    )

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.subheader("Security profile")
        st.markdown(
            glossary_card(
                "TOTP authentication",
                (
                    "Auditr now uses a six-digit time-based one-time password before the workspace opens. "
                    "That means the app is no longer just a raw dashboard; it has a basic security story for presentation."
                ),
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            glossary_card(
                "Auth mode",
                (
                    "Demo mode is active because environment variables were not provided."
                    if auth["demo_mode"]
                    else "Production-style mode is active because both username and secret are coming from environment variables."
                ),
            ),
            unsafe_allow_html=True,
        )

    with detail_cols[1]:
        st.subheader("Workspace preferences")
        st.markdown(
            glossary_card(
                "Current theme",
                f"The selected visual theme is `{current_theme_name()}`. Switch it from Settings.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            glossary_card(
                "Model profile",
                (
                    f"Current model version is `{model_meta['model_version']}` with decision threshold "
                    f"`{float(model_meta['decision_threshold']):.2f}` and feature set `{model_meta['feature_set_version']}`."
                ),
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            glossary_card(
                "Presentation readiness",
                (
                    "The app now has a branded shell, clearer labels, focused charts, a dedicated profile page, "
                    "and a settings page instead of a raw sidebar."
                ),
            ),
            unsafe_allow_html=True,
        )

    st.subheader("Session control")
    st.caption("Use this if you want to return to the TOTP sign-in screen.")
    if st.button("Log out", key="profile_logout", width="stretch"):
        logout_user()
        st.rerun()
