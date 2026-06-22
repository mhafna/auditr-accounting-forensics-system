from __future__ import annotations

import streamlit as st

from backend.utils import (
    SAMPLE_DATASETS,
    THEMES,
    apply_theme_from_widget,
    current_analysis,
    current_dataset_label,
    current_model_metadata,
    format_manual_key,
    generate_totp,
    get_auth_config,
    hero_panel,
    load_sample_dataset,
    notice_panel,
    set_active_dataset,
    sync_theme_widget,
)


def render() -> None:
    auth = get_auth_config()
    analysis = current_analysis()
    model_meta = current_model_metadata()

    st.markdown(
        hero_panel(
            "Workspace controls",
            "Theme, data source, and security settings",
            (
                "This page replaces the old raw sidebar with a proper control panel. "
                "Use it to switch the visual theme, load a different ledger, and review how TOTP auth is configured."
            ),
        ),
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([0.55, 0.45])

    with left_col:
        st.subheader("Visual theme")
        sync_theme_widget("settings_theme_select")
        st.selectbox(
            "Choose how Auditr looks",
            options=list(THEMES.keys()),
            key="settings_theme_select",
            on_change=apply_theme_from_widget,
            args=("settings_theme_select",),
        )
        st.caption("Theme changes apply immediately and stay active after refresh.")

        st.subheader("Load a demo ledger")
        sample_name = st.selectbox("Available demo files", options=list(SAMPLE_DATASETS.keys()))
        if st.button("Switch to selected demo", key="switch_demo", width="stretch"):
            set_active_dataset(load_sample_dataset(sample_name), sample_name, f"sample:{sample_name}")
            st.rerun()

        st.markdown(
            notice_panel(
                "Primary project flow",
                "Real auditors should create or switch engagements from Projects. Settings keeps the demo data switch, but Projects is the place for real audit files.",
            ),
            unsafe_allow_html=True,
        )

        if analysis is not None:
            st.markdown(
                notice_panel(
                    "Current data source",
                    (
                        f"Active source: `{current_dataset_label()}`. "
                        f"Transactions in memory: {analysis['scored'].shape[0]:,}. "
                        f"Review queue size: {int(analysis['scored']['fraud_prediction'].sum()):,}."
                    ),
                ),
                unsafe_allow_html=True,
            )

    with right_col:
        st.subheader("Security")
        auth_mode_text = (
            "Demo mode is active because no environment variables were provided."
            if auth["demo_mode"]
            else "Production-style mode is active because the username and secret were supplied through environment variables."
        )
        st.markdown(notice_panel("Authentication mode", auth_mode_text), unsafe_allow_html=True)

        st.markdown(
            notice_panel(
                "TOTP details",
                (
                    f"Username: `{auth['username']}`. "
                    f"The TOTP code rotates every {auth['window_seconds']} seconds and requires a six-digit code."
                ),
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            notice_panel(
                "Model metadata",
                (
                    f"Version: `{model_meta['model_version']}`. "
                    f"Feature set: `{model_meta['feature_set_version']}`. "
                    f"Decision threshold: `{float(model_meta['decision_threshold']):.2f}`. "
                    f"Chronological holdout precision: `{float((model_meta.get('validation_metrics', {}) or {}).get('test_precision', 0.0)):.1%}`. "
                    f"Recall: `{float((model_meta.get('validation_metrics', {}) or {}).get('test_recall', 0.0)):.1%}`."
                ),
            ),
            unsafe_allow_html=True,
        )

        if auth["demo_mode"]:
            st.markdown(
                notice_panel(
                    "Manual key",
                    (
                        "Paste this into your authenticator app if you are not using a QR code. "
                        f"Demo manual key: `{format_manual_key(auth['secret'])}`."
                    ),
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                notice_panel(
                    "Current demo code",
                    (
                        "This is only shown because the app is running in local demo mode. "
                        f"Current rotating code: `{generate_totp(auth['secret'])}`."
                    ),
                ),
                unsafe_allow_html=True,
            )
