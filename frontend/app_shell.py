from __future__ import annotations

import streamlit as st

from backend.utils import (
    PAGE_META,
    current_page,
    ensure_app_state,
    inject_app_styles,
    render_auth_gate,
    set_view,
    shell_brand,
    sync_client_persistence,
)
from frontend.pages import Dashboard, Explainability, Home, Projects, Transactions


def run_app() -> None:
    st.set_page_config(
        page_title="Auditr",
        page_icon="A",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    ensure_app_state()
    inject_app_styles()
    sync_client_persistence()
    render_auth_gate()

    nav_routes = {
        "home": Home.render,
        "projects": Projects.render,
        "dashboard": Dashboard.render,
        "transactions": Transactions.render,
        "explainability": Explainability.render,
    }

    if current_page() not in nav_routes:
        set_view("home")

    def open_page(view_name: str) -> None:
        st.session_state["page_loading_label"] = PAGE_META.get(view_name, {}).get("label", "Page")
        set_view(view_name)
        st.rerun()

    menu_col, brand_col = st.columns([1.05, 18.95], gap="medium")

    with menu_col:
        with st.popover("\u2630"):
            st.caption("Menu")
            for view_name in nav_routes:
                meta = PAGE_META[view_name]
                if view_name == current_page():
                    st.markdown(f"<div class='drawer-current'>{meta['label']}</div>", unsafe_allow_html=True)
                else:
                    if st.button(meta["label"], key=f"nav_{view_name}", width="stretch"):
                        open_page(view_name)

    with brand_col:
        label = PAGE_META.get(current_page(), PAGE_META["home"])["label"]
        st.markdown(shell_brand(label), unsafe_allow_html=True)

    page_renderer = nav_routes.get(current_page(), Home.render)
    st.session_state.pop("page_loading_label", None)
    page_renderer()


if __name__ == "__main__":
    run_app()
