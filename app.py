"""
Entry point:  streamlit run app.py

This file is now only the ROUTER. It sets up the page, the shared styling and
the sidebar, then hands over to whichever page is selected:

    views/user_interface.py        the student chat            (was the old app.py)
    views/admin_interface.py       the analytics dashboard
    views/live_admin_interface.py  the real-time monitor        (new)
    views/settings.py              system status + classifier tester
"""
import streamlit as st

import analytics_db
from logo import logo_html
from ui_theme import inject_css, wide

st.set_page_config(
    page_title="UGBS Student Welfare AI",
    page_icon="ðŸŽ“",
    layout="wide",
    initial_sidebar_state="expanded",
)
analytics_db.init_db()
inject_css()

ui_page = st.Page("views/user_interface.py", title="User Interface", icon="ðŸ’¬", default=True)
admin_page = st.Page("views/admin_interface.py", title="Admin Interface", icon="ðŸ“Š")
live_page = st.Page("views/live_admin_interface.py", title="Live Admin Interface", icon="ðŸ”´")
settings_page = st.Page("views/settings.py", title="Settings", icon="âš™ï¸")

# Pages shown in the main navigation block (Settings sits separately, below a divider).
main_pages = [ui_page, admin_page, live_page]

# Newer Streamlit can hide its automatic menu so we control the exact order
# (logo first, Settings last). Older versions fall back to the built-in menu
# with named sections, which still keeps the same order.
try:
    nav = st.navigation([*main_pages, settings_page], position="hidden")
    custom_nav = True
except TypeError:
    nav = st.navigation({
        "Student": [ui_page],
        "Administration": [admin_page, live_page],
        "System": [settings_page],
    })
    custom_nav = False


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.nationality = None
    st.session_state.awaiting_nationality = False
    st.session_state.pending_query = None
    st.session_state.awaiting_department = False
    st.session_state.active_topic = None
    st.session_state.show_guidance = False


with st.sidebar:
    # 1. Logo, always first.
    st.markdown(
        '<div class="side-brand">'
        + logo_html(52) +
        '<div><div class="name">UGBS Welfare Hub</div>'
        '<div class="tag">AI Intake &amp; Triage System</div></div></div>',
        unsafe_allow_html=True)

    if custom_nav:
        # 2. Navigation, in the requested order.
        for page in main_pages:
            st.page_link(page)

        # 3. Push everything below towards the bottom of the sidebar.
        st.markdown('<div style="height:max(24px, calc(100vh - 480px));"></div>',
                    unsafe_allow_html=True)

    # 4. Chat-only action (visible, named button).
    if nav.title == ui_page.title:
        st.button("ðŸ—‘ï¸  Clear chat history", key="clear_chat", on_click=_clear_chat,
                  **wide(st.button))

    # 5. Settings, last.
    if custom_nav:
        st.divider()
        st.page_link(settings_page)

nav.run()
# python -m streamlit run app.py