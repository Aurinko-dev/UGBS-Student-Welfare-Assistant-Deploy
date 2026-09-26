"""
Shared look-and-feel for every page (injected once from app.py).

Design notes
------------
* Navy + gold, from the BHJCR constitution cover used elsewhere in this project.
* Severity badges stay on a green/amber/red scale on purpose: that is a
  functional urgency signal, not branding.
* BUTTONS USE FIXED COLOURS. The previous stylesheet switched button colours
  with `prefers-color-scheme` (the *computer's* dark mode). When the computer
  was in dark mode but Streamlit's own theme was light, buttons became a dark
  background with dark-navy text, i.e. unreadable. Gold background + navy
  text is readable on both light and dark themes, so no switching is needed.
"""
import inspect

import streamlit as st

_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --navy: #0B1F4D;
        --navy-light: #1E3A72;
        --gold: #C9A227;
        --gold-soft: #F3E3A6;
        --gold-light: #EDE3C6;
        --muted: #5B6472;
        --border: #E4E4E0;
        --title: __TITLE__;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* --- Page header ------------------------------------------------- */
    .ugbs-header { display: flex; align-items: center; gap: 16px; margin-bottom: 4px; }
    .ugbs-crest {
        width: 52px; height: 52px; border-radius: 50%; background: var(--navy);
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .ugbs-crest span { font-family: Georgia, serif; font-weight: 700; color: var(--gold);
                       font-size: 0.65rem; letter-spacing: 0.3px; }
    .main-header { font-size: 1.9rem; font-weight: 700; color: var(--title); margin: 0; line-height: 1.2; }
    .sub-header { font-size: 1.0rem; color: var(--muted); margin: 3px 0 0 0; }
    .ugbs-rule { height: 3px; background: linear-gradient(90deg, var(--gold) 0%, var(--gold-light) 100%);
                 border-radius: 2px; margin: 14px 0 20px 0; width: 100%; }
    .assist-prompt { font-size: 1.15rem; font-weight: 600; margin: 0 0 2px 0; }
    .assist-hint { color: var(--muted); margin: 0 0 14px 0; font-size: 0.95rem; }

    /* --- Sidebar brand ------------------------------------------------ */
    .side-brand { display:flex; align-items:center; gap:10px; margin: 2px 0 10px 0; }
    .side-brand .name { font-weight: 700; color: var(--title); font-size: 1.05rem; line-height: 1.2; }
    .side-brand .tag { font-size: 0.78rem; color: var(--muted); }

    /* --- Badges (functional colours) --------------------------------- */
    .badge { display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: 0.72rem;
             font-weight: 600; margin-right: 6px; letter-spacing: 0.2px; }
    .badge-category { background:#EDE3C6; color:#0B1F4D; }
    .badge-low { background:#DCFCE7; color:#166534; }
    .badge-medium { background:#FEF9C3; color:#854D0E; }
    .badge-high { background:#FFEDD5; color:#9A3412; }
    .badge-critical { background:#FEE2E2; color:#991B1B; }
    .badge-escalated { background:#DC2626; color:#FFFFFF; }
    .badge-ack { background:#E5E7EB; color:#374151; }
    .badge-new { background:#C9A227; color:#0B1F4D; }

    /* --- Buttons: fixed colours, readable on any theme ----------------- */
    div.stButton button {
        border-radius: 10px; border: 1.5px solid var(--gold);
        background-color: var(--gold); font-weight: 600;
        transition: all 0.15s ease-in-out;
    }
    div.stButton button, div.stButton button * { color: var(--navy) !important; }
    div.stButton button:hover { background-color: var(--navy); border-color: var(--navy); }
    div.stButton button:hover, div.stButton button:hover * { color: var(--gold-soft) !important; }

    /* selected topic box (primary) = navy with gold text */
    div.stButton button[kind="primary"],
    div.stButton button[data-testid="stBaseButton-primary"] {
        background-color: var(--navy); border-color: var(--navy);
    }
    div.stButton button[kind="primary"], div.stButton button[kind="primary"] *,
    div.stButton button[data-testid="stBaseButton-primary"],
    div.stButton button[data-testid="stBaseButton-primary"] * { color: var(--gold-soft) !important; }

    /* example-question chips shown under a topic box */
    .st-key-examples div.stButton button {
        background-color: var(--gold-light); border: 1.5px solid var(--gold);
        font-weight: 500; text-align: left; justify-content: flex-start;
    }
    .st-key-examples div.stButton button:hover { background-color: var(--gold); border-color: var(--gold); }
    .st-key-examples div.stButton button:hover, .st-key-examples div.stButton button:hover * { color: var(--navy) !important; }

    /* --- Source expander ---------------------------------------------- */
    div[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 8px; }

    /* --- Live admin feed ---------------------------------------------- */
    .live-card { background:#FFFFFF; color:#1A1F2B; border:1px solid var(--border);
                 border-left:5px solid var(--gold); border-radius:10px;
                 padding:10px 14px; margin-bottom:9px; }
    .live-card.escalated { border-left-color:#DC2626; background:#FEF2F2; }
    .live-top { display:flex; flex-wrap:wrap; align-items:center; gap:2px; margin-bottom:5px; }
    .live-time { font-size:0.78rem; color:#5B6472; margin-right:10px; font-variant-numeric: tabular-nums; }
    .live-text { font-size:0.98rem; margin: 2px 0 4px 0; word-break: break-word; }
    .live-meta { font-size:0.78rem; color:#5B6472; }
</style>
"""


def _is_dark() -> bool:
    """Active Streamlit theme (not the OS setting). Falls back to light."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def inject_css() -> None:
    title = "#F3E3A6" if _is_dark() else "#0B1F4D"
    st.markdown(_CSS.replace("__TITLE__", title), unsafe_allow_html=True)


def wide(fn) -> dict:
    """Keyword argument that makes a widget fill its container, for whichever
    Streamlit version is installed. New versions use width="stretch"; the old
    use_container_width flag is deprecated and slated for removal."""
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}