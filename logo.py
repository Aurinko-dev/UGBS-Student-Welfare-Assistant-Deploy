"""
Shared UGBS logo, so every page uses the same image.

Put the file at  assets/ugbs_logo.png  (next to app.py). If it is missing,
this falls back to the old "UGBS" text crest instead of breaking the page.

    from logo import logo_html
    st.markdown(f'<div class="ugbs-header">{logo_html(64)}...', unsafe_allow_html=True)
"""
import base64
from functools import lru_cache
from pathlib import Path

_LOGO_PATH = Path(__file__).parent / "assets" / "ugbs_logo.png"


@lru_cache(maxsize=1)
def _logo_b64() -> str:
    try:
        return base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    except OSError:
        return ""


def logo_html(height: int = 52) -> str:
    """An <img> tag for the crest, `height` pixels tall (width follows)."""
    b64 = _logo_b64()
    if b64:
        return (f'<img src="data:image/png;base64,{b64}" alt="UGBS crest" '
                f'style="height:{height}px;width:auto;flex-shrink:0;">')
    return (f'<div class="ugbs-crest" style="width:{height}px;height:{height}px;">'
            '<span style="font-size:0.6rem;">UGBS</span></div>')