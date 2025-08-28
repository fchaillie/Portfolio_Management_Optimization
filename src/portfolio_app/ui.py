"""
UI helpers: global CSS, banner, disclaimer, background image and sidebar width.
These utilities are intentionally content-agnostic and only deal with presentation.
"""
from __future__ import annotations

import base64
from pathlib import Path
import streamlit as st

# ---- Public constants ----
DEFAULT_BG_PATH = "assets/background.jpg"  # expected location for the background image
DEFAULT_SIDEBAR_WIDTH_PX = 540             # makes the sidebar wide enough for dense controls

@st.cache_resource
def _bg_b64_cached(img_path: str = DEFAULT_BG_PATH) -> str:
    """
    Read and base64-encode the background image ONCE per session.
    Subsequent calls reuse the cached string (faster reruns).
    """
    return base64.b64encode(Path(img_path).read_bytes()).decode()

def inject_global_css_banner_and_disclaimer() -> None:
    """
    Injects a single CSS block (safe to call once at the top of the app) that:
    - maximizes working space and removes default toolbar,
    - styles headings and tables,
    - tightens vertical spacing,
    - adjusts sidebar paddings and hides toggle buttons,
    - renders a centered header banner and a pinned disclaimer (HTML content).
    The banner/disclaimer use CSS classes so their styling stays in this block.
    """
    # --- Inject CSS ---
    st.markdown(
        """
        <style>
        /* ===== Layout: pull main content up, full-width, no header/toolbar ===== */
        header[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
        .block-container{
            transform: translateY(12px);
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
            width: 100% !important;
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        .block-container > div:first-child{ margin-top: 0 !important; padding-top: 0 !important; }

        /* ===== Typography & spacing ===== */
        h1, h2, h3{
            color:#000 !important; background:#fff !important; padding:6px 2px;
            border-radius:0px; display:inline-block; box-shadow:0 2px 6px rgba(0,0,0,0.06);
            margin:2px 0 2px 0;
        }
        h2, h3{ margin-bottom: 4px !important; }
        .element-container:has(div[data-testid="stPlotlyChart"]){ margin-top: 0px !important; }
        [data-testid="element-container"]{ margin-bottom: 8px !important; }

        /* ===== Tables ===== */
        div[data-testid="stTable"] th,
        div[data-testid="stDataFrame"] th{ color: black !important; font-weight: 700 !important; }
        div[data-testid="stTable"]{
            background:#fff !important; border-radius:0px !important;
            box-shadow:0 2px 8px rgba(0,0,0,0.10) !important;
            padding:4px 6px !important; font-size:1.1rem !important; margin-bottom:8px !important;
        }

        /* ===== Sidebar: hide toggles and tighten spacing ===== */
        [data-testid="collapsedControl"],
        [data-testid="baseButton-toggleSidebar"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarToggle"],
        [data-testid="stIconButton"],
        button[title="Collapse sidebar"],
        button[title="Expand sidebar"],
        button[title="Toggle sidebar"]{ display:none !important; }
        section[data-testid="stSidebar"] header{ display:none !important; height:0 !important; }
        [data-testid="stSidebar"]{ padding-top:0 !important; }
        [data-testid="stSidebar"] > div{ padding-top:0 !important; margin-top:0 !important; }
        [data-testid="stSidebarContent"]{ padding-top:0 !important; margin-top:0 !important; transform: translateY(-75px); }
        [data-testid="stSidebarContent"] > div:first-child{ margin-top:0 !important; padding-top:0 !important; }
        [data-testid="stSidebarContent"] [data-testid="element-container"]{ margin-bottom:6px !important; }

        /* Lock the Tickers textarea size in the sidebar */
        section[data-testid="stSidebar"] textarea[aria-label="Tickers"]{
          resize: none !important;
          height: 200px !important;
          overflow: auto !important;
        }

        /* ===== Utility class for banner ===== */
        .banner{
          background-color:#D0F0C0; padding:18px 40px; font-size:2.2rem; font-weight:bold;
          border-radius:0px; box-shadow:0 2px 6px rgba(0,0,0,0.06); display:inline-block; margin-top:0;
          margin-bottom: 35px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Render Banner HTML ---
    st.markdown(
        """
        <div style="text-align: center;">
          <div class="banner">Design your investment strategy !</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def apply_background_image(path: str = DEFAULT_BG_PATH, overlay_rgba: str = "rgba(0,0,0,0.10)") -> None:
    """
    Reads an image from `path` (cached), encodes it as base64, and applies it as a fixed full-screen
    background with a dark overlay. The overlay improves contrast for foreground widgets.
    """

    try:
        b64 = _bg_b64_cached(path)  # cached read (faster on reruns)
    except Exception:
        # Fail silently if the file isn't present; the app still works without a background.
        return

    st.markdown(
        f"""
        <style>
          html, body, .stApp {{ height: 100%; min-height: 100vh; }}
          .stApp {{
            background: url("data:image/jpg;base64,{b64}") center / cover fixed no-repeat;
            position: relative;
          }}
          .stApp::before {{
            content: "";
            position: fixed; inset: 0;
            background: {overlay_rgba};
            pointer-events: none; z-index: 0;
          }}
          .stApp > div {{ position: relative; z-index: 1; }}
        </style>
        """, unsafe_allow_html=True
    )

def set_sidebar_width(width_px: int = DEFAULT_SIDEBAR_WIDTH_PX) -> None:
    """
    Expands the sidebar width via CSS so inputs do not wrap awkwardly.
    """
    st.markdown(
        f"""
        <style>
          section[data-testid="stSidebar"] {{ width: {width_px}px !important; }}
          div[data-testid="stSidebar"] > div {{ padding-right: 8px; }}
        </style>
        """, unsafe_allow_html=True
    )