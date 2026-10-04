"""
Welmia Face — Streamlit UI.

Visual system aligned with welmia.kesug.com (projects.php / about.php):
white canvas, indigo accent, Inter + JetBrains Mono, soft 1px borders,
16px radii, quiet shadows.

Model logic, widgets, session state and download behaviour are unchanged —
this file is a presentation layer only.
"""

import io
import time
import zipfile

import streamlit as st
import torch
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Welmia Face",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOGO
# ============================================================

LOGO_PATH = "static/welmia_logo.png"

try:
    st.logo(
        LOGO_PATH,
        size="medium",
        link="https://welmia.kesug.com",
    )
except Exception:
    pass


# ============================================================
# DESIGN TOKENS
# Mirrors the :root block in projects.php
# ============================================================

TOKENS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink:        #0b0d12;
    --ink-2:      #4a5160;
    --ink-3:      #8b93a3;
    --line:       #e8eaef;
    --line-2:     #f2f3f6;
    --card:       #ffffff;
    --soft:       #fafbfc;
    --accent:     #4338ca;
    --accent-2:   #6d67f5;
    --accent-soft:#eef0ff;
    --accent-line:#e0e3ff;
    --good:       #0f766e;
    --good-soft:  #ecfdf5;
    --good-line:  #c9f2e4;
    --warn:       #b45309;
    --warn-soft:  #fffbeb;
    --warn-line:  #fde68a;
    --radius:     16px;
    --radius-sm:  10px;
    --shadow:     0 1px 2px rgba(11,13,18,.04), 0 8px 24px -12px rgba(11,13,18,.12);
    --shadow-lg:  0 1px 2px rgba(11,13,18,.04), 0 24px 48px -24px rgba(11,13,18,.20);
    --mono:       'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}
</style>
"""


# ============================================================
# UI STYLESHEET
# ============================================================

st.html(
    """
    <style>

    /* ========================================================
       BASE
    ======================================================== */

    html, body, [class*="css"], .stApp {
        font-family: Inter, -apple-system, BlinkMacSystemFont,
                     'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background: #ffffff;
        color: var(--ink);
        -webkit-font-smoothing: antialiased;
    }

    /* tight top padding: the generate controls must sit above the fold */
    .main .block-container {
        max-width: 1120px;
        padding-top: 1.1rem;
        padding-bottom: 4rem;
    }

    ::selection {
        background: var(--accent-soft);
        color: var(--accent);
    }


    /* ========================================================
       STREAMLIT CHROME
    ======================================================== */

    #MainMenu, footer { visibility: hidden; }

    header[data-testid="stHeader"] { background: transparent; }

    header[data-testid="stHeader"] button[kind="header"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: var(--radius-sm);
        box-shadow: var(--shadow);
        transition: transform .15s ease, box-shadow .15s ease;
    }

    header[data-testid="stHeader"] button[kind="header"]:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
    }

    /* Deploy / hamburger chrome colours */
    .stApp [data-testid="stToolbar"] { right: 1.1rem; }


    /* ========================================================
       HERO  (compact — stays above the fold on a laptop)
    ======================================================== */

    .hero {
        text-align: center;
        padding: 6px 20px 20px;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 7px;

        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: .05em;
        text-transform: uppercase;

        color: var(--accent);
        background: var(--accent-soft);
        border: 1px solid var(--accent-line);
        border-radius: 999px;

        padding: 5px 12px;
        margin-bottom: 14px;
    }

    .hero-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 0 3px rgba(34, 197, 94, .18);
    }

    .hero-title {
        margin: 0 0 8px;

        font-size: clamp(25px, 3.2vw, 37px);
        line-height: 1.12;
        letter-spacing: -.035em;
        font-weight: 700;

        color: var(--ink);
    }

    .hero-sub {
        margin: 0 auto;
        max-width: 660px;

        font-size: 14.5px;
        line-height: 1.55;
        color: var(--ink-2);
    }


    /* ========================================================
       SPEC CHIPS
       One line instead of a tall stat strip.
    ======================================================== */

    .spec-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;

        margin: 14px 0 18px;
    }

    .spec-chip {
        display: inline-flex;
        align-items: baseline;
        gap: 5px;

        padding: 5px 12px;
        border-radius: 999px;

        border: 1px solid var(--line);
        background: var(--soft);

        font-size: 12.5px;
        color: var(--ink-3);
        white-space: nowrap;
    }

    .spec-chip b {
        font-weight: 600;
        font-size: 13px;
        color: var(--ink);
        letter-spacing: -.01em;
    }


    /* ========================================================
       CARD CONTAINER
       (matches the .card treatment on projects.php)
    ======================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 6px 26px 22px;
    }

    .section-label {
        display: flex;
        align-items: center;
        gap: 11px;

        margin: 12px 0 6px;

        font-size: 19px;
        letter-spacing: -.03em;
        font-weight: 700;
        color: var(--ink);
    }

    .section-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;

        width: 32px;
        height: 32px;
        border-radius: var(--radius-sm);

        background: var(--accent-soft);
        border: 1px solid var(--accent-line);
        color: var(--accent);

        font-size: 15px;
    }

    .section-sub {
        margin: 0 0 18px;
        font-size: 13.5px;
        color: var(--ink-2);
    }


    /* ========================================================
       THEME VARIABLES
       Streamlit drives most widget colours from CSS custom
       properties. Overriding them here fixes widgets that injected
       stylesheet rules were losing to — and works whether or not
       a custom theme is set in .streamlit/config.toml.
    ======================================================== */

    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    .stApp {
        --primary-color: var(--accent);
        --primary-color-fade: var(--accent-soft);
        --background-color: #ffffff;
        --secondary-background-color: var(--soft);
        --text-color: var(--ink);
        --border-color: var(--line);
        --link-color: var(--accent);
    }


    /* ========================================================
       WIDGET LABELS
    ======================================================== */

    [data-testid="stApp"] [data-testid="stWidgetLabel"] p,
    [data-testid="stApp"] [data-testid="stWidgetLabel"] label,
    [data-testid="stApp"] .stSelectbox label p,
    [data-testid="stApp"] .stSlider label p,
    [data-testid="stApp"] .stNumberInput label p {
        color: var(--ink-2) !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        letter-spacing: -.005em;
    }

    [data-testid="stApp"] .stCaption,
    [data-testid="stApp"] [data-testid="stCaptionContainer"],
    [data-testid="stApp"] [data-testid="stCaptionContainer"] p {
        color: var(--ink-3) !important;
        font-size: 12.5px !important;
        line-height: 1.6;
    }


    /* ========================================================
       BUTTONS
    ======================================================== */

    [data-testid="stApp"] .stButton > button,
    [data-testid="stApp"] .stButton > button:focus,
    [data-testid="stApp"] .stButton > button:hover {
        min-height: 50px !important;
        border-radius: 11px !important;

        background: var(--ink) !important;
        background-color: var(--ink) !important;
        border: 1px solid var(--ink) !important;
        color: #fff !important;

        font-size: 15px !important;
        font-weight: 600 !important;
        letter-spacing: -.01em;

        box-shadow: var(--shadow);

        transition:
            transform .12s ease,
            box-shadow .15s ease,
            background .15s ease !important;
    }

    [data-testid="stApp"] .stButton > button:hover {
        transform: translateY(-1px);
        background: #1a1e27 !important;
        background-color: #1a1e27 !important;
        border-color: #1a1e27 !important;
        box-shadow: var(--shadow-lg) !important;
    }

    [data-testid="stApp"] .stButton > button:active { transform: translateY(0); }

    [data-testid="stApp"] .stButton > button:focus:not(:focus-visible) {
        box-shadow: var(--shadow) !important;
    }

    /* download buttons — the quieter secondary treatment */
    [data-testid="stApp"] .stDownloadButton > button,
    [data-testid="stApp"] .stDownloadButton > button:focus,
    [data-testid="stApp"] .stDownloadButton > button:hover {
        min-height: 40px !important;
        border-radius: var(--radius-sm) !important;

        background: #fff !important;
        background-color: #fff !important;
        border: 1px solid var(--line) !important;
        color: var(--ink) !important;

        font-size: 13.5px !important;
        font-weight: 600 !important;

        box-shadow: none;

        transition:
            transform .12s ease,
            box-shadow .15s ease,
            border-color .15s ease !important;
    }

    [data-testid="stApp"] .stDownloadButton > button:hover {
        transform: translateY(-1px);
        border-color: #d9dce4 !important;
        box-shadow: var(--shadow) !important;
    }

    [data-testid="stApp"] .stDownloadButton > button:active { transform: translateY(0); }

    /* "Download All" gets the primary weight */
    [data-testid="stApp"] .stDownloadButton > button[kind="primary"],
    [data-testid="stApp"] [data-testid="stDownloadButton"] > button[kind="primary"],
    [data-testid="stApp"] [data-testid="stDownloadButton"] button[kind="primary"] {
        background: var(--ink) !important;
        background-color: var(--ink) !important;
        border-color: var(--ink) !important;
        color: #fff !important;
        font-weight: 600 !important;
        min-height: 50px !important;
        border-radius: 11px !important;
    }

    [data-testid="stApp"] .stDownloadButton > button[kind="primary"]:hover {
        background: #1a1e27 !important;
        background-color: #1a1e27 !important;
        border-color: #1a1e27 !important;
        box-shadow: var(--shadow-lg) !important;
    }


    /* ========================================================
       SLIDER
    ======================================================== */

    [data-testid="stApp"] div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background: var(--accent) !important;
        background-color: var(--accent) !important;
        border: 2px solid #fff !important;
        box-shadow: 0 0 0 1px var(--accent-line),
                    0 4px 12px rgba(67, 56, 202, .28) !important;
    }

    [data-testid="stApp"] div[data-testid="stSlider"] div[role="slider"] {
        background: var(--accent) !important;
        background-color: var(--accent) !important;
        border: 2px solid #fff !important;
    }

    [data-testid="stApp"] div[data-testid="stSlider"] div[role="slider"]:focus {
        box-shadow: 0 0 0 4px var(--accent-soft) !important;
    }

    [data-testid="stApp"] div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
        background: var(--accent-line) !important;
        background-color: var(--accent-line) !important;
    }

    [data-testid="stApp"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stApp"] div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: var(--ink-3) !important;
        font-size: 11.5px !important;
    }

    /* the value bubble above the thumb */
    [data-testid="stApp"] div[data-testid="stSlider"] [data-testid="stTooltipContent"],
    [data-testid="stApp"] div[data-testid="stSlider"] div[data-baseweb="tooltip"] {
        background: var(--ink) !important;
        color: #fff !important;
    }


    /* ========================================================
       SELECT + NUMBER INPUT
    ======================================================== */

    [data-testid="stApp"] div[data-baseweb="select"] > div,
    [data-testid="stApp"] div[data-baseweb="select"] > div:hover,
    [data-testid="stApp"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #fff !important;
        background-color: #fff !important;
        border: 1px solid var(--line) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--ink) !important;
        font-size: 14px;

        transition:
            border-color .15s ease,
            box-shadow .15s ease !important;
    }

    [data-testid="stApp"] div[data-baseweb="select"] > div > div {
        background: transparent !important;
        background-color: transparent !important;
        color: var(--ink) !important;
    }

    [data-testid="stApp"] div[data-baseweb="select"] svg {
        color: var(--ink-3) !important;
        fill: currentColor !important;
    }

    [data-testid="stApp"] div[data-baseweb="select"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    /* the dropdown list itself */
    [data-testid="stApp"] ul[role="listbox"],
    [data-testid="stApp"] div[data-baseweb="menu"] {
        background: #fff !important;
        background-color: #fff !important;
        border: 1px solid var(--line) !important;
        border-radius: var(--radius-sm) !important;
    }

    [data-testid="stApp"] ul[role="listbox"] li:hover {
        background: var(--soft) !important;
    }

    [data-testid="stApp"] ul[role="listbox"] li[aria-selected="true"] {
        background: var(--accent-soft) !important;
        color: var(--accent) !important;
    }

    [data-testid="stApp"] div[data-testid="stNumberInput"] input {
        background: #fff !important;
        background-color: #fff !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--line) !important;
        color: var(--ink) !important;
        font-size: 14px;

        transition:
            border-color .15s ease,
            box-shadow .15s ease !important;
    }

    [data-testid="stApp"] div[data-testid="stNumberInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    /* the +/- steppers */
    [data-testid="stApp"] div[data-testid="stNumberInput"] button {
        background: var(--soft) !important;
        background-color: var(--soft) !important;
        border-color: var(--line) !important;
        color: var(--ink-2) !important;
    }

    [data-testid="stApp"] div[data-testid="stNumberInput"] button:hover {
        background: var(--line-2) !important;
        background-color: var(--line-2) !important;
        color: var(--ink) !important;
    }


    /* ========================================================
       MODE NOTE (random-mode placeholder column)
    ======================================================== */

    .mode-note {
        display: flex;
        align-items: center;
        gap: 10px;

        height: 42px;
        padding: 0 14px;

        border: 1px solid var(--line);
        border-radius: var(--radius-sm);
        background: var(--soft);

        color: var(--ink-2);
        font-size: 13.5px;
        font-weight: 500;
    }

    .mode-note .mode-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent);
        flex-shrink: 0;
    }

    .mode-note b {
        color: var(--ink);
        font-weight: 600;
    }


    /* ========================================================
       RESULT BAR
    ======================================================== */

    /* ========================================================
       GENERATION CONFIRMATION
       Sits directly under the Generate button, so the user sees
       the outcome without scrolling. Sticky so it stays visible
       while they browse the gallery.
    ======================================================== */

    .gen-done {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;

        margin-top: 14px;
        padding: 12px 15px;

        border: 1px solid var(--good-line);
        border-radius: var(--radius-sm);
        background: var(--good-soft);

        font-size: 13px;
        color: var(--ink-2);
    }

    .gen-done b {
        color: var(--ink);
        font-weight: 600;
    }

    .gen-done-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--good);
        box-shadow: 0 0 0 3px rgba(15, 118, 110, .14);
        flex-shrink: 0;
    }

    .gen-done-link {
        margin-left: auto;
        white-space: nowrap;

        display: inline-flex;
        align-items: center;
        gap: 5px;

        padding: 5px 11px;
        border-radius: 8px;

        background: #fff;
        border: 1px solid var(--good-line);
        color: var(--good);

        font-size: 12.5px;
        font-weight: 600;
        text-decoration: none;

        transition:
            transform .12s ease,
            box-shadow .15s ease;
    }

    .gen-done-link:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow);
        color: var(--good);
    }

    .gen-done-link svg { width: 13px; height: 13px; }


    /* ========================================================
       RESULT BAR  (sticky — follows the user into the gallery)
    ======================================================== */

    .result-bar {
        position: sticky;
        top: 3.6rem;
        z-index: 30;

        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 15px;
        flex-wrap: wrap;

        margin: 8px 0 20px;
        padding: 13px 18px;

        border: 1px solid var(--line);
        border-left: 3px solid var(--accent);
        border-radius: var(--radius-sm);

        background: rgba(255, 255, 255, .93);
        backdrop-filter: saturate(180%) blur(12px);
        -webkit-backdrop-filter: saturate(180%) blur(12px);

        box-shadow: var(--shadow);

        font-size: 13px;
        color: var(--ink-2);
    }

    .result-status {
        display: flex;
        align-items: center;
        gap: 9px;

        font-weight: 600;
        color: var(--ink);
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--good);
        box-shadow: 0 0 0 3px var(--good-soft);
    }

    .result-meta {
        font-family: var(--mono);
        font-size: 12.5px;
        color: var(--ink-3);
    }

    /* keep anchored targets clear of the pinned bar */
    .anchor { scroll-margin-top: 7rem; height: 0; }

    [data-testid="stAppViewContainer"] { scroll-behavior: smooth; }


    /* ========================================================
       IMAGE CARDS
    ======================================================== */

    div[data-testid="stImage"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--soft);
        padding: 6px;
        overflow: hidden;

        transition:
            transform .18s ease,
            box-shadow .18s ease,
            border-color .18s ease;
    }

    div[data-testid="stImage"]:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
        border-color: #dfe2ea;
    }

    div[data-testid="stImage"] img {
        border-radius: 9px;
        display: block;
        width: 100%;
        image-rendering: auto;
    }

    /* tighten the gap between a face and its download button */
    [data-testid="stVerticalBlock"]:has(> [data-testid="stImage"]) {
        gap: 0.5rem;
    }

    /* Never upscale the 128x128 output beyond a sensible size. */
    [data-testid="stApp"] div[data-testid="stImage"] img {
        max-width: 176px;
        width: 100% !important;
        margin: 0 auto;
        image-rendering: auto;
    }

    /* Gallery rows only — the settings row has no stImage, so it keeps
       its own 3-up layout. Wrapping + an even column ladder stops the
       grid from ever leaving a lone stretched tile on its own row. */
    [data-testid="stApp"] div[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) {
        flex-wrap: wrap;
        gap: 0.85rem;
    }

    [data-testid="stApp"] div[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) > [data-testid="stColumn"] {
        min-width: 148px;
    }

    @media (max-width: 1000px) {
        [data-testid="stApp"] div[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) > [data-testid="stColumn"] {
            min-width: 30%;
        }
    }

    @media (max-width: 620px) {
        [data-testid="stApp"] div[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) > [data-testid="stColumn"] {
            min-width: 46%;
        }
    }


    /* ========================================================
       EMPTY STATE
    ======================================================== */

    .empty-card {
        margin-top: 26px;
        padding: 62px 26px;

        text-align: center;
        border-radius: var(--radius);
        border: 1px dashed var(--line);
        background: var(--soft);
    }

    .empty-icon {
        width: 56px;
        height: 56px;
        margin: 0 auto 16px;

        display: grid;
        place-items: center;
        border-radius: 16px;

        background: var(--accent-soft);
        border: 1px solid var(--accent-line);

        font-size: 24px;
    }

    .empty-title {
        color: var(--ink);
        font-size: 19px;
        font-weight: 700;
        letter-spacing: -.02em;
    }

    .empty-text {
        margin: 7px auto 0;
        max-width: 360px;

        color: var(--ink-2);
        font-size: 14.5px;
        line-height: 1.6;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .footer {
        text-align: center;
        margin-top: 56px;
        padding-top: 26px;

        border-top: 1px solid var(--line);

        color: var(--ink-3);
        font-size: 12.5px;
        line-height: 1.9;
    }

    .footer-brand {
        color: var(--ink-2);
        font-weight: 600;
    }

    .footer-link {
        color: var(--accent);
        text-decoration: none;
        font-weight: 500;
        transition: color .15s ease;
    }

    .footer-link:hover { text-decoration: underline; }

    .footer .lic {
        display: inline-block;
        margin-top: 2px;
        padding: 3px 9px;
        border-radius: 999px;

        background: var(--warn-soft);
        border: 1px solid var(--warn-line);
        color: var(--warn);

        font-size: 11px;
        font-weight: 600;
    }


    /* ========================================================
       RESPONSIVE
    ======================================================== */

    @media (max-width: 700px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .hero { padding: 4px 4px 16px; }
        .hero-title { font-size: 25px; }
        .hero-sub { font-size: 14px; }

        .spec-row { margin: 12px 0 14px; gap: 6px; }
        .spec-chip { font-size: 11.5px; padding: 4px 10px; }
        .spec-chip b { font-size: 12px; }

        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 4px 16px 18px;
        }

        .section-label { font-size: 17px; margin: 10px 0 6px; }
        .section-icon { width: 29px; height: 29px; font-size: 14px; }
        .section-sub { font-size: 13px; margin-bottom: 14px; }

        .result-bar {
            align-items: flex-start;
            flex-direction: column;
            gap: 9px;
            top: 3.4rem;
        }

        .gen-done { align-items: flex-start; }
        .gen-done-link { margin-left: 0; }

        .empty-card { padding: 46px 20px; }
    }

    /* genuinely short windows: shed the marketing copy so the
       Generate button is guaranteed visible without scrolling */
    @media (min-width: 701px) and (max-height: 680px) {
        .hero { padding: 2px 20px 14px; }
        .hero-eyebrow { margin-bottom: 10px; }
        .hero-sub { display: none; }
        .spec-row { margin: 12px 0 14px; }
        [data-testid="stVerticalBlockBorderWrapper"] { padding: 4px 26px 18px; }
    }

    @media (prefers-reduced-motion: reduce) {
        * {
            animation: none !important;
            transition: none !important;
        }
        div[data-testid="stImage"]:hover,
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: none;
        }
    }

    </style>
    """
)


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_REPO = "Welmia/welmia-face-1.0-3.6m-base"

MODEL_FILENAME = "model.safetensors"

LATENT_DIM = 100

DEVICE = torch.device("cpu")


# ============================================================
# GENERATOR
# ============================================================

class Generator(torch.nn.Module):

    def __init__(self, nz=100, ngf=64):

        super().__init__()

        self.net = torch.nn.Sequential(

            torch.nn.ConvTranspose2d(
                nz,
                ngf * 8,
                4,
                1,
                0,
                bias=False,
            ),

            torch.nn.BatchNorm2d(
                ngf * 8
            ),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf * 8,
                ngf * 4,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(
                ngf * 4
            ),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(
                ngf * 2
            ),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(
                ngf
            ),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf,
                ngf // 2,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(
                ngf // 2
            ),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf // 2,
                3,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.Tanh(),
        )


    def forward(self, z):

        return self.net(z)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
    )

    model = Generator()

    state_dict = load_file(
        model_path,
        device="cpu",
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# GENERATE FACES
# ============================================================

def generate_faces(
    model,
    num_images,
    seed=None,
):

    if seed is not None:

        generator = torch.Generator(
            device=DEVICE
        ).manual_seed(seed)

        z = torch.randn(
            num_images,
            LATENT_DIM,
            1,
            1,
            generator=generator,
            device=DEVICE,
        )

    else:

        z = torch.randn(
            num_images,
            LATENT_DIM,
            1,
            1,
            device=DEVICE,
        )


    with torch.inference_mode():

        output = model(z)

        output = (
            output.clamp(-1, 1) + 1
        ) / 2


    images = []


    for image_tensor in output:

        image_tensor = (
            image_tensor
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        image_array = (
            image_tensor * 255
        ).astype("uint8")

        image = Image.fromarray(
            image_array,
            mode="RGB",
        )

        images.append(image)


    return images


# ============================================================
# IMAGE TO PNG
# ============================================================

def image_to_bytes(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


# ============================================================
# CREATE ZIP
# ============================================================

def create_zip(images):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for index, image in enumerate(
            images,
            start=1,
        ):

            image_bytes = image_to_bytes(
                image
            )

            zip_file.writestr(
                f"welmia_face_{index}.png",
                image_bytes,
            )


    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# ============================================================
# HERO
# ============================================================

st.html(TOKENS)


st.html(
    """
    <div class="hero">

        <span class="hero-eyebrow">
            <span class="hero-dot"></span>
            Welmia Face 1.0
        </span>

        <h1 class="hero-title">
            Synthetic faces from noise, in real time.
        </h1>

        <p class="hero-sub">
            A 3.6M parameter DCGAN, trained from scratch, running on your CPU.
            Every face it invents is synthetic — nobody real is behind any of them.
        </p>

    </div>
    """
)


st.html(
    """
    <div class="spec-row">

        <span class="spec-chip"><b>3.6M</b> parameters</span>

        <span class="spec-chip"><b>128 × 128</b> output</span>

        <span class="spec-chip"><b>CPU</b> real time</span>

        <span class="spec-chip"><b>100%</b> synthetic</span>

    </div>
    """
)


# ============================================================
# GENERATION SETTINGS
# ============================================================

with st.container(border=True):

    st.html(
        """
        <div class="section-label">
            <span class="section-icon">✨</span>
            Generation
        </div>

        <div class="section-sub">
            Pick how many faces to make, and whether the result
            should be random or reproducible.
        </div>
        """
    )


    col1, col2, col3 = st.columns(
        [1, 1, 1],
        gap="large",
    )


    with col1:

        num_images = st.slider(
            "Number of faces",
            min_value=1,
            max_value=50,
            value=8,
            step=1,
        )


    with col2:

        generation_mode = st.selectbox(
            "Generation mode",
            [
                "Random",
                "Fixed seed",
            ],
            index=0,
        )


    with col3:

        if generation_mode == "Fixed seed":

            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=999999999,
                value=42,
                step=1,
            )

        else:

            seed = None

            st.html(
                """
                <div class="mode-note">
                    <span class="mode-dot"></span>
                    <span><b>Random mode</b> · fresh noise every run</span>
                </div>
                """
            )


    st.write("")


    generate_button = st.button(
        "✨  Generate Faces",
        use_container_width=True,
    )


    st.caption(
        "Random mode creates new faces every time. "
        "Fixed seed mode makes results reproducible."
    )


    # ---- in-place confirmation ----------------------------------------
    # Mirrors the result state right where the user is already looking,
    # so nobody has to scroll down to find out whether it worked.
    done_images = st.session_state.get("images")

    if done_images:

        done_time = st.session_state.get(
            "generation_time",
            0,
        )

        done_seed = st.session_state.get(
            "generation_seed",
            None,
        )

        done_seed_text = (
            "Random seed"
            if done_seed is None
            else f"Fixed seed: {done_seed}"
        )

        st.html(
            f"""
            <div class="gen-done">

                <span class="gen-done-dot"></span>

                <span>
                    <b>{len(done_images)} faces ready</b>
                    &nbsp;·&nbsp; {done_time:.2f}s
                    &nbsp;·&nbsp; {done_seed_text}
                </span>

                <a class="gen-done-link" href="#gallery">
                    View faces
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2.2" stroke-linecap="round"
                         stroke-linejoin="round">
                        <path d="M12 5v14M19 12l-7 7-7-7"/>
                    </svg>
                </a>

            </div>
            """
        )


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model()


# ============================================================
# GENERATION
# ============================================================

if generate_button:

    start_time = time.perf_counter()


    with st.spinner(
        "Generating synthetic faces..."
    ):

        images = generate_faces(
            model=model,
            num_images=num_images,
            seed=seed,
        )


    elapsed = (
        time.perf_counter()
        - start_time
    )


    st.session_state["images"] = images

    st.session_state[
        "generation_time"
    ] = elapsed

    st.session_state[
        "generation_seed"
    ] = seed


# ============================================================
# RESULTS
# ============================================================

if "images" in st.session_state:

    images = st.session_state["images"]

    generation_time = (
        st.session_state.get(
            "generation_time",
            0,
        )
    )

    generation_seed = (
        st.session_state.get(
            "generation_seed",
            None,
        )
    )


    if generation_seed is None:

        seed_text = "Random seed"

    else:

        seed_text = (
            f"Fixed seed: "
            f"{generation_seed}"
        )


    st.html(
        f"""
        <div class="result-bar">

            <div class="result-status">

                <span class="status-dot"></span>

                Generation complete

            </div>

            <div class="result-meta">
                {len(images)} faces
                &nbsp;·&nbsp;
                {generation_time:.2f}s
                &nbsp;·&nbsp;
                {seed_text}
            </div>

        </div>
        """
    )


    st.html('<div id="gallery" class="anchor"></div>')


    # ========================================================
    # GALLERY
    # 6 per row keeps each 128x128 output close to its native size,
    # so faces stay crisp instead of being upscaled and smeared.
    # The flex-wrap rule below reflows these to 2-up on phones.
    # ========================================================

    columns_per_row = 6


    for row_start in range(
        0,
        len(images),
        columns_per_row,
    ):

        row_images = images[
            row_start:
            row_start + columns_per_row
        ]


        columns = st.columns(
            columns_per_row,
            gap="medium",
        )


        for local_index, image in enumerate(
            row_images
        ):

            absolute_index = (
                row_start
                + local_index
                + 1
            )


            with columns[local_index]:

                st.image(
                    image,
                    use_container_width=True,
                )


                image_bytes = (
                    image_to_bytes(
                        image
                    )
                )


                st.download_button(
                    label="↓  PNG",
                    data=image_bytes,
                    file_name=(
                        f"welmia_face_"
                        f"{absolute_index}.png"
                    ),
                    mime="image/png",
                    use_container_width=True,
                    key=(
                        f"download_"
                        f"{absolute_index}"
                    ),
                )


    # ========================================================
    # DOWNLOAD ALL
    # ========================================================

    st.write("")


    zip_bytes = create_zip(
        images
    )


    st.download_button(
        label="📦  Download All Faces",
        data=zip_bytes,
        file_name="welmia_faces.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_all_faces",
        type="primary",
    )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.html(
        """
        <div class="empty-card">

            <div class="empty-icon">
                🧬
            </div>

            <div class="empty-title">
                Ready to generate
            </div>

            <div class="empty-text">
                Choose your settings and create
                a new set of synthetic faces.
            </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        <span class="footer-brand">
            Welmia Face 1.0
        </span>

        · 3.6M parameter synthetic face generator

        <br>

        Synthetic faces only · Not real individuals

        <br>

        <span class="lic">CC-BY-NC-4.0 · non-commercial</span>

        <br>

        <a
            class="footer-link"
            href="https://welmia.kesug.com"
            target="_blank"
        >
            welmia.kesug.com
        </a>

    </div>
    """
)
