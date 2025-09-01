"""
Streamlit callbacks and small UI helpers that interact with session_state.
Kept separate for clarity and easier testing.
"""

from __future__ import annotations

import streamlit as st
import yfinance as yf 


def current_list() -> list[str]:
    """
    Read the current universe from the textarea (one ticker per line).
    """
    return [
        x.strip().upper()
        for x in st.session_state.get("tickers_text", "").splitlines()
        if x.strip()
    ]


def cb_add() -> None:
    """
    Validate and add a new ticker to the universe list.
    We download 1 month of data to quickly weed out unknown symbols.
    """
    t = st.session_state.get("new_ticker", "").strip().upper()
    if not t:
        return
    try:
        test = yf.download(t, period="1mo", auto_adjust=True, progress=False)
        if test.empty:
            return
        cur = current_list()
        cur.append(t)
        st.session_state["tickers_text"] = "\n".join(sorted(set(cur)))
        st.session_state["new_ticker"] = ""
    except Exception:
        pass


def cb_remove() -> None:
    """
    Remove the ticker selected in the 'to_remove' selectbox, if any.
    """
    tr = st.session_state.get("to_remove", "(none)")
    if tr == "(none)":
        return
    cur = current_list()
    st.session_state["tickers_text"] = "\n".join(
        sorted(set([x for x in cur if x != tr]))
    )
    st.session_state["to_remove"] = "(none)"
