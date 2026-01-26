from __future__ import annotations
import streamlit as st
from typing import Any

def cache_get(key: str)-> Any:
    return st.session_state.get("_cache", {}).get(key)

def cache_set(key: str, value: Any) -> None:
    if "_cache" not in st.session_state:
        st.session_state["_cache"] = {}
    st.session_state["_cache"][key] = value

def cache_clear() -> None:
    st.session_state["_cache"]={}