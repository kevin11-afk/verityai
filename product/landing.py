"""Streamlit's own page for the Cloud deployment: a hand-off to the React app at ./product/."""
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="VerityAI", page_icon="🛡️", initial_sidebar_state="collapsed")
st.markdown("""
<style>
  [data-testid="stSidebar"], [data-testid="stHeader"] { display: none; }
  .block-container { max-width: 520px; padding-top: 18vh; }
</style>
""", unsafe_allow_html=True)
st.title("VerityAI")
st.caption("Intelligence for media authenticity")
st.link_button("Open the VerityAI app", "product/", type="primary", use_container_width=True)
# Best-effort automatic hand-off; the button above always works.
components.html(
    "<script>try{var p=window.parent;p.location.replace(new URL('product/',p.location.href).href);}catch(e){}</script>",
    height=0,
)
