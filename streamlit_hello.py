import streamlit as st

st.set_page_config(page_title="Halo Streamlit", layout="centered")
st.title("🎉 Streamlit berhasil!")
nama = st.text_input("Namamu siapa?")
if st.button("Sapa"):
    st.success(f"Halo, {nama or 'kamu'}! 🚀")
