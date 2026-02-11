import streamlit as st

pg = st.navigation(
    [
        st.Page("Convert.py"),
        st.Page("Index.py"),
        st.Page("Search.py")
    ]
)

pg.run()