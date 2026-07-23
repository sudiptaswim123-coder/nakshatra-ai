import streamlit as st



def metric_card(title,value):


    st.markdown(
    f"""

    <div class="card">

    <h4>{title}</h4>

    <h2>{value}</h2>


    </div>

    """,
    unsafe_allow_html=True
    )
