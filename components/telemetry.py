import streamlit as st



def telemetry():


    st.subheader(
    "SYSTEM TELEMETRY"
    )


    col1,col2,col3=st.columns(3)


    with col1:
        st.metric(
        "CPU",
        "42%"
        )


    with col2:
        st.metric(
        "GPU",
        "56%"
        )


    with col3:
        st.metric(
        "MODEL",
        "READY"
        )
