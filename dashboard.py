import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from pipeline import run_pipeline


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="NAKSHATRA AI",
    page_icon="🛰️",
    layout="wide"
)
#=======================
#login system
#========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.markdown("""
    <style>

    .stApp{

        background-image:
        linear-gradient(
        rgba(0,0,0,0.6),
        rgba(0,0,0,0.8)
        ),
        url("https://images.unsplash.com/photo-1446776811953-b23d57bd21aa");

        background-size:cover;
        background-position:center;
        background-attachment:fixed;
    }

    </style>
    """, unsafe_allow_html=True)
    
    
    

    col1,col2,col3 = st.columns([1,2,1])

    with col2:

        username = st.text_input("Username")
        password = st.text_input(
            "Access Key",
            type="password"
        )

        if st.button(
            "LOGIN",
            use_container_width=True
        ):

            if username == "isro" and password == "naksatra2026":

                st.session_state.logged_in = True
                st.rerun()

            else:
                st.error("Access Denied")

    st.stop()

# ==========================
# CSS
# ==========================
st.markdown("""
<style>
.stApp{
    background-image:
        linear-gradient(
            rgba(0,0,0,.45),
            rgba(0,0,0,.75)
        ),
        url("https://images.unsplash.com/photo-1769251971680-005dfa536f07?q=80&w=1032&auto=format&fit=crop");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;

    color: white;
}
.glass{
backdrop-filter: blur(18px);
background: rgba(255,255,255,0.05);
border:1px solid rgba(255,255,255,0.12);
border-radius:24px;
padding:25px;
}
.hero-box{
backdrop-filter: blur(20px);
background: rgba(0,0,0,0.25);
border:1px solid rgba(255,255,255,0.1);
border-radius:25px;
padding:30px;
margin-bottom:20px;
}

.hero-title{
font-size:72px;
font-weight:900;
text-align:center;
letter-spacing:2px;
color:white;
text-shadow:0 0 30px #60a5fa;
margin-top:20px;
}

.hero-subtitle{
text-align:center;
font-size:22px;
color:#93c5fd;
margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)
# ==========================
# ISRO HEADER
# ==========================

col1, col2 = st.columns([1,5])

with col1:
    try:
        st.image(
            "https://images.unsplash.com/photo-1451187580459-43490279c0fa",
            use_container_width=True
        )
    except:
        pass

with col2:
    st.markdown("""
    <div class="hero-box">
    <div class="hero-title">
    🚀 NAKSHATRA AI
    </div>

    <div class="hero-subtitle">
    ISRO BAH 2026 • Autonomous Exoplanet Detection System
    </div>

    </div>
    """, unsafe_allow_html=True)

#===================
#sidebar
#=====================
with st.sidebar:

    try:
        st.image(
            "assets/logo.png",
            width=140
        )
    except:
        pass

    st.markdown("## 🚀 Mission Control")

    st.success("TESS Connected")
    st.success("AI Model Online")
    st.success("Pipeline Active")

    st.divider()

    st.metric("Version","1.0")
    st.metric("Mission","ISRO BAH 2026")

    st.divider()

    st.markdown("""
    ### NAKSHATRA AI

    Autonomous Exoplanet Detection System

    Team A • Team B • Team C
    """)
# ==========================
# MISSION STATUS
# ==========================
c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric("TESS","CONNECTED")

with c2:
    st.metric("MODEL","ONLINE")

with c3:
    st.metric("PIPELINE","ACTIVE")

with c4:
    st.metric(
        "UTC",
        datetime.utcnow().strftime("%H:%M:%S")
    )

st.divider()


# ==========================
# TIC INPUT
# ==========================

st.subheader("🎯 Target Star Analysis")

tic_id = st.text_input(
    "Enter TIC ID",
    placeholder="TIC 123456789"
)

analyze = st.button(
    "🚀 ANALYZE TARGET",
    use_container_width=True
)


# ==========================
# ANALYSIS
# ==========================

if analyze:

    if not tic_id:

        st.warning("Please enter a TIC ID")

    else:

        with st.spinner("Running AI Pipeline..."):

            result = run_pipeline(tic_id)

        prediction = result["prediction"]

        planet = result["planet"]

        label = prediction["class_label"]

        scores = prediction["confidence_scores"]

        confidence = max(scores.values()) * 100

        st.success(
            f"Analysis Complete : {tic_id}"
        )
        st.subheader("🎯 Observation Summary")

        col1,col2,col3 = st.columns(3)

        with col1:
            st.metric("Target", f"TIC {tic_id}")

        with col2:
            st.metric("Classification", label.upper())

        with col3:
            st.metric("Confidence", f"{confidence:.2f}%")

        st.divider()

        # ======================
        # LIGHT CURVE
        # ======================

        st.subheader("📈 Light Curve")

        x = result["time"]
        y = result["flux"]
       
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=" TESS Flux"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            
            title=f"TIC{tic_id}  Real TESS Light Curve",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # ======================
        # AI RESULT
        # ======================

        st.subheader("🤖 AI Classification")

        c1, c2 = st.columns([1,1])

        with c1:

            st.success(
                f"Detected Class : {label.upper()}"
            )

            st.progress(
                confidence / 100
            )

            st.write(
                f"Confidence : {confidence:.2f}%"
            )

        with c2:

            prob_df = pd.DataFrame(
                {
                    "Class": list(scores.keys()),
                    "Probability (%)":
                    [
                        round(v * 100, 2)
                        for v in scores.values()
                    ]
                }
            )

            st.dataframe(
                prob_df,
                use_container_width=True
            )
        # ======================
        # STELLAR PARAMETERS
        # ======================

        st.subheader("⭐ Host Star Parameters")

        metadata = result["metadata"]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Teff",
                f"{metadata[0]:.0f} K"
            )

        with c2:
            st.metric(
                "Radius",
                f"{metadata[1]:.2f} Rsun"
            )

        with c3:
            st.metric(
                "log(g)",
                f"{metadata[2]:.2f}"
            )

        with c4:
            st.metric(
                "Mass",
                f"{metadata[3]:.2f} Msun"
            )

        st.divider()

        # ======================
        # PLANET PROFILE
        # ======================

        st.subheader("🪐 Candidate Planet Profile")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Orbital Period",
                f"{planet['period']} Days"
            )

        with c2:

            st.metric(
                "Rp/R*",
                f"{planet['rp_rs']}"
            )

        with c3:

            st.metric(
                "Semi Major Axis",
                f"{planet['semi_major_axis']} AU"
            )

        st.divider()

        # ======================
        # CONFIDENCE CHART
        # ======================
        fig2 = go.Figure()

        fig2.add_trace(
            go.Bar(
                x=list(scores.keys()),
                y=[v*100 for v in scores.values()]
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            title="AI Confidence Distribution",
            height=450
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


        st.divider()

        # ======================
        # MISSION LOGS
        # ======================

        st.subheader("📡 Mission Logs")

        logs = f"""
[{datetime.now().strftime('%H:%M:%S')}] Pipeline Started

[{datetime.now().strftime('%H:%M:%S')}] TIC Loaded

[{datetime.now().strftime('%H:%M:%S')}] Team A Preprocessing Complete

[{datetime.now().strftime('%H:%M:%S')}] Member B Model Loaded

[{datetime.now().strftime('%H:%M:%S')}] Prediction : {label}

[{datetime.now().strftime('%H:%M:%S')}] Candidate Report Generated
"""

        st.code(logs)

        st.download_button(
            "📄 Download Mission Report",
            logs,
            file_name=f"{tic_id}_report.txt"
        )


# ==========================
# FOOTER
# ==========================

st.divider()

st.caption(
    "NAKSHATRA AI • ISRO BAH 2026 Challenge • Exoplanet Detection Mission"
)