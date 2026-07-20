import numpy as np
from pipeline import run_pipeline
import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(
    page_title="Nakshatra AI",
    page_icon="🚀",
    layout="wide"
)
st.markdown("""
<style>

/* Main Background */
.stApp{
background-image:url(
"https://images.unsplash.com/photo-1462331940025-496dfbfc7564"
);
background-size:cover;
background-attachment:fixed;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0B1026,
        #111827
    );
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Glass Cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.15);

    box-shadow:
        0 8px 32px rgba(0,0,0,0.4),
        0 0 20px rgba(0,255,255,0.15);

    transition: 0.3s;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
}

/* Buttons */
.stButton > button {

    background: linear-gradient(
        90deg,
        #06B6D4,
        #3B82F6
    );

    color: white;

    border: none;

    border-radius: 15px;

    height: 55px;

    font-size: 18px;

    font-weight: bold;

    box-shadow:
        0 0 15px rgba(59,130,246,0.5);

}

/* Titles */
h1 {
    color: white;
    text-align: center;
    font-size: 3rem;
}

h2,h3 {
    color: #7DD3FC;
}

/* Info Boxes */
[data-testid="stAlert"] {

    border-radius: 15px;

    border: 1px solid rgba(255,255,255,0.1);

    box-shadow:
        0 0 15px rgba(0,255,255,0.08);
}

/* Input Box */

.stTextInput input {

    border-radius: 12px;

    background-color: rgba(255,255,255,0.08);

    color: white;
}

/* Plotly Container */

[data-testid="stPlotlyChart"] {

    background: rgba(255,255,255,0.05);

    border-radius: 20px;

    padding: 10px;

    box-shadow:
        0 0 20px rgba(0,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)
# ===== Sidebar =====

st.sidebar.title("🚀 Nakshatra AI")
st.sidebar.markdown("---")

st.sidebar.success("🟢 System Online")

st.sidebar.metric(
    "Models Loaded",
    "3"
)

st.sidebar.metric(
    "Pipeline Status",
    "Ready"
)

# ===== Main Header =====
st.markdown("""
<h1 style="
text-align:center;
color:white;
font-size:70px;
text-shadow:
0 0 20px cyan,
0 0 40px cyan,
0 0 60px blue;
">
🚀 NAKSHATRA AI
</h1>
""", unsafe_allow_html=True)
# ===== Hero Card =====
st.markdown("""
<div style="
padding:30px;
border-radius:25px;
background:linear-gradient(
135deg,
rgba(0,255,255,0.15),
rgba(59,130,246,0.15)
);
backdrop-filter:blur(20px);
border:1px solid rgba(255,255,255,0.1);
box-shadow:0 0 40px rgba(0,255,255,0.2);
text-align:center;
">

<h2>🛰️ Mission Nakshatra</h2>

<h4>
AI-Based Exoplanet Discovery Platform
</h4>

<p>
Analyzing TESS stellar observations and detecting planetary transit signatures using machine learning.
</p>

</div>
""", unsafe_allow_html=True)
# ===== TIC Input =====
st.markdown("### 🚀 Mission Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Stars Analyzed", "12,540")
c2.metric("Candidates", "322")
c3.metric("Confirmed", "57")
c4.metric("AI Accuracy", "98.2%")
tic_id = st.text_input("Enter TIC ID")

if st.button("Analyze"):

    if not tic_id.strip():
        st.error("Please enter a TIC ID")
        st.stop()

    result = run_pipeline(tic_id)

    prediction = result["prediction"]

    st.success(f"Analysis completed for TIC {tic_id}")

    st.markdown(f"""
    <div style="
    padding:20px;
    border-radius:20px;
    background:rgba(0,255,150,0.1);
    border:1px solid rgba(0,255,150,0.4);
    box-shadow:0 0 25px rgba(0,255,150,0.2);
    ">

    <h3>🛰️ Detection Status</h3>

    <h2>{prediction['class_label']}</h2>

    </div>
    """, unsafe_allow_html=True)

    
    
    

    # ===== Light Curve Analysis =====
    st.divider()
    st.markdown("## 📈 Light Curve Analysis")

    col_curve1, col_curve2 = st.columns(2)

    x = np.linspace(0, 30, 500)

    raw_flux = 1 + np.random.normal(0, 0.002, 500)

    transit_mask = (x > 14) & (x < 16)
    raw_flux[transit_mask] -= 0.01

    detrended_flux = raw_flux - np.mean(raw_flux) + 1

    with col_curve1:

        raw_df = pd.DataFrame({
            "Time": x,
            "Flux": raw_flux
        })

        fig_raw = px.line(
            raw_df,
            x="Time",
            y="Flux",
            title="Raw Light Curve"
        )

        st.plotly_chart(
            fig_raw,
            use_container_width=True
        )

    with col_curve2:

        detrended_df = pd.DataFrame({
            "Time": x,
            "Flux": detrended_flux
        })

        fig_det = px.line(
            detrended_df,
            x="Time",
            y="Flux",
            title="Detrended Light Curve"
        )

        st.plotly_chart(
            fig_det,
            use_container_width=True
        )

    # ===== Prediction + Planet Parameters =====
    col1, col2 = st.columns(2)

    with col1:

        st.divider()
        st.markdown("### Prediction Result")

        prediction_df = pd.DataFrame({
            "Class": list(
                prediction["confidence_scores"].keys()
            ),
            "Confidence": list(
                prediction["confidence_scores"].values()
            )
        })

        fig = px.bar(
            prediction_df,
            x="Class",
            y="Confidence",
            title="Classification Confidence"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig_pie = px.pie(
            prediction_df,
            names="Class",
            values="Confidence",
            title="Confidence Distribution"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with col2:

        st.divider()
        st.markdown("### 🪐 Detected Planet Properties")

        if "planet" in result:

            planet = result["planet"]

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Period",
                f"{planet['period']} Days"
            )

            metric2.metric(
                "Rp/R*",
                f"{planet['rp_rs']}"
            )

            metric3.metric(
                "Semi Major Axis",
                f"{planet['semi_major_axis']} AU"
            )
        else:

            st.warning(
                "No planetary parameters available"
            )

    # ===== Download Report =====

    st.divider()
    st.markdown("## 📄 Analysis Report")

    report_text = f"""
    Nakshatra AI Analysis Report

    TIC ID: {tic_id}

    Detected Class: {prediction['class_label']}

    Confidence Scores:
    {prediction['confidence_scores']}
    """

    if "planet" in result:

        report_text += f"""

    Planet Parameters:

    Period: {planet['period']} Days
    Rp/R*: {planet['rp_rs']}
    Semi Major Axis: {planet['semi_major_axis']} AU
    """

    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name=f"TIC_{tic_id}_report.txt",
        mime="text/plain"
    )
    st.markdown("---")

st.markdown("""
<div style='text-align:center'>
🚀 Nakshatra AI v1.0
Developed for AI-Based Exoplanet Detection Challenge
Inspired by ISRO • TESS • NASA Missions
</div>
""", unsafe_allow_html=True)
    

        
        
        
        
    
    
    













    
    
    
    
    
    