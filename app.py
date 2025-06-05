import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

rf_model = joblib.load("parkinsons_model.pkl") 
svm_model = joblib.load("svm_model.pkl")       
scaler = joblib.load("scaler.pkl")

features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

st.set_page_config("Parkinson's Detection", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def change_page(page):
    st.session_state.page = page

st.markdown("""
<style>
.sidebar .stButton>button {
    width: 100%;
    background-color: #1c2023;
    color: white;
    font-weight: 600;
    border-radius: 6px;
    padding: 0.7em 1em;
    margin: 0.4em 0;
    border: none;
    transition: background-color 0.3s ease;
}
.sidebar .stButton>button:hover {
    background-color: #446D8C;
}
.sidebar-title {
    font-size: 1.4em;
    font-weight: bold;
    margin-bottom: 1em;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button('Home'):
        change_page('Home')
with col2:
    if st.button('Predict'):
        change_page('Predict')
with col3:
    if st.button('About'):
        change_page('About')

st.markdown("<h1 style='text-align: center;'>Parkinson's Detection from Voice</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if st.session_state.page == "Home":
    st.subheader("Welcome!")
    st.markdown("""
    This application detects the likelihood of **early-stage Parkinson's Disease** using 22 voice measurements.
    
    **How it works**:
    - Manually input voice features or upload a CSV file.
    - Choose between machine learning models for prediction.
    - Visualize and download results.
    """)

elif st.session_state.page == "Predict":
    tab1, tab2 = st.tabs([" Manual Input", " Upload CSV"])

    st.sidebar.markdown("### Choose Model")
    selected_model = st.sidebar.selectbox("Model", ["Random Forest", "SVM"])
    model = rf_model if selected_model == "Random Forest" else svm_model

    with tab1:
        input_data = []
        with st.form("manual_form"):
            st.write("Enter the 22 required voice measurements below:")
            for feature in features:
                val = st.number_input(feature, min_value=0.0, value=0.0, format="%.6f")
                input_data.append(val)
            submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                input_df = pd.DataFrame([input_data], columns=features)
                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)[0]
                confidence = model.predict_proba(scaled_input)[0][prediction]

                if prediction == 1:
                    st.error(f"Likely signs of Parkinson's detected. Confidence: **{confidence:.2%}**")
                else:
                    st.success(f"No significant signs of Parkinson's detected. Confidence: **{confidence:.2%}**")
                st.progress(confidence)

                if selected_model == "Random Forest":
                    st.markdown("#### Feature Importance")
                    importances = model.feature_importances_
                    sorted_idx = np.argsort(importances)[::-1]
                    plt.figure(figsize=(8, 6))
                    sns.barplot(x=importances[sorted_idx][:10], y=np.array(features)[sorted_idx][:10])
                    plt.title("Top 10 Important Features")
                    st.pyplot(plt)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    with tab2:
        uploaded_file = st.file_uploader("Upload a CSV file with 22 features", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                df.columns = df.columns.str.strip()
                missing = [f for f in features if f not in df.columns]

                if missing:
                    st.warning(f"Missing features: {missing}")
                else:
                    scaled_inputs = scaler.transform(df[features])
                    preds = model.predict(scaled_inputs)
                    probs = model.predict_proba(scaled_inputs).max(axis=1)

                    results = pd.DataFrame({
                        "Prediction": ["Parkinson's" if p == 1 else "Healthy" for p in preds],
                        "Confidence": [f"{p:.2%}" for p in probs]
                    })

                    st.success("Predictions completed:")
                    st.dataframe(results)

                    import plotly.express as px
                    counts = results["Prediction"].value_counts().reset_index()
                    counts.columns = ["Prediction", "Count"]

                    fig = px.bar(
                        counts,
                        x="Prediction",
                        y="Count",
                        color="Prediction",
                        color_discrete_map={"Healthy": "#4BB543", "Parkinson's": "#FF4B4B"},
                        title="Prediction Summary",
                        width=300,
                        height=250
                    )
                    fig.update_layout(
                        title_font_size=14,
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title=None,
                        yaxis_title=None,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=False)

            except Exception as e:
                st.error(f"Upload error: {e}")


elif st.session_state.page == "About":
    tab1, tab2 = st.tabs([" Project Info", " Educational Content"])

    with tab1:
        st.subheader("About the Project")
        st.markdown("""
        This project uses a machine learning model trained on the **UCI Parkinson's Dataset** to analyze vocal biomarker data.
        
        - Dataset: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons)
        - Model: Random Forest Classifier + SVM
        - Features: 22 vocal measurements
        
        Built using **Streamlit** and **scikit-learn**.
        """)

    with tab2:
        st.subheader("Understanding Parkinson's Disease")
        st.markdown("""
        Parkinson’s Disease is a neurodegenerative disorder affecting movement and speech. Early diagnosis can help manage symptoms better.

        - **Voice Impairment** is a key early symptom.
        - Changes in pitch, tone, jitter, and shimmer can indicate early onset.
        
        Machine learning can assist in early detection through vocal analysis, as demonstrated in this app.
        """)
