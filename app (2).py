
import streamlit as st
import pickle

st.set_page_config(
    page_title="Email Spam Detector Pro",
    page_icon="🛡️",
    layout="wide"
)

# Load model and vectorizer
with open("vectorizer", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def check_spam(message):
    if not message or not message.strip():
        return None, None

    transformed = tfidf.transform([message])
    prediction = model.predict(transformed)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index] * 100

    return prediction, confidence


# Custom dark theme
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080f20, #111c35, #0b2535);
    color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.title-box {
    text-align: center;
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(90deg, #312e81, #164e63);
    border: 2px solid #38bdf8;
    margin-bottom: 30px;
    box-shadow: 0 0 25px rgba(56,189,248,0.15);
}

.title-box h1 {
    color: #67e8f9;
    font-size: 36px;
    font-weight: 900;
}

.title-box p {
    color: white;
    font-size: 16px;
}

/* Text area */
.stTextArea textarea {
    background-color: #111c35 !important;
    color: #ffffff !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 12px !important;
    caret-color: #67e8f9;
}

.stTextArea textarea::placeholder {
    color: #94a3b8 !important;
}

/* Labels */
label, .stMarkdown p {
    color: #e2e8f0 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #164e63, #312e81) !important;
    color: #ffffff !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 10px !important;
    font-weight: bold !important;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0e7490, #4338ca) !important;
    border: 1px solid #67e8f9 !important;
    color: white !important;
}

/* Result cards */
.result-card {
    background: #111c35;
    border: 1px solid #38bdf8;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 15px;
}

.result-card h2 {
    color: #67e8f9;
}

[data-testid="stMetricValue"] {
    color: #67e8f9;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #334155;
}
</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class="title-box">
    <h1>🛡️ EMAIL SPAM DETECTOR PRO</h1>
    <p>AI-Powered Email Security | Machine Learning Project</p>
    <p>Detect Spam Messages Instantly!</p>
</div>
""", unsafe_allow_html=True)


# Main layout
col1, col2 = st.columns([2, 1], gap="large")

if "message" not in st.session_state:
    st.session_state.message = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

with col1:
    st.subheader("✉️ Message Analysis")

    message = st.text_area(
        "Enter Your Message",
        key="message",
        placeholder="Paste your email or SMS message here...",
        height=190
    )

    if st.button("🔍 CHECK MESSAGE", use_container_width=True):
        if not message.strip():
            st.warning("⚠️ Please enter a message.")
            st.session_state.prediction = None
        else:
            prediction, confidence = check_spam(message)
            st.session_state.prediction = prediction
            st.session_state.confidence = confidence

with col2:
    st.subheader("🧪 Try Sample Messages")

    samples = [
        "Congratulations! You won a free lottery. Click here now!",
        "Hey bro, are you coming to college tomorrow?",
        "URGENT! Claim your cash prize now!",
        "Meeting scheduled for tomorrow at 10 AM."
    ]

    for i, sample in enumerate(samples):
        if st.button(
            f"Sample {i+1}: {sample}",
            key=f"sample_{i}",
            use_container_width=True
        ):
            st.session_state.message = sample
            st.session_state.prediction = None
            st.rerun()

# Display prediction
if st.session_state.prediction is not None:
    st.divider()
    st.subheader("📊 Prediction Result")

    if st.session_state.prediction == 1:
        st.error("🚨 SPAM DETECTED")
    else:
        st.success("✅ NOT SPAM")

    if st.session_state.confidence is not None:
        st.metric(
            "Model Confidence",
            f"{st.session_state.confidence:.2f}%"
        )

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px;">
    <h3 style="color:#67e8f9;">🛡️ Email Spam Detector Pro</h3>
    <p>Built with Python, Scikit-learn, TF-IDF and Naive Bayes</p>
</div>
""", unsafe_allow_html=True)
