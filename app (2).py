
import streamlit as st
import pickle

# Page configuration
st.set_page_config(
    page_title="Email Spam Detector Pro",
    page_icon="🛡️",
    layout="wide"
)

# Load trained model and vectorizer
with open("vectorizer", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# Prediction function
def check_spam(message):
    if not message or not message.strip():
        return None, None

    transformed = tfidf.transform([message])
    prediction = model.predict(transformed)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index] * 100
    else:
        confidence = None

    return prediction, confidence


# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.title-box {
    text-align: center;
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(90deg, #312e81, #164e63);
    border: 2px solid #38bdf8;
    margin-bottom: 25px;
}
.title-box h1 {
    color: #67e8f9;
    font-size: 38px;
    font-weight: 900;
}
.title-box p {
    color: white;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)


# Main heading
st.markdown("""
<div class="title-box">
    <h1>🛡️ EMAIL SPAM DETECTOR PRO</h1>
    <p>AI-Powered Email Security | Machine Learning Project</p>
    <p>Detect Spam Messages Instantly!</p>
</div>
""", unsafe_allow_html=True)


st.header("✉️ Message Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    message = st.text_area(
        "Enter Your Message",
        placeholder="Paste your email or SMS message here...",
        height=180
    )

    check_btn = st.button(
        "🔍 CHECK MESSAGE",
        use_container_width=True
    )

with col2:
    st.subheader("🧪 Try Sample Messages")

    samples = [
        "Congratulations! You won a free lottery. Click here now!",
        "Hey bro, are you coming to college tomorrow?",
        "URGENT! Claim your cash prize now!",
        "Meeting scheduled for tomorrow at 10 AM."
    ]

    for sample in samples:
        if st.button(sample, use_container_width=True):
            st.session_state.message = sample
            st.rerun()


# Use selected sample if available
if "message" in st.session_state:
    message = st.session_state.message

# Prediction
if check_btn:
    if not message.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        prediction, confidence = check_spam(message)

        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.error("🚨 SPAM DETECTED")
        else:
            st.success("✅ NOT SPAM")

        if confidence is not None:
            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )


st.divider()

st.markdown("""
<div style="text-align:center;">
    <h3>🛡️ Email Spam Detector Pro</h3>
    <p>Built with Python, Scikit-learn, TF-IDF and Naive Bayes</p>
</div>
""", unsafe_allow_html=True)
