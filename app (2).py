
import streamlit as st
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Email Spam Detector Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD ML MODEL ----------------
@st.cache_resource
def load_model():
    with open("vectorizer", "rb") as f:
        tfidf = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    return tfidf, model


try:
    tfidf, model = load_model()
except Exception as e:
    st.error("⚠️ Could not load the ML model or vectorizer.")
    st.code("app.py\nmodel.pkl\nvectorizer")
    st.stop()


# ---------------- SAMPLE CALLBACK ----------------
def set_sample(sample):
    st.session_state.message = sample


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #080d1b, #101a30, #101827);
    color: #ffffff;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #172554, #1e3a8a, #312e81);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #334c87;
    margin-bottom: 30px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 10px;
    font-weight: 800;
}

.hero p {
    color: #dbeafe;
    font-size: 16px;
    margin-bottom: 0;
}

.section-title {
    color: #93c5fd;
    font-size: 23px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 15px;
}

.info-card {
    background: #17243b;
    border: 1px solid #304362;
    border-radius: 15px;
    padding: 20px;
    margin-top: 15px;
    margin-bottom: 15px;
}

div[data-testid="stTextArea"] textarea {
    background-color: #17243b !important;
    color: white !important;
    border: 1px solid #405575 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: #94a3b8 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    border: 1px solid #4f6fbd;
    border-radius: 10px;
    padding: 12px;
    font-weight: 650;
    min-height: 48px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #4338ca);
    border: 1px solid #93c5fd;
    color: white;
}

div[data-testid="stMetric"] {
    background: #17243b;
    border: 1px solid #304362;
    padding: 18px;
    border-radius: 14px;
}

.confidence-card {
    background: linear-gradient(135deg, #172554, #17243b);
    border: 1px solid #3b82f6;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
}

.confidence-number {
    font-size: 52px;
    font-weight: 800;
    color: #60a5fa;
    margin: 5px 0;
}

.confidence-label {
    color: #cbd5e1;
    font-size: 17px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 25px;
    margin-top: 35px;
    border-top: 1px solid #304362;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🛡️ EMAIL SPAM DETECTOR PRO</h1>
    <p>AI-Powered Email Security | Machine Learning Project</p>
    <p>Detect Spam Messages Instantly!</p>
</div>
""", unsafe_allow_html=True)


# ---------------- MESSAGE ANALYSIS ----------------
st.markdown(
    '<div class="section-title">✉️ Message Analysis</div>',
    unsafe_allow_html=True
)

if "message" not in st.session_state:
    st.session_state.message = ""


# ---------------- SAMPLE MESSAGES ----------------
st.markdown(
    '<div class="section-title">🧪 Try Sample Messages</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-card">
    <b>Test your spam detector!</b><br>
    Select any example below. The message will automatically appear
    in the input box.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.button(
        "🎁 Winning Prize Message",
        key="sample1",
        use_container_width=True,
        on_click=set_sample,
        args=("Congratulations! You have won a free prize. Click here to claim your reward now!",)
    )

    st.button(
        "💰 Lottery Message",
        key="sample2",
        use_container_width=True,
        on_click=set_sample,
        args=("URGENT! You have won $1000 in our lottery. Send your bank details to claim your money now!",)
    )

with col2:
    st.button(
        "📅 Meeting Reminder",
        key="sample3",
        use_container_width=True,
        on_click=set_sample,
        args=("Hello, our project meeting is scheduled for tomorrow at 10 AM. Please join on time.",)
    )

    st.button(
        "📧 Normal Email",
        key="sample4",
        use_container_width=True,
        on_click=set_sample,
        args=("Hi, please find the project report attached. Let me know if you have any questions. Thank you.",)
    )


# ---------------- TEXT INPUT ----------------
message = st.text_area(
    "Enter your email message below:",
    key="message",
    height=180,
    placeholder="Paste your email message here..."
)


# ---------------- PREDICTION ----------------
st.markdown("---")

if st.button(
    "🔍 ANALYZE MESSAGE",
    use_container_width=True,
    type="primary"
):

    if not message.strip():
        st.warning("⚠️ Please enter a message before analyzing.")

    else:
        try:
            transformed_message = tfidf.transform([message])

            prediction = model.predict(transformed_message)[0]

            prediction_text = str(prediction).strip().lower()

            is_spam = prediction_text in [
                "spam", "1", "true", "yes"
            ]

            # ---------------- CONFIDENCE ----------------
            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(
                    transformed_message
                )[0]

                confidence = max(probabilities) * 100

            elif hasattr(model, "decision_function"):
                st.info(
                    "This model does not provide probability estimates. "
                    "Confidence percentage is unavailable."
                )

            # ---------------- RESULT ----------------
            st.markdown("---")

            st.markdown(
                '<div class="section-title">📊 Analysis Result</div>',
                unsafe_allow_html=True
            )

            if is_spam:

                st.error("🚨 SPAM MESSAGE DETECTED!")

                st.markdown("""
                <div class="info-card">
                    <h2 style="color:#f87171;">🚨 This is a Spam Email</h2>
                    <p style="color:#e2e8f0;">
                    This message has been classified as spam.
                    Be careful before clicking links or sharing personal information.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.success("✅ NOT SPAM - SAFE MESSAGE")

                st.markdown("""
                <div class="info-card">
                    <h2 style="color:#4ade80;">✅ This is a Normal Email</h2>
                    <p style="color:#e2e8f0;">
                    This message has been classified as legitimate (ham).
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # ---------------- CONFIDENCE DISPLAY ----------------
            if confidence is not None:

                st.markdown(f"""
                <div class="confidence-card">
                    <div class="confidence-label">
                        🎯 MODEL PREDICTION CONFIDENCE
                    </div>
                    <div class="confidence-number">
                        {confidence:.2f}%
                    </div>
                    <div class="confidence-label">
                        Confidence in predicted class
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(int(confidence))

            else:
                st.warning(
                    "Prediction confidence is unavailable for this model."
                )

            st.caption(
                "Note: Prediction confidence is not the same as model accuracy."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <h4>🛡️ Email Spam Detector Pro</h4>
    <p>Built with Python | Scikit-learn | TF-IDF | Streamlit</p>
    <p>Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)
