import gradio as gr
import pickle

# Load the trained model and TF-IDF vectorizer
with open("vectorizer", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def check_spam(message):
    if not message or not message.strip():
        return "⚠️ Please enter a message.", "—"

    transformed = tfidf.transform([message])
    prediction = model.predict(transformed)[0]

    # Show the probability for the predicted class
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index] * 100
        confidence_text = f"{confidence:.2f}%"
    else:
        confidence_text = "Not available"

    if prediction == 1:
        result = "🚨 SPAM DETECTED"
    else:
        result = "✅ NOT SPAM"

    return result, confidence_text


custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
#main-title {
    text-align: center;
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(90deg, #312e81, #164e63);
    border: 2px solid #38bdf8;
    margin-bottom: 20px;
}
#main-title h1 {
    color: #67e8f9 !important;
    font-size: 38px !important;
    font-weight: 900 !important;
    text-shadow: 0 2px 15px rgba(34, 211, 238, 0.5);
}
#main-title p {
    color: #ffffff !important;
    font-size: 17px !important;
}
#result {
    font-size: 23px;
    font-weight: bold;
    text-align: center;
}
"""

with gr.Blocks(title="Email Spam Detector Pro") as app:
    gr.Markdown("""
    <div id="main-title">
        <h1>🛡️ EMAIL SPAM DETECTOR PRO</h1>
        <p>AI-Powered Email Security | Machine Learning Project</p>
        <p>Detect Spam Messages Instantly!</p>
    </div>
    """)

    gr.Markdown("## ✉️ Message Analysis")

    with gr.Row():
        with gr.Column(scale=2):
            message = gr.Textbox(
                label="Enter Your Message",
                placeholder="Paste your email or SMS message here...",
                lines=7
            )
            with gr.Row():
                check_btn = gr.Button("🔍 CHECK MESSAGE", variant="primary")
                clear_btn = gr.ClearButton(
                    components=[message],
                    value="🗑️ CLEAR"
                )

        with gr.Column(scale=1):
            result = gr.Textbox(
                label="Prediction Result",
                elem_id="result",
                interactive=False
            )
            confidence = gr.Textbox(
                label="Model Confidence",
                interactive=False
            )

    gr.Markdown("## 🧪 Try Sample Messages")
    gr.Examples(
        examples=[
            ["Congratulations! You won a free lottery. Click here now!"],
            ["Hey bro, are you coming to college tomorrow?"],
            ["URGENT! Claim your cash prize now!"],
            ["Meeting scheduled for tomorrow at 10 AM."]
        ],
        inputs=message
    )

    check_btn.click(
        fn=check_spam,
        inputs=message,
        outputs=[result, confidence]
    )

    gr.Markdown("""
    ---
    <div style="text-align:center; padding:15px;">
        <h3>🛡️ Email Spam Detector Pro</h3>
        <p>Built with Python, Scikit-learn, TF-IDF, Naive Bayes and Gradio</p>
    </div>
    """)

if __name__ == "__main__":
    app.launch(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
        css=custom_css
    )
