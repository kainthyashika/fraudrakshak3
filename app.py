import streamlit as st
import numpy as np
import cv2
import librosa
import tempfile
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Fraud Rakshak", layout="wide")
st.title("Fraud Rakshak – AI Scam Detection")

# ------------------ 1. FACE DETECTION ------------------
st.header("1. Face Detection")
col1, col2 = st.columns(2)

with col1:
    real_face = st.file_uploader("Upload Real Face Image", type=["jpg", "jpeg", "png"], key="real_face")
with col2:
    ai_face = st.file_uploader("Upload AI-Generated Face", type=["jpg", "jpeg", "png"], key="ai_face")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(image_file):
    img_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return img, len(faces)

fraud_score = 0
if real_face and ai_face:
    img1, f1 = detect_face(real_face)
    img2, f2 = detect_face(ai_face)

    st.image(img1, caption=f"Real Face - {f1} face(s) detected", channels="BGR")
    st.image(img2, caption=f"AI Face - {f2} face(s) detected", channels="BGR")

    if f2 == 0:
        st.warning("AI Face likely generated (No face detected)")
        fraud_score += 40
    if f1 >= 1:
        st.success("Real face detected")

# ------------------ 2. VOICE COMPARISON ------------------
st.header("2. Voice Comparison")
col3, col4 = st.columns(2)

with col3:
    real_voice = st.file_uploader("Upload Real Voice", type=["wav", "mp3"], key="real_audio")
    if real_voice:
        st.audio(real_voice, format="audio/wav")
with col4:
    ai_voice = st.file_uploader("Upload AI Voice", type=["wav", "mp3"], key="ai_audio")
    if ai_voice:
        st.audio(ai_voice, format="audio/wav")

def extract_voice_features(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        path = tmp.name
    y, sr = librosa.load(path, sr=None)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    return zcr, centroid

if real_voice and ai_voice:
    real_zcr, real_cent = extract_voice_features(real_voice)
    ai_zcr, ai_cent = extract_voice_features(ai_voice)

    st.write(f"*Real Voice ZCR*: {real_zcr:.4f}, Centroid: {real_cent:.2f}")
    st.write(f"*AI Voice ZCR*: {ai_zcr:.4f}, Centroid: {ai_cent:.2f}")

    fig, ax = plt.subplots()
    labels = ["ZCR", "Spectral Centroid"]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, [real_zcr, real_cent], width, label="Real", color='green')
    ax.bar(x + width/2, [ai_zcr, ai_cent], width, label="AI", color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Voice Feature Comparison")
    ax.legend()
    st.pyplot(fig)

    if ai_zcr < 0.05:
        fraud_score += 30
    if ai_cent < 2000:
        fraud_score += 30

# ------------------ 3. TEXT SCAM ANALYSIS ------------------
st.header("3. Scam Text Analysis")
text_input = st.text_area("Paste a suspicious message or SMS")

scam_keywords = [
    r"urgent(ly)?", r"verify.*account", r"OTP", r"password", r"payment link", r"bank alert",
    r"you[' ]?won", r"click here", r"call now", r"limited time", r"suspended", r"unauthorized access",
    r"congratulations", r"claim.*reward", r"lucky winner"
]

risk_score = 0
if text_input:
    matches = []
    for pattern in scam_keywords:
        if re.search(pattern, text_input, re.IGNORECASE):
            matches.append(pattern)
            risk_score += 10

    st.write("### Detected Scam Patterns:")
    for m in matches:
        st.markdown(f"- *{m}*")

    st.write(f"### Text Scam Score: {risk_score}%")
    if risk_score >= 50:
        st.error("Likely a scam message.")
    elif risk_score > 0:
        st.warning("Suspicious content.")
    else:
        st.success("No obvious scam patterns.")

# ------------------ 4. FINAL VERDICT ------------------
st.header("4. Final Verdict")

final_score = fraud_score + risk_score
verdict = "Unknown"

if final_score >= 100:
    verdict = "Likely Fraud"
    st.error(f"Final Verdict: {verdict} ({final_score}%)")
elif final_score >= 60:
    verdict = "Suspicious"
    st.warning(f"Final Verdict: {verdict} ({final_score}%)")
elif final_score > 0:
    verdict = "Possibly Safe"
    st.info(f"Final Verdict: {verdict} ({final_score}%)")
else:
    verdict = "Safe"
    st.success(f"Final Verdict: {verdict} ({final_score}%)")

# ------------------ 5. BLOCK AND REPORT ------------------
st.subheader("Actions")

col_block, col_report = st.columns(2)

with col_block:
    if st.button("Block Caller / Sender"):
        st.warning("Caller/Sender Blocked. Actions logged locally (simulated).")

with col_report:
    if st.button("Auto Generate Report"):
        st.info("*Fraud Rakshak Report*")
        st.markdown(f"""
        - *Face Detection Score*: {fraud_score if fraud_score else 'N/A'}
        - *Voice Analysis Score*: {fraud_score if fraud_score else 'N/A'}
        - *Text Scam Score*: {risk_score if risk_score else 'N/A'}
        - *Final Verdict: *{verdict}**
        - *Total Fraud Score*: {final_score}%
        """)
