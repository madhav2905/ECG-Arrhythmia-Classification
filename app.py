import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/cnn_lstm_best_model.keras")
    return model

model = load_model()
class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unclassifiable']

st.title("ECG Arrhythmia Classifier (CNN + LSTM)")
st.markdown("Upload or simulate ECG data and classify arrhythmia types in real-time.")

uploaded_file = st.file_uploader("Upload ECG Signal (.npy or .txt file)", type=['npy', 'txt'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.npy'):
        signal = np.load(uploaded_file)
    else:
        signal = np.loadtxt(uploaded_file)
    
    signal = signal.reshape(1, 187, 1)

    preds = model.predict(signal)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = preds[0][predicted_class] * 100

    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {class_names[predicted_class]} ({predicted_class})")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("ECG Signal Visualization")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(signal.flatten(), color='blue')
    ax.set_title("ECG Signal")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    st.subheader("Model Confidence for Each Class")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars = ax2.bar(class_names, preds[0] * 100, color='seagreen', alpha=0.8)
    bars[predicted_class].set_color('gold')
    ax2.set_ylim([0, 100])
    ax2.set_ylabel("Confidence (%)")
    for i, bar in enumerate(bars):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{bar.get_height():.2f}%", ha='center', fontsize=10)
    st.pyplot(fig2)