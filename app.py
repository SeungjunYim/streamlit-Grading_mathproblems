import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import io
import math

# -----------------------------
# 모델 클래스 정의 (MNIST용 CNN + Dropout)
# -----------------------------
class BayesianCNN(nn.Module):
    def __init__(self):
        super(BayesianCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

# -----------------------------
# 모델 불러오기
# -----------------------------
@st.cache_resource
def load_model():
    model = BayesianCNN()
    model.load_state_dict(torch.load("model_mnist.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# -----------------------------
# 이미지 전처리 함수
# -----------------------------
def preprocess_image(image):
    image = image.convert("L")  # grayscale
    image = image.resize((28, 28))
    image_np = np.array(image)
    image_np = 255 - image_np  # 반전
    image_np = image_np / 255.0  # 정규화
    image_tensor = torch.tensor(image_np).unsqueeze(0).unsqueeze(0).float()
    return image_tensor

# -----------------------------
# MC Dropout 기반 예측
# -----------------------------
def predict_with_uncertainty(model, image_tensor, n_iter=20):
    model.train()  # dropout 활성화
    outputs = []
    with torch.no_grad():
        for _ in range(n_iter):
            output = model(image_tensor)
            outputs.append(torch.exp(output))  # softmax 확률
    probs = torch.stack(outputs)
    mean_prob = probs.mean(dim=0).squeeze()
    entropy = -torch.sum(mean_prob * torch.log(mean_prob + 1e-10)).item()
    confidence = torch.max(mean_prob).item()
    prediction = torch.argmax(mean_prob).item()
    return prediction, confidence, entropy

# -----------------------------
# Streamlit 웹 UI
# -----------------------------
st.set_page_config(page_title="수학 문제 채점기", page_icon="📐")
st.markdown("""
    <h2 style='color:#10FF90'>54 ÷ 9 = ?</h2>
    <p>답을 손글씨로 적은 이미지를 업로드하세요 </p>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
user_id = st.text_input("학습자 ID 입력")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="입력 이미지", width=150)

    image_tensor = preprocess_image(image)
    prediction, confidence, entropy = predict_with_uncertainty(model, image_tensor)

    st.markdown(f"**예측 결과:** {prediction}")
    st.markdown(f"**신뢰도 (Confidence):** {confidence:.4f}")
    st.markdown(f"**불확실도 (Entropy):** {entropy:.4f}")

    # 정답 비교: 54 / 9 = 6
    correct_answer = 6
    CONF_THRESH = 0.65
    ENTROPY_THRESH = 0.9

    if confidence >= CONF_THRESH and entropy <= ENTROPY_THRESH:
        if prediction == correct_answer:
            st.success("✅ 정답입니다!")
        else:
            st.error("❌ 오답입니다.")
    else:
        st.warning("⚠️ 예측이 불확실하거나 신뢰도가 낮습니다. 답을 다시 써서 업로드 해주세요.")
