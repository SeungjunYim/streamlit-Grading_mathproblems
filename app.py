import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------
# Bayesian CNN 모델 정의
# ------------------------
class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc1(x)

# ------------------------
# 이미지 전처리 함수
# ------------------------
def preprocess(image):
    image = image.convert('L').resize((28, 28))
    tensor = transforms.ToTensor()(image)
    return tensor.unsqueeze(0)

# ------------------------
# 예측 함수 (MC Dropout)
# ------------------------
def predict_with_uncertainty(model, image, n_iter=30):
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_iter):
            out = model(image)
            prob = torch.softmax(out, dim=1)
            preds.append(prob.cpu().numpy())
    preds = np.array(preds)
    mean = preds.mean(axis=0).squeeze()
    entropy = -np.sum(mean * np.log(mean + 1e-10))
    confidence = float(np.max(mean))
    predicted_label = int(np.argmax(mean))
    return predicted_label, confidence, entropy

# ------------------------
# 앱 시작
# ------------------------
st.set_page_config(page_title="수학 문제 채점기 (Bayesian AI)", layout="centered")
st.title("수학 문제 손글씨 자동 채점기")
st.markdown("다음 문제를 풀어 손글씨로 작성하고 사진을 업로드하세요:\n\n### `54 ÷ 9 = ?`")

uploaded_file = st.file_uploader("답을 손글씨로 적은 이미지를 업로드하세요 (숫자 하나)", type=["png", "jpg", "jpeg"])
user_id = st.text_input("학습자 ID 입력", "")

@st.cache_resource
def load_model():
    model = BayesianCNN()
    model.load_state_dict(torch.load("model_mnist.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

if uploaded_file and user_id:
    image = Image.open(uploaded_file)
    tensor = preprocess(image)

    label, conf, entropy = predict_with_uncertainty(model, tensor)

    st.image(image, caption="입력 이미지", width=150)
    st.write(f"**예측 결과:** {label}")
    st.write(f"**신뢰도 (Confidence):** {conf:.4f}")
    st.write(f"**불확실도 (Entropy):** {entropy:.4f}")

    # 채점 조건 확인
    if conf < 0.7 or entropy > 1.5:
        st.warning("⚠️ 예측이 불확실하거나 신뢰도가 낮습니다. 답을 다시 써서 업로드 해주세요.")
    else:
        if label == 6:
            st.success("✅ 정답입니다!")
        else:
            st.error("❌ 오답입니다.")

        # 기록 저장
        record = {
            "user_id": user_id,
            "image_file": uploaded_file.name,
            "prediction": int(label),
            "confidence": float(conf),
            "entropy": float(entropy),
            "is_correct": label == 6
        }
        os.makedirs("records", exist_ok=True)
        with open(f"records/{user_id}_grading_log.json", "a") as f:
            f.write(json.dumps(record) + "\n")

# ------------------------
# 기록 조회
# ------------------------
st.markdown("---")
st.subheader("예측 기록 분석")
selected_user = st.text_input("기록을 조회할 학습자 ID 입력", "")

if selected_user:
    filepath = f"records/{selected_user}_grading_log.json"
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        st.dataframe(df)

        st.write("### Confidence & Entropy 그래프")
        fig, ax = plt.subplots()
        ax.plot(df["confidence"], label="Confidence")
        ax.plot(df["entropy"], label="Entropy")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("해당 ID의 기록이 존재하지 않습니다.")
