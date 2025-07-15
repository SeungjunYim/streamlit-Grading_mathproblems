import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import torch.nn.functional as F
import os

# -------------------------------
# 모델 로드 함수
# -------------------------------
@st.cache_resource
def load_model():
    model = torch.load("model_mnist.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

# -------------------------------
# 이미지 전처리 함수
# -------------------------------
def preprocess_image(image):
    image = image.convert("L")                      # 흑백 변환
    image = ImageOps.invert(image)                  # 흰 배경 검정 글씨로 반전
    image = image.resize((28, 28))                  # MNIST와 같은 크기
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)            # 배치 차원 추가

# -------------------------------
# 예측 함수 (MC Dropout 50회 평균)
# -------------------------------
def predict_digit(model, image_tensor):
    outputs = []
    with torch.no_grad():
        for _ in range(50):
            output = model(image_tensor)
            outputs.append(F.softmax(output, dim=1))
    outputs = torch.stack(outputs)
    probs = outputs.mean(0).squeeze()
    prediction = torch.argmax(probs).item()
    confidence = probs[prediction].item()
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    return prediction, confidence, entropy

# -------------------------------
# Streamlit UI 구성
# -------------------------------
st.set_page_config(page_title="수학 문제 채점기", page_icon="➗")
st.markdown("""
# <span style='color:#00FFAA'>54  ÷  9  =  ?</span>
답을 손글씨로 적은 이미지를 업로드하세요
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
student_id = st.text_input("학습자 ID 입력")

if uploaded_file and student_id:
    image = Image.open(uploaded_file)
    st.image(image, caption="입력 이미지", width=150)
    input_tensor = preprocess_image(image)

    model = load_model()
    prediction, confidence, entropy = predict_digit(model, input_tensor)

    st.markdown(f"**예측 결과:** {prediction}")
    st.markdown(f"**신뢰도 (Confidence):** {confidence:.4f}")
    st.markdown(f"**불확실도 (Entropy):** {entropy:.4f}")

    correct_answer = 6
    CONFIDENCE_THRESHOLD = 0.80
    ENTROPY_THRESHOLD = 0.75

    if confidence >= CONFIDENCE_THRESHOLD and entropy <= ENTROPY_THRESHOLD:
        if prediction == correct_answer:
            st.success("정답입니다! 채점 완료 ✅")
        else:
            st.error("오답입니다. 다시 풀어보세요 ❌")
    else:
        st.warning("예측이 불확실하거나 신뢰도가 낮습니다. 답을 다시 써서 업로드 해주세요.")
