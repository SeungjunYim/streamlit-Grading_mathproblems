import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import json
import os
import datetime

# ------------------ 모델 정의 ------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ------------------ 불러오기 ------------------
@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load("model_mnist.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

def mc_dropout_predict(model, image_tensor, n_iter=20):
    model.train()  # Dropout 활성화
    outputs = []
    with torch.no_grad():
        for _ in range(n_iter):
            output = model(image_tensor)
            outputs.append(F.softmax(output, dim=1))
    stacked = torch.stack(outputs)
    mean_prob = stacked.mean(dim=0).squeeze()
    entropy = -torch.sum(mean_prob * torch.log(mean_prob + 1e-6)).item()
    confidence = mean_prob.max().item()
    predicted = mean_prob.argmax().item()
    return predicted, confidence, entropy

# ------------------ 이미지 전처리 ------------------
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# ------------------ 기록 저장/조회 ------------------
def save_record(user_id, predicted, confidence, entropy, correct, status):
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted": predicted,
        "confidence": round(confidence, 4),
        "entropy": round(entropy, 4),
        "correct": correct,
        "status": status
    }
    if os.path.exists("records.json"):
        with open("records.json", "r") as f:
            data = json.load(f)
    else:
        data = {}
    if user_id not in data:
        data[user_id] = []
    data[user_id].append(record)
    with open("records.json", "w") as f:
        json.dump(data, f, indent=2)

def load_user_history(user_id):
    if not os.path.exists("records.json"):
        return []
    with open("records.json", "r") as f:
        data = json.load(f)
    return data.get(user_id, [])

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="수학 문제 채점기", page_icon="➗")
st.title("📐 수학 문제 채점기")
st.markdown("**54 ÷ 9 = ?**")

st.write("답을 손글씨로 적은 이미지를 업로드하세요")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
user_id = st.text_input("학습자 ID 입력")

model = load_model()

if uploaded_file and user_id:
    image = Image.open(uploaded_file)
    st.image(image, caption="입력 이미지", width=100)

    image_tensor = preprocess_image(image)
    pred, conf, entropy = mc_dropout_predict(model, image_tensor)

    st.write(f"**예측 결과:** {pred}")
    st.write(f"신뢰도 (Confidence): {conf:.4f}")
    st.write(f"불확실도 (Entropy): {entropy:.4f}")

    # 채점 조건
    CORRECT_ANSWER = 6
    if conf >= 0.85 and entropy <= 0.6:
        if pred == CORRECT_ANSWER:
            st.success("✅ 정답입니다! 잘 했어요.")
            status = "정답"
            correct = True
        else:
            st.error("❌ 오답입니다. 다시 풀어보세요.")
            status = "오답"
            correct = False
    else:
        st.warning("⚠️ 예측이 불확실하거나 신뢰도가 낮습니다. 답을 다시 써서 업로드 해주세요.")
        status = "불확실"
        correct = None

    save_record(user_id, pred, conf, entropy, correct, status)

# ------------------ 기록 조회 ------------------
st.markdown("---")
st.header("실시간 학습자 예측 기록 조회")
query_id = st.text_input("기록을 조회할 학습자 ID 입력")

if query_id:
    history = load_user_history(query_id)
    if history:
        for rec in reversed(history[-10:]):  # 최근 10개만 표시
            st.write(f"[{rec['timestamp']}] 예측: {rec['predicted']} / 신뢰도: {rec['confidence']} / 불확실도: {rec['entropy']} → {rec['status']}")
    else:
        st.info("해당 학습자의 기록이 없습니다.")
