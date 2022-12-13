import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

@st.cache
def load_model():
    return AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")

tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
model = load_model()
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 웹 페이지 제목
st.title("호재/악재 판독기 🤓")
# 웹 페이지 부제목
st.subheader('뉴스 기사 제목을 올려주세요!')

# 웹 페이지에 입력
with st.form(key="form"):
    sentence = st.text_input(label="기사 제목", placeholder="삼성전자, 2년 만에 인도 스마트폰 시장 점유율 1위 '왕좌 탈환'")
    submit = st.form_submit_button("Go!")

# 버튼을 눌렀을 때, 작동되는 코드
if submit:
    st.write("이 종목은... 🤓")

    # 모델의 inference가 끝날 때까지 기다림
    with st.spinner("오이 종목은... 🤓"):
        # classifier 라는 이름의 딥러닝 모델사용.
        # 원하는 모델로 변경할 수 있다.
        results = classifier(sentence)[0]

    label = results["label"]
    score = results["score"]
    
    # 모델로부터 얻은 결과를 원하는 방식, 폰트, 배치대로 화면에 보여줌
    # 나만의 디자인을 원한다면 아래 링크 참고
    # https://docs.streamlit.io/library/api-reference/layout
    col1, col2 = st.columns(2)
    possibility = score * 100

    if label == "positive": col1.metric("호재입니다! 추매각?", "😁", f"{possibility} %")
    elif label == "negative": col1.metric("악재입니다ㅠㅠ 손절각?", "😭", f"{possibility} %")
    else: col1.metric("이건...저도 잘 모르겠어요... 다른 기사 있나요?", "😅")
