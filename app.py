import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

st.set_page_config(
    page_title="Smart FAQ Assistant",
    page_icon="🤖",
    layout="centered"
)
# Load FAQ data
with open("faq_data.json", "r") as file:
    faqs = json.load(file)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# Convert questions into vectors
vectorizer = TfidfVectorizer(stop_words="english")
question_vectors = vectorizer.fit_transform(questions)

def get_answer(user_question):
    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_score = similarity.max()
    best_match = similarity.argmax()

    if best_score < 0.8:
        return "❌ Sorry, this question is not available in the FAQ database."

    return answers[best_match]

# Streamlit UI
st.title("🤖 Smart FAQ Assistant")

st.markdown("""
Welcome! Ask me questions about:

✅ Artificial Intelligence  
✅ Python Programming  
✅ Machine Learning  
✅ GitHub  
✅ Natural Language Processing (NLP)
""")


if "messages" not in st.session_state:
    st.session_state.messages = []
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

user_input = st.text_input("Ask a question:")

if user_input:
    answer = get_answer(user_input)    

    st.session_state.messages.append(
        {
            "question": user_input,
            "answer": answer,
            "time": datetime.now().strftime("%H:%M:%S")
        }
    )

for msg in reversed(st.session_state.messages):
    with st.chat_message("user"):
        st.write(msg["question"])

    with st.chat_message("assistant"):
        st.write(msg["answer"])

st.caption("Developed by Bindu Sri Pinnelli | CodeAlpha AI Internship Task 2")

st.sidebar.title("Categories")

category = st.sidebar.selectbox(
    "Choose Category",
    ["All", "AI", "Python", "Programming"]
)

st.sidebar.markdown("---")
st.sidebar.write(
    f"Total Questions Asked: {len(st.session_state.messages)}"
)