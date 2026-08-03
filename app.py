import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# تحميل متغيرات البيئة
load_dotenv()

# الإعدادات العامة والأيقونات
SYSTEM_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"
USER_AVATAR = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80"

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

@st.cache_resource
def get_rag_chain():
    vectorstore = get_vectorstore()
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0.1, max_retries=2)
    system_prompt = (
        "أنت مساعد ذكي مخصص للإجابة عن استفسارات مديرية تربية نينوى وجامعة الموصل.\n"
        "استخدم المعلومات الواردة في السياق التالي فقط للإجابة على سؤال المستخدم.\n"
        "إذا لم تكن الإجابة موجودة في السياق، قل بوضوح أنك لا تملك المعلومة الرسمية حول ذلك.\n"
        "أجب باللغة العربية وبشكل دقيق ومختصر.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

def process_rag_response(user_query):
    try:
        rag_chain = get_rag_chain()
        
        with st.chat_message("assistant", avatar=SYSTEM_AVATAR):
            with st.spinner("جاري البحث ..."):
                response = rag_chain.invoke({'input': user_query})
                result = response["answer"]
                st.markdown(result)
        
        st.session_state.messages.append({'role': 'assistant', 'content': result})

    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الطلب: {str(e)}")

def main():
    st.set_page_config(
        page_title="المجيب الآلي - تربية نينوى وجامعة الموصل",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
            html, body, [class*="css"] {
                font-family: 'Cairo', sans-serif;
                direction: ltr !important;
                text-align: left !important;
            }
            [data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="collapsedControl"] { display: none !important; }
            footer {visibility: hidden;}
            header [data-testid="stAppDeployButton"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

    # الهيدر العلوي بأيقونة الروبوت 🤖
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; direction: ltr; margin-bottom: 15px;">
            <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; width: 45px; height: 45px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px;">🤖</div>
            <h1 style="margin: 0; color: #3b82f6; font-weight: 700; font-size: 22px;">المجيب الآلي (تربية نينوى & جامعة الموصل)</h1>
        </div>
    """, unsafe_allow_html=True)

    # عرض سجل المحادثة
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message['role'] == 'user' else SYSTEM_AVATAR
        st.chat_message(message['role'], avatar=avatar).markdown(message['content'])

    prompt = st.chat_input("اكتب سؤالك هنا...")
    if prompt:
        st.chat_message('user', avatar=USER_AVATAR).markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        process_rag_response(prompt)

if __name__ == "__main__":
    main()
