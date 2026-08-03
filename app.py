import os
from dotenv import load_dotenv
import chainlit as cl

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# تحميل متغيرات البيئة
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

def load_rag_chain():
    """تحميل نموذج الـ RAG وقاعدة بيانات FAISS"""
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.1, 
        max_retries=2,
        streaming=True  # تفعيل البث المباشر للردود
    )
    
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
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={'k': 3}), 
        combine_docs_chain
    )
    return retrieval_chain


@cl.on_chat_start
async def on_chat_start():
    """تنفّذ هذه الدالة عند فتح المستخدم للمحادثة"""
    # إرسال رسالة ترحيبية
    await cl.Message(
        content="مرحباً بك! 🤖 أنا المساعد الذكي لمديرية تربية نينوى وجامعة الموصل. كيف يمكنني مساعدتك اليوم؟"
    ).send()
    
    # تحميل الـ RAG chain وحفظه في جلسة المستخدم
    rag_chain = load_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)


@cl.on_message
async def on_message(message: cl.Message):
    """تنفّذ هذه الدالة عند استقبال أي سؤال من المستخدم"""
    rag_chain = cl.user_session.get("rag_chain")
    
    # إنشاء رسالة فارغة ليتم ملؤها تدريجياً (Streaming)
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # استدعاء السلسلة مع دعم البث
        async for chunk in rag_chain.astream({"input": message.content}):
            if "answer" in chunk:
                await msg.stream_token(chunk["answer"])
                
        await msg.update()
        
    except Exception as e:
        msg.content = f"حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
