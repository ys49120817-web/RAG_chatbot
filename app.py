# ==========================================================
# app.py
# 7주차 RAG Chatbot - Streamlit
# ==========================================================

# ==========================================================
# 1. 라이브러리 불러오기
# ==========================================================

from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# ==========================================================
# 2. Streamlit 기본 설정
# ==========================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬"
)

st.title(
    "한성대학교 학사공지 RAG Chatbot"
)

# ==========================================================
# 3. RAG 환경 생성
# ==========================================================

@st.cache_resource
def build_rag():

    # ------------------------------------------------------
    # 현재 app.py 위치
    # C:\2026-open\7w
    # ------------------------------------------------------

    current_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    # ------------------------------------------------------
    # 데이터 폴더
    # C:\2026-open\6w\data
    # ------------------------------------------------------

    data_dir = (
        current_dir.parent
        / "6w"
        / "data"
    )

    # ------------------------------------------------------
    # 공지 PDF 검색
    # ------------------------------------------------------

    pdf_files = sorted(
        data_dir.glob(
            "공지*.pdf"
        )
    )

    if not pdf_files:

        raise FileNotFoundError(
            "6w/data 폴더에서 "
            "공지 PDF를 찾을 수 없습니다."
        )

    # ------------------------------------------------------
    # PDF 문서 불러오기
    # ------------------------------------------------------

    documents = []

    for pdf_file in pdf_files:

        loader = PyPDFLoader( str(pdf_file) )

        pages = loader.load()

        # 파일명 Metadata 추가
        for page in pages:

            page.metadata["file_name"] = pdf_file.name

        documents.extend(pages)

    # ------------------------------------------------------
    # Chunk 생성
    # ------------------------------------------------------

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )
    )

    chunks = (
        splitter
        .split_documents(documents)
    )

    # ------------------------------------------------------
    # Embedding 모델
    # ------------------------------------------------------

    embedding_model = (
        HuggingFaceEmbeddings(
            model_name=(
                "sentence-transformers/"
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
        )
    )

    # ------------------------------------------------------
    # Chroma Vector DB
    # ------------------------------------------------------

    vectorstore = (
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=(
                "week7_streamlit"
            )
        )
    )

    # ------------------------------------------------------
    # Retriever
    # 관련 Chunk 3개 검색
    # ------------------------------------------------------

    retriever = (
        vectorstore.as_retriever(
            search_kwargs={
                "k": 3
            }
        )
    )

    # ------------------------------------------------------
    # Ollama LLM
    # ------------------------------------------------------

    llm = ChatOllama(
        model="gemma3:4b",
        temperature=0,
        num_predict=120
    )

    return retriever, llm

# ==========================================================
# 4. RAG 생성
# ==========================================================

retriever, llm = build_rag()

# ==========================================================
# 5. 대화형 RAG 함수
# ==========================================================

def ask_chat_rag(
    question,
    chat_history
):

    # ------------------------------------------------------
    # 이전 사용자 질문 찾기
    # ------------------------------------------------------

    previous_question = None

    for message in reversed(chat_history):

        if (message["role"] == "user"):

            previous_question = (message["content"])

            break

    # ------------------------------------------------------
    # 검색 Query 생성
    # ------------------------------------------------------

    if previous_question:

        search_query = f"""
이전 질문:
{previous_question}

현재 질문:
{question}
"""

    else:

        search_query = question

    # ------------------------------------------------------
    # 관련 문서 검색
    # ------------------------------------------------------

    docs = retriever.invoke(search_query )

    # ------------------------------------------------------
    # 검색된 문서 하나의 문자열로 결합
    # ------------------------------------------------------

    context_parts = []

    for i, doc in enumerate(
        docs,
        start=1
    ):

        file_name = (
            doc.metadata.get(
                "file_name",
                "확인 불가"
            )
        )

        context_parts.append(
            f"""
[문서 {i}]

파일명:
{file_name}

내용:
{doc.page_content}
"""
        )

    context = "\n\n".join( context_parts  )

    # ------------------------------------------------------
    # 이전 대화 내용 생성
    # ------------------------------------------------------

    history_parts = []

    for message in chat_history:

        role = (
            "사용자"
            if message["role"] == "user"
            else "AI"
        )

        history_parts.append(
            f"{role}: "
            f"{message['content']}"
        )

    history_text = "\n".join(history_parts  )

    # ------------------------------------------------------
    # Prompt 생성
    # ------------------------------------------------------

    prompt = f"""
당신은 한성대학교 학사공지 문서를 근거로
질문에 답하는 AI입니다.

다음 규칙을 지키세요.

1. 이전 대화는 현재 질문의 의미를 이해하는 데 사용하세요.

2. 실제 답변은 검색된 문서의 내용을 근거로 작성하세요.

3. 문서에 없는 내용은 추측하지 마세요.

4. 질문과 문서의 표현이 조금 다르더라도
   의미가 같으면 관련 정보를 사용하세요.

5. 날짜, 기간, 제출 방법 등 구체적인 정보가 있으면
   문서의 내용을 정확하게 사용하세요.

6. 검색된 문서에서 확인할 수 없는 내용은
   "제공된 문서에서 확인할 수 없습니다."
   라고 답하세요.

7. 답변은 한국어로 간결하게 작성하세요.

[이전 대화]

{history_text}

[검색된 문서]

{context}

[현재 질문]

{question}

[답변]
"""

    # ------------------------------------------------------
    # Ollama 답변 생성
    # ------------------------------------------------------

    response = llm.invoke(prompt  )

    return (response.content,  docs,  search_query  )

# ==========================================================
# 6. Session State 초기화
# ==========================================================

if ( "messages"  not in st.session_state ):
    st.session_state.messages = []

if ( "pending_question" not in st.session_state ):
    st.session_state.pending_question = None

# ==========================================================
# 7. 대화 초기화
# ==========================================================

if st.button( "대화 초기화"):
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.rerun()

# ==========================================================
# 8. 기존 대화 출력
# ==========================================================

for message in (st.session_state.messages ):

    with st.chat_message( message["role"] ):

        st.write(message["content"]  )

        # --------------------------------------------------
        # AI 답변에 검색 정보가 저장되어 있으면 표시
        # --------------------------------------------------

        if ( message["role"] == "assistant" and "sources" in message ):

            with st.expander( "검색된 문서 확인" ):

                st.write( "검색 Query")

                st.code( message[ "search_query" ] )

                for i, source in enumerate(message["sources"],  start=1 ):

                    st.write( f"검색 결과 {i}" )

                    st.write( "파일:", source[ "file_name" ]  )

                    st.write(source["content" ] )

                    st.divider()

# ==========================================================
# 9. RAG 실행
# ==========================================================

if ( st.session_state.pending_question is not None):

    question = (st.session_state.pending_question)

    # ------------------------------------------------------
    # 현재 질문을 제외한 이전 대화
    # ------------------------------------------------------

    previous_history = ( st.session_state.messages[:-1] )

    # ------------------------------------------------------
    # 답변 생성 중 표시
    # 이 상태에서는 chat_input이 생성되지 않음
    # ------------------------------------------------------

    with st.chat_message("assistant" ):

        with st.spinner( "답변을 생성하고 있습니다..." ):

            answer, docs, search_query = (
                ask_chat_rag(  question, previous_history)
            )

    # ------------------------------------------------------
    # 검색 결과 저장용 데이터 생성
    # ------------------------------------------------------

    sources = []

    for doc in docs:

        sources.append({
            "file_name": doc.metadata.get( "file_name", "확인 불가"),
            "content":  doc.page_content[:800]
        })

    # ------------------------------------------------------
    # AI 답변 저장
    # ------------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "search_query": search_query,
        "sources": sources
    })

    # ------------------------------------------------------
    # 처리 완료
    # ------------------------------------------------------

    st.session_state.pending_question = ( None)

    # ------------------------------------------------------
    # 화면 갱신
    # ------------------------------------------------------

    st.rerun()

# ==========================================================
# 10. 새로운 질문 입력
# ==========================================================

else:
    question = st.chat_input( "질문을 입력하세요." )

    if question:

        # --------------------------------------------------
        # 사용자 질문 저장
        # --------------------------------------------------
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # --------------------------------------------------
        # 처리할 질문 저장
        # --------------------------------------------------

        st.session_state.pending_question = (
            question
        )

        # --------------------------------------------------
        # 다시 실행
        #
        # 다음 실행에서는
        # pending_question이 존재하기 때문에
        # 질문 입력창이 표시되지 않고
        # RAG 답변 생성으로 이동
        # --------------------------------------------------

        st.rerun()
