
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# --------------------------------------------------
# 1. 문서 준비
# --------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    CURRENT_DIR.parent
    / "6w"
    / "data"
)

pdf_files = sorted(
    DATA_DIR.glob("공지*.pdf")
)

documents = []

for pdf_file in pdf_files:

    loader = PyPDFLoader(
        str(pdf_file)
    )

    pages = loader.load()

    for page in pages:
        page.metadata["file_name"] = pdf_file.name

    documents.extend(pages)

# --------------------------------------------------
# 2. Chunking
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = splitter.split_documents(
    documents
)

# --------------------------------------------------
# 3. Embedding + Chroma
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name=(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="week7_step4"
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)

# --------------------------------------------------
# 4. Ollama
# --------------------------------------------------

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0,
    num_predict=100
)

# --------------------------------------------------
# 5. 대화형 RAG 함수
# --------------------------------------------------

def ask_chat_rag(
    question,
    chat_history
):

    # 직전 사용자 질문 찾기
    previous_question = ""

    for message in reversed(
        chat_history
    ):

        if message["role"] == "user":

            previous_question = (
                message["content"]
            )

            break

    # Retrieval용 질문
    if previous_question:

        search_query = f"""
이전 질문:
{previous_question}

현재 질문:
{question}
"""

    else:

        search_query = question

    # 문서 검색
    docs = retriever.invoke(
        search_query
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # 이전 대화 만들기
    history_text = ""

    for message in chat_history:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    # Prompt
    prompt = f"""
당신은 공지 문서를 기반으로 답하는
대화형 RAG Chatbot입니다.

[규칙]

1. 이전 대화는 현재 질문의 의미를 이해할 때 사용하세요.
2. 답변 내용은 검색된 문서를 근거로 작성하세요.
3. 문서에 없는 내용은 추측하지 마세요.
4. 확인할 수 없다면
   "제공된 문서에서 확인할 수 없습니다."
   라고 답하세요.

[이전 대화]

{history_text}

[검색된 문서]

{context}

[현재 질문]

{question}
"""

    response = llm.invoke(
        prompt
    )

    return (
        response.content,
        docs,
        search_query
    )

# --------------------------------------------------
# 6. 대화 시작
# --------------------------------------------------

chat_history = []

print(
    "대화형 RAG Chatbot"
)

print(
    "종료하려면 exit를 입력하세요."
)

while True:

    question = input(
        "\n사용자: "
    )

    if question.lower() == "exit":
        break

    answer, docs, search_query = (
        ask_chat_rag(
            question,
            chat_history
        )
    )

    print(
        "\n[검색 Query]"
    )

    print(
        search_query
    )

    print(
        "\nAI:",
        answer
    )

    # 대화 저장
    chat_history.append({
        "role": "user",
        "content": question
    })

    chat_history.append({
        "role": "assistant",
        "content": answer
    })
