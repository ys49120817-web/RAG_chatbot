
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# --------------------------------------------------
# 1. 데이터 폴더
# --------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    CURRENT_DIR.parent
    / "6w"
    / "data"
)

# --------------------------------------------------
# 2. PDF 불러오기
# --------------------------------------------------

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
# 3. Chunking
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = splitter.split_documents(
    documents
)

# --------------------------------------------------
# 4. Embedding
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name=(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
)

# --------------------------------------------------
# 5. Chroma
# --------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="week7_step3"
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)

# --------------------------------------------------
# 6. Ollama
# --------------------------------------------------

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

# --------------------------------------------------
# 7. Basic RAG 함수
# --------------------------------------------------

def ask_rag(question):

    # 관련 문서 검색
    docs = retriever.invoke(
        question
    )
    print("\n========== 검색된 실제 내용 ==========")

    for i, doc in enumerate(docs, start=1):
        print(f"\n[검색 결과 {i}]")
        print("파일:", doc.metadata.get("file_name"))
        print("페이지:", doc.metadata.get("page"))
        print(doc.page_content[:1000])

    print("======================================")

    # 검색된 문서를 하나의 문자열로 결합
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
당신은 제공된 공지 문서를 근거로 답하는 AI입니다.

다음 규칙을 지키세요.

1. 검색된 문서에 있는 내용만 사용하세요.
2. 문서에 없는 내용은 추측하지 마세요.
3. 확인할 수 없는 경우
   "제공된 문서에서 확인할 수 없습니다."
   라고 답하세요.

[검색된 문서]

{context}

[질문]

{question}
"""

    response = llm.invoke(
        prompt
    )

    return response.content, docs

# --------------------------------------------------
# 8. 질문
# --------------------------------------------------

question = "인턴십 신청 기간은 언제인가요?"

answer, docs = ask_rag(
    question
)

# --------------------------------------------------
# 9. 결과
# --------------------------------------------------

print("질문:")
print(question)

print("\n검색된 문서:")

for doc in docs:

    print(
        "-",
        doc.metadata.get(
            "file_name",
            "확인 불가"
        )
    )

print("\nRAG 답변:")
print(answer)
