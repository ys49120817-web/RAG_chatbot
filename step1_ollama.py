from langchain_ollama import ChatOllama

# 1. Ollama에서 실행할 모델 설정
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

# 2. 모델에 질문
question = "RAG가 무엇인지 한 문장으로 설명해줘."

response = llm.invoke(question)

# 3. 결과 출력
print("질문:")
print(question)

print("\n답변:")
print(response.content)