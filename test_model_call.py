from langchain_ollama import ChatOllama, OllamaLLM

llm = ChatOllama(base_url="http://10.5.16.143:11434", model="MedAIBase/Qwen3-VL-Embedding:2b")

print(llm.invoke("Hello, how are you?"))

# qwen3-vl:8b-instruct
# qwen3.5:35b
