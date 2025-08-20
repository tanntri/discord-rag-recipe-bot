# from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class LLMModel:
    def __init__(self, model_name: str = "llama3.2"):
        if not model_name:
            # model_name = "llama3.2"
            model_name = "gpt-4o"
        # self.model = ChatOllama(model=model_name, temperature=0.0)
        self.model = ChatOpenAI(model=model_name, temperature=0.0)

    def get_model(self):
        return self.model
    
class EmbeddingModel:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        if not model_name:
            # model_name = "mxbai-embed-large"
            model_name = "text-embedding-3-small"
        # self.embedding_model = OllamaEmbeddings(model=model_name)
        self.embedding_model = OpenAIEmbeddings(model=model_name)

    def get_embedding_model(self):
        return self.embedding_model
    
if __name__ == "__main__":
    llm_instance = LLMModel()  
    llm_model = llm_instance.get_model()
    response=llm_model.invoke("does mapo tofu have tofu?")

    print(response)