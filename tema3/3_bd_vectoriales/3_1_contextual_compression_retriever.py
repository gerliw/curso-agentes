from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

vectorstore = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="tema3/3_bd_vectoriales/chroma_db"
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)


consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = compression_retriever.invoke(consulta)

print("Top documentos mas similares a la consulta (comprimidos):\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")