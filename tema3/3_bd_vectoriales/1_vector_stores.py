from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFDirectoryLoader("tema3/3_bd_vectoriales/contratos")
documentos = loader.load()

print(f"Se cargaron {len(documentos)} documentos desde el directorio.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=1000
)

docs_split = text_splitter.split_documents(documentos)

print(f"Se crearon {len(docs_split)} chunks de texto.")
vectorstore = Chroma.from_documents(
    docs_split,
    embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="./tema3/3_bd_vectoriales/chroma_db"
)

consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez "

resultados = vectorstore.similarity_search(consulta, k=10)

print("Top " + str(len(resultados)) + " documentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    # print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")