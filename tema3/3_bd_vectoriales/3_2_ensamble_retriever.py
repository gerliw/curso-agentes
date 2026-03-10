from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import MultiQueryRetriever

# 1. Carga de documentos para el retriever BM25
# BM25 es un algoritmo de búsqueda de texto que funciona con palabras clave.
# Necesita los documentos en texto plano para construir su índice.
loader = PyPDFDirectoryLoader("tema3/3_bd_vectoriales/contratos")
documentos = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=1000
)

docs_split = text_splitter.split_documents(documentos)

# 2. Creación del retriever BM25
# Este retriever se encargará de la búsqueda por palabras clave.
bm25_retriever = BM25Retriever.from_documents(docs_split)
bm25_retriever.k = 2

# 3. Creación del retriever de ChromaDB con MultiQuery
# Este retriever utiliza embeddings para realizar una búsqueda semántica (por significado).
# MultiQueryRetriever mejora la búsqueda generando varias consultas desde diferentes perspectivas.
vectorstore = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="tema3/3_bd_vectoriales/chroma_db"
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retriever_chroma = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

# 4. Creación del Ensemble Retriever
# EnsembleRetriever combina los resultados de múltiples recuperadores.
# En este caso, combina la búsqueda por palabras clave (BM25) y la búsqueda semántica (ChromaDB).
# El parámetro `weights` determina la importancia que se le da a cada recuperador en los resultados finales.
# Un peso de 0.5 para cada uno significa que ambos son igualmente importantes.
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever_chroma],
    weights=[0.5, 0.5]
)

# 5. Ejecución de la consulta
# Al ejecutar la consulta con el ensemble_retriever, se obtienen resultados que son relevantes
# tanto a nivel de palabras clave como de significado, proporcionando una respuesta más completa.
consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = ensemble_retriever.invoke(consulta)

print("Top documentos mas similares a la consulta (combinando BM25 y búsqueda semántica):\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
