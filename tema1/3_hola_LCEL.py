from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7 )

plantilla = PromptTemplate(
    input_variables=["nombre"],
    template="Saluda al usuario con su nombre y un chiste gracioso por su nombre.\nNombre del usuario: {nombre}\nAsistente:"
)

chain = plantilla | llm 

resultado = chain.invoke({"nombre":"Papichulo"})
print (resultado.content)