from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class AnalisisTexto(BaseModel):
    resumen: str = Field(description="Resumen breve del texto.")
    sentimiento: str = Field(description="Sentimiento del texto (Positivo, neutro o negativo)")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6)

structured_llm = llm.with_structured_output(AnalisisTexto)

texto_prueba = "Me parecio una verga todo, no le entiendo sentido. una pelicula de accion? mas bien una pelicula de princesa."

resultado = structured_llm.invoke(f"Analiza el siguiente texto: {texto_prueba}")

print(resultado.model_dump_json())