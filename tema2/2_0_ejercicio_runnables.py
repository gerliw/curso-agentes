from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json
import re
 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0 )

def preprocess_text(text):
    """Limpia el texto eliminando espacios extras y limitando longitud"""
    # Pista: usa .strip() para eliminar espacios
    # Pista: limita a 500 caracteres con slicing [:500]
    return text.strip()[:500]
 
# Convertir la función en un Runnable
preprocessor = RunnableLambda(preprocess_text)


# promptSummary = PromptTemplate(
#     input_variables=["text"],
#     template="Actúa como un experto en síntesis de información. Lee el siguiente texto y resúmelo en una sola oración que sea clara, concisa y que capture la idea central junto con su conclusión principal. Evita introducciones como ""El texto dice que..."" o ""Este artículo trata sobre..."". Ve directamente al grano. texto: {text}")




def generate_summary(text):
    """Genera un resumen conciso del texto"""
    prompt = f"Resume en una sola oración: {text}"
    response = llm.invoke(prompt)
    return response.content





def analyze_sentiment(text):
    """Analiza el sentimiento y devuelve resultado estructurado"""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde ÚNICAMENTE en formato JSON válido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificación breve"}}
    
    Texto: {text}"""
    # print ("prompt: ", prompt)
    response = llm.invoke(prompt)
    try:
        # print("respuesta:",response.content)
        # Limpieza rápida de la respuesta del modelo
        raw_response = response.content.strip().replace("```json", "").replace("```", "")
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en análisis"}




def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado"""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"]
    }


def process_one(t):
    resumen = generate_summary(t)              # Llamada 1 al LLM
    sentimiento_data = analyze_sentiment(t)    # Llamada 2 al LLM
    return merge_results({
        "resumen": resumen,
        "sentimiento_data": sentimiento_data
    })
 
# Convertir en Runnable
process = RunnableLambda(process_one)


chain  = preprocessor | process


textos_prueba = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde."
]
 
for texto in textos_prueba:
    resultado = chain.invoke(texto)
    print(f"Texto: {texto}")
    print(f"Resultado: {resultado}")
    print("-" * 50)
