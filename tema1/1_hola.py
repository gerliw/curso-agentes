from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7 )

pregunta = "¿en que año se conquistó roma a israel?"

print( "Pregunta: ", pregunta)

respuesta = llm.invoke(pregunta)

print( "Respuesta: ", respuesta.content)