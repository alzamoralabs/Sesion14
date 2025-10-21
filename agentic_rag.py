### main.py ### UTEC - Sesion 13 - AgenticRAG
# por Boris Alzamora - 2025

from dotenv import load_dotenv # carga variables de entorno desde un archivo .env
from pydantic import BaseModel # estructura y valida datos usando modelos

### Pydantic ### es una biblioteca de Python que se utiliza para la validación de datos y la gestión de configuraciones.
# Aprovecha las sugerencias de tipo de Python para definir modelos de datos y valida automáticamente los datos entrantes 
# con respecto a estos modelos. Esto garantiza la integridad de los datos y ayuda a detectar errores en las primeras 
# etapas del proceso de desarrollo.
###

# Establece la conexión con el modelo de lenguaje de OpenAI
from langchain_openai import ChatOpenAI 
# Plantillas de mensajes para estructurar las conversaciones
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# Estructura las respuestas del modelo usando Pydantic
from langchain_core.output_parsers import PydanticOutputParser

### Lancghain Agents ###
# Las siguientes librerias nos permiten crear agentes que pueden interactuar con herramientas externas.
from langchain.agents import create_tool_calling_agent # crea un agente que usa herramientas generico podria ser anthropic, cohere, etc.
from langchain.agents import create_openai_tools_agent # crea un agente de openai que puede usar herramientas
from langchain.agents import AgentExecutor # Implementa el ejecutor para el agente definido

from vector import get_rag, mesa247list # importamos la herramienta de busqueda en wikipedia definida en tools.py

load_dotenv()

# Define el modelo de datos para la respuesta del investigador usando Pydantic
# Aqui estamos heredando de BaseModel para crear un modelo de datos que valida y estructura la respuesta esperada
class ResearcherResponse(BaseModel):
    descripcion_plato: str
    recomendaciones: str
    referencias: list[str]
    tools_usadas: list[str]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
# establece el formato de la respuesta hacia el objeto de python, de forma que el modelo de lenguaje
# entienda que debe responder en ese formato
parser = PydanticOutputParser(pydantic_object=ResearcherResponse)

# Define la plantilla del mensaje del sistema que guiará el comportamiento del agente
sysprompt = """
Eres un experto en responder preguntas sobre restaurantes buenazos de Lima, Peru.
y recolectas informacion de las reviews de los influencers gastronómicos mas reconocidos de Lima.
Tienes acceso a la siguiente informacion de restaurantes:
Nombre Restaurante, Autor Review, Calificacion (del 0 al 10, donde 10 es la mejor), Resumen de Review,
Plato estrella, Distrito y Tipo de comida.
Proporcionas respuestas en este formato y no provees otro texto:\n{format_instructions}
"""

# Define la plantilla del mensaje del sistema
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        sysprompt
    ),
    ("placeholder", "{chat_history}"),
    HumanMessagePromptTemplate.from_template(
        "busqueda del usuario: {query}."
        # aqui estamos usando solamente una variable, pero podrian ser muchas mas del contexto del usuario
    ),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions()) # aqui enviamos el parser como string para reemplazar en la plantilla del prompt
# partial permite prellenar ciertos valores en la plantilla del mensaje, pero no absolutamente todos.
# En este caso, prellenamos 'format_instructions' con las instrucciones de formato obtenidas del parser.

tools = [get_rag, mesa247list] # lista de herramientas para usar la busqueda vectorial de reviews
agent = create_openai_tools_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Este es el ejecutor del agente, que maneja la interacción entre el usuario, el agente y las herramientas
# si queremos que el agente use herramientas, las agregamos en la lista de tools, y si queremos ver el proceso de pensamiento el verbose=True
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

raw_response = agent_executor.invoke({"query": (input("Pregunta sobre restaurantes buenazos en Lima: "))})
# la respuesta cruda del agente, que incluye toda la información generada por el modelo y las herramientas usadas
print("RAW RESPONSE:", raw_response)

try:
    # parseamos la respuesta del modelo usando el parser definido anteriormente
    structured_response = parser.parse(raw_response["output"])
    print("STRUCTURED RESPONSE:", structured_response)

except Exception as e:
    print("Error al parsear la respuesta:", e, "RAW:", raw_response)
    # si hay un error en el parseo, mostramos el error y la respuesta cruda para diagnosticar