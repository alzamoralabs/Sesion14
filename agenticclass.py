### agenticclass.py ### UTEC - Sesion 13 - AgenticRAG
# por Boris Alzamora - 2025

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_openai_tools_agent, AgentExecutor
from vector import get_rag, mesa247list

load_dotenv()

class ResearcherResponse(BaseModel):
    descripcion_plato: str
    recomendaciones: str
    referencias: list[str]
    tools_usadas: list[str]

class RestaurantAgent:
    def __init__(self, model="gpt-4o-mini", temperature=0.2):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.memory = ConversationBufferMemory()
        self.parser = PydanticOutputParser(pydantic_object=ResearcherResponse)
        self.tools = [get_rag, mesa247list]
        self.prompt = self._create_prompt()
        self.agent = create_openai_tools_agent(llm=self.llm, prompt=self.prompt, tools=self.tools)
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools, verbose=True)

    def _create_prompt(self):
        sysprompt = """
        Eres un experto en responder preguntas sobre restaurantes buenazos de Lima, Peru.
        y recolectas informacion de las reviews de los influencers gastronómicos mas reconocidos de Lima.
        Tienes acceso a la siguiente informacion de restaurantes:
        Nombre Restaurante, Autor Review, Calificacion (del 0 al 10, donde 10 es la mejor), Resumen de Review,
        Plato estrella, Distrito y Tipo de comida.
        Proporcionas respuestas en este formato y no provees otro texto:\n{format_instructions}
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(sysprompt),
            ("placeholder", "{chat_history}"),
            HumanMessagePromptTemplate.from_template("busqueda del usuario: {query}."),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(format_instructions=self.parser.get_format_instructions())
        return prompt

    def invoke_agent(self, user_query: str):
        raw_response = self.agent_executor.invoke({"query": user_query})
        print("RAW RESPONSE:", raw_response)

        try:
            structured_response = self.parser.parse(raw_response["output"])
            print("STRUCTURED RESPONSE:", structured_response)
            return structured_response
        except Exception as e:
            print("Error al parsear la respuesta:", e, "RAW:", raw_response)
            return None
