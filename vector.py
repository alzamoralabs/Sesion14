from langchain_ollama import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import requests

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Función para cargar las reviews y crear el retriever
# Usando OllamaEmbeddings o OpenAIEmbeddings segun el modo elegido
def load_reviews(aimode: str):
    
    embeddings = None
    db_location = ""

    df = pd.read_csv("data/data.csv")

    if aimode != "1":
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        db_location = "./chroma_ollama_rest_reviews"
    else:
        embeddings = OpenAIEmbeddings()
        db_location = "./chroma_openai_rest_reviews"
    
    # Load data
    add_documents = not os.path.exists(db_location)

    if add_documents:
        documents = []
        ids = []

        for i, row in df.iterrows():
            document = Document(
                page_content = "Nombre Restaurante: " + row["Nombre"] + "\n" + "Autor Review:" + row["Autor"] + "\n" +
                "Calificacion:" + row["Calificacion"] + "\n" + "Resumen de Review:" + row["Review"] + "\n" +
                "Plato estrella:"+ row["Plato_Estrella"] + "\n" + "Distrito:" + row["Distrito"] + "\n" +
                "Tipo de comida:" + row["Tipo_comida"] + "\n",
                metadata={"Clasificacion": row["Tipo_comida"]},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)

    vector_store = Chroma(
        collection_name= "restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    if add_documents:
        vector_store.add_documents(documents=documents, ids=ids)

    retriever = vector_store.as_retriever(
        search_kwargs={"k":5} # numero de documentos a recuperar
    )
    return retriever

##################################################################################################################
# Herramienta para obtener los datos de chroma
# Usando librerias clasicas de langchain
from langchain.tools import Tool

def get_rag(query: str) -> str:
    # Esta función recupera información relevante de la base de datos vectorial basada en la consulta del usuario.
    # Utiliza el retriever para buscar y devolver los documentos más relevantes.
    retriever = load_reviews(aimode="2") # aqui puedes cambiar a "1" si usas openai embeddings
    results = retriever.get_relevant_documents(query)
    combined_text = "\n\n".join([doc.page_content for doc in results])
    return combined_text

get_rag = Tool(
    name="retrieval_tool",
    func=get_rag,
    description="""Usa esta herramienta para responder preguntas sobre restaurantes buenazos en Lima, Peru.
    La entrada es una pregunta y la salida es un texto con la información relevante extraída de las reviews de los influencers gastronómicos más reconocidos de Lima.""")

def mesa247list(foodtype: str) -> str:
    ## funcion que confirma disponibilidad de restaurante en mesa 247
    # esta te lista restaurantes de miraflores https://restaurantes.mesa247.pe/v3/public/search?language=en&country_code=pe&page=1&sort=relevance&ubication_ids=7&paginate=25&department_code=lim
    results = ""
    try:
        #full url
        url = f"https://restaurantes.mesa247.pe/v3/public/search?language=en&country_code=pe&page=1&sort=relevance&ubication_ids=7&paginate=25&department_code=lim"
        loader = WebBaseLoader(url)
        documents = loader.load()

        # 2. Dividir el texto en fragmentos manejables
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # 3. Crear embeddings usando un modelo local de Ollama (por ejemplo, 'nomic-embed-text')
        embeddings = OpenAIEmbeddings()

        # 4. Persistir los embeddings en una base de datos Chroma local
        persist_directory = "./chroma_db_mesa247"
        m247vectorstore = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_directory
        )

        # 5. Ejemplo de búsqueda semántica
        query = "Disponibilidad de restaurantes especializados en " + foodtype + " entrega la informacion en json"
        results = m247vectorstore.similarity_search(query, k=3)
        for i, res in enumerate(results, 1):
            print(f"Resultado {i}:\n{res.page_content}\n")
        combined_text = "\n\n".join([doc.page_content for doc in results])
        return combined_text
    except Exception as e:
        return f"Error al obtener datos de Mesa 247: {e}"

mesa247list = Tool(
    name="mesa247_list_tool",
    func=mesa247list,
    description="""Usa esta herramienta para listar restaurantes en Miraflores, Lima a través de Mesa 247.
    La entrada es un tipo de comida (por ejemplo, 'ceviche o marina', 'criolla').
    La salida es un texto con la información de restaurantes indicando sus caracteristicas y tipo de comida especificados.""")