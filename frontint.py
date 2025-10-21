## WebUI con Streamlit
# por Boris Alzamora
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from agenticclass import RestaurantAgent

# --- Streamlit UI Setup ---
st.set_page_config(page_title="🦜🔗 LangChain Food Reviews Lima!")
st.title("Descubre restaurantes buenazos en Lima, Peru 🍲🇵🇪")

# Instanciamos el agente de restaurantes
localagent = RestaurantAgent()

 # Inicializamos la memoria de la conversacion
if "memory" not in st.session_state:
    st.session_state.memory = localagent.memory

# Establecemos los mensajes en el estado de la sesion
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Conozco las fijas donde comer buenazo en Lima. Preguntame algo sobre ello!"}]

# Mostrar los mensajes del chat desde el historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Adminnistramos el input del usuario para enviarlo al agente
if prompt := st.chat_input("Lanza tu consulta sobrino!..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Invocamos al agente de restraurantes definido en agenticclass.py
            response = RestaurantAgent.invoke_agent(user_query=prompt, self=localagent)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
