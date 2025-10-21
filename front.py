import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Initialize the OpenAI Chat Model
load_dotenv()
llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini", streaming=True)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="🦜🔗 LangChain Food Reviews Lima!")
st.title("Descubre restaurantes buenazos en Lima, Peru 🍲🇵🇪")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

 # Initialize conversation memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Create the Conversation Chain
conversation = ConversationChain(llm=llm, memory=st.session_state.memory, verbose=True)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola! Conozco restaurantes buenazos en Lima, Peru. Preguntame algo sobre ello!"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Invoke the LangChain conversation chain
            response = conversation.predict(input=prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
