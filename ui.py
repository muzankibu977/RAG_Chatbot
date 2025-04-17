import streamlit as st
import time
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Chroma + Embedding Setup
db_dir = "./chroma_db"
embedding = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
try:
    vector_db = Chroma(collection_name="local-rag", persist_directory=db_dir, embedding_function=embedding)
    print("Vector DB loaded successfully!")
except Exception as e:
    st.error(f"Failed to load vector database: {e}")
    raise

# LLM Setup
local_model = "llama3.2"
llm = ChatOllama(model=local_model, streaming=True)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
        You are an AI assistant for a bank. Generate 4 different rephrasings of the user's question
        to help retrieve relevant documents from a vector database.
        Original question: {question}
        Provide 4 alternate versions, each on a new line:
    """
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_db.as_retriever(),
    llm=llm,
    prompt=QUERY_PROMPT,
)

# RAG Prompt Template
template = """
    You are a helpful and intelligent AI customer service assistant for a bank.
    Use the following context if relevant to answer the user’s question. If the context does not contain the answer,
    use your general knowledge of banking to provide the most helpful possible answer.
    If the question cannot reasonably be answered, politely suggest contacting the bank’s helpline at 16234.
    Context:
    {context}
    Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG Chain (Streaming)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Bot Response Cleaner
def clean_bot_response(text):
    patterns_to_remove = [
        r"\baccording to (the )?(text|document|context|information)\b[:,]?",
        # ... (other patterns remain the same)
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    sentence_end = re.compile(r'(?<=[.!?])(\s+|\n+)')
    sentences = sentence_end.split(text.strip())
    final = ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        separator = sentences[i+1] if i+1 < len(sentences) else ""
        final += sentence[:1].upper() + sentence[1:] + separator
    return final.strip()

# Stream Bot Reply
def get_bot_response(user_message):
    try:
        responses = chain.invoke({"question": user_message})  # Pass input as a dictionary
        time.sleep(1)  # Simulates a delay in bot response
        return responses
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI Setup
st.set_page_config(page_title="Bank Service Chatbot", page_icon="🏦", layout="wide")

custom_css = """
<style>
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #121212; /* Dark background */
        color: white;
        margin: 0;
        padding: 0;
    }
    .chat-container {
        max-width: 700px;
        margin: auto;
        padding: 20px;
        background-color: transparent; /* No background for chat container */
        border-radius: 15px;
        box-shadow: none; /* No shadow */
    }
    .user-message {
        text-align: right;
        margin-bottom: 10px;
    }
    .user-message div {
        display: inline-block;
        background-color: #dcf8c6;
        color: black;
        padding: 10px 15px;
        border-radius: 20px;
        max-width: 70%;
        word-wrap: break-word;
        font-size: 14px;
    }
    .bot-message {
        text-align: left;
        margin-bottom: 10px;
    }
    .bot-message div {
        display: inline-block;
        background-color: #e9ecef;
        color: black;
        padding: 10px 15px;
        border-radius: 20px;
        max-width: 70%;
        word-wrap: break-word;
        font-size: 14px;
    }
    .timestamp {
        font-size: 12px;
        color: #6c757d;
        margin-top: 5px;
        text-align: left;
    }
    .input-container {
        display: flex;
        gap: 10px;
        margin-top: 20px;
    }
    .input-container input {
        flex: 1;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
        background-color: #333; /* Dark input box */
        color: white;
    }
    .sidebar {
        background-color: #e9f7fd; /* Light blue sidebar background */
        padding: 20px;
        height: 100vh;
        position: fixed;
        top: 0;
        left: 0;
        width: 300px;
        overflow-y: auto;
    }
    .sidebar h3 {
        color: #007bff;
        margin-bottom: 20px;
    }
    .sidebar ul {
        list-style-type: none;
        padding: 0;
    }
    .sidebar li {
        margin-bottom: 10px;
    }
    .sidebar button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar Instructions
with st.sidebar:
    st.markdown('<div class="sidebar">', unsafe_allow_html=True)
    st.markdown("### Instructions")
    st.write("""
    1. Enter your question in the text box.
    2. Click **Send** or press **Enter**.
    3. View chatbot responses in a chat format.
    4. Response time is displayed for reference.
    """)
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.last_input = ""
        st.session_state.clear_input_flag = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "clear_input_flag" not in st.session_state:
    st.session_state.clear_input_flag = False

# Clear input box before it's rendered
if st.session_state.clear_input_flag:
    st.session_state["user_input"] = ""
    st.session_state.clear_input_flag = False

# Main Content Area
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.title("🏛️ Bank Service Chatbot 🤖")

# Render chat history
chat_placeholder = st.empty()

def render_chat_history():
    chat_content = ""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            chat_content += f'<div class="user-message"><div>{message["text"]}</div></div>'
        else:
            chat_content += f'<div class="bot-message"><div>{message["text"]}</div></div><div class="timestamp">Response time: {message["response_time"]} seconds</div>'
    chat_placeholder.markdown(chat_content, unsafe_allow_html=True)

render_chat_history()

# Text input box
st.markdown('<div class="input-container">', unsafe_allow_html=True)
current_input = st.text_input("", placeholder="Type your message here...", key="user_input", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# Handle messages
def send_message(input_text):
    if input_text.strip():
        st.session_state.chat_history.append({"role": "user", "text": input_text})
        start_time = time.time()
        bot_response = clean_bot_response(get_bot_response(input_text))
        response_time = round(time.time() - start_time, 2)
        st.session_state.chat_history.append({
            "role": "bot",
            "text": bot_response,
            "response_time": response_time
        })
        render_chat_history()

# Trigger on Enter
if current_input and st.session_state.get("last_input") != current_input:
    send_message(current_input)
    st.session_state.last_input = current_input
    st.session_state.clear_input_flag = True
    st.rerun()