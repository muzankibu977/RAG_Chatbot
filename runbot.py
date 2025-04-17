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
        r"\bbased on (the )?(provided )?(text|context|information)\b[:,]?",
        r"\bthe (text|context|document) (says|states|mentions)\b[:,]?",
        r"\b(as|from) (mentioned|seen|stated) (in|within) (the )?(context|document)\b[:,]?",
        r"\breferring to (the )?(content|context)\b[:,]?",
        r"\bit is (stated|mentioned) (in|within) (the )?(context|document)\b[:,]?",
        r"\bthe following was found in the (documents|context)\b[:,]?",
        r"\bit (appears|seems|looks like)\b[^.]*\.",
        r"\bthis (might|may|could) not be (covered|mentioned)\b[^.]*\.",
        r"\bis not explicitly mentioned\b[^.]*\.",
        r"\bthe answer is not clearly available\b[^.]*\.",
        r"unfortunately,\s*the context provided doesn’t mention[^.]*\.",
        r"unfortunately,\s*the context does not include[^.]*\.",
        r"the context (doesn’t|does not) mention[^.]*\.",
        r"the context (doesn’t|does not) provide specific details[^.]*\.",
        r"i (couldn’t|cannot|can't) find information about[^.]*\.",
        r"\bI (couldn't|cannot|can't) (find|locate)\b[^.]*\.",
        r"\bunfortunately,\s*(the )?(information|context) does not\s*contain[^.]*\.",
        r"\bthere is no (clear|specific) mention\b[^.]*\.",
        r"\bthis is not mentioned in the (context|document)\b[^.]*\.",
        r"\bthe answer is not explicitly mentioned\b[^.]*\.",
        r"\baccording to (the )?(provided )?(context|information|text|document)( (above|below))?\b[:,.]?",
        r"\baccording to (the )?(context|information|text|document) (provided|above|below)?\b[:,.]?",
        r"\bbased on (the )?(details|content|context|information|document)( (above|below))?\b[:,.]?",
        r"\baccording to (section|clause|article) \d+(\.\d+)? (of|in) (the )?(document|context|text|information)\b[:,.]?",
        r"\baccording to (the )?(section|clause|article) (number )?\d+(\.\d+)?\b[:,.]?",
        r"\baccording to (the )?(document|text|context|information),? (section|clause|article) \d+(\.\d+)?\b[:,.]?",
        r"^however[,.\s]+",
        r"^but[,.\s]+",
        r"^still[,.\s]+",
        r"^that said[,.\s]+",
        r"^nonetheless[,.\s]+",
        r"^although[,.\s]+",
        r"^in some cases[,.\s]+",
        r"^even though[,.\s]+",
        r"^though[,.\s]+",
        r"\bthere is no (specific|clear|relevant) information (provided|available) (in|within) (the )?(document|context|text|information)\b[^.]*\.",
        r"\b(no|not enough) (details|information|context|data) (is|are)? (provided|available)\b[^.]*\.",
        r"\bthe document (does not|doesn’t) contain (any )?(specific|relevant|clear)? information\b[^.]*\.",
        r"\b(the )?document lacks (clear|specific|relevant)? information\b[^.]*\.",
        r"\bit is unclear from the (context|document|text)\b[^.]*\.",
        r"\binformation (regarding|about) this topic (is|was) not found\b[^.]*\.",
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Capitalize each sentence even after newlines
    sentence_end = re.compile(r'(?<=[.!?])(\s+|\n+)')
    sentences = sentence_end.split(text.strip())

    final = ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        separator = sentences[i+1] if i+1 < len(sentences) else ""
        final += sentence[:1].upper() + sentence[1:] + separator

    return final.strip()

# Stream Bot Reply
# def stream_bot_response(user_message):
#     try:
#         response_stream = chain.stream({"question": user_message})
#         response_text = ""
#         response_placeholder = st.empty()

#         for chunk in response_stream:
#             token = chunk.content if hasattr(chunk, "content") else str(chunk)
#             response_text += token
#             response_placeholder.markdown(f'<div class="bot-message"><div>{response_text}</div></div>', unsafe_allow_html=True)

#         return response_text

#     except Exception as e:
#         return f"An error occurred: {str(e)}"

def get_bot_response(user_message):
    try:
        responses = chain.invoke({"question": user_message})  # Pass input as a dictionary
        time.sleep(1)  # Simulates a delay in bot response
        return responses
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI Setup
st.set_page_config(page_title="Bank Service Chatbot", page_icon="🏦", layout="centered")

custom_css = """
<style>
    .chat-container {
        max-width: 600px;
        margin: auto;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .user-message {
        text-align: right;
        margin-bottom: 10px;
    }
    .user-message div {
        display: inline-block;
        background-color: #dcf8c6;
        padding: 10px 15px;
        border-radius: 20px;
        max-width: 70%;
        word-wrap: break-word;
    }
    .bot-message {
        text-align: left;
        margin-bottom: 10px;
    }
    .bot-message div {
        display: inline-block;
        background-color: #e9ecef;
        padding: 10px 15px;
        border-radius: 20px;
        max-width: 70%;
        word-wrap: break-word;
    }
    .timestamp {
        font-size: 12px;
        color: #6c757d;
        margin-top: 5px;
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
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "clear_input_flag" not in st.session_state:
    st.session_state.clear_input_flag = False

# Clear input box before it's rendered
if st.session_state.clear_input_flag:
    st.session_state["user_input"] = ""
    st.session_state.clear_input_flag = False

# UI container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.title("🏛️ Bank Service Chatbot 🤖")

# Clear chat button
if st.button("🧹 Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.last_input = ""
    st.session_state.clear_input_flag = True
    st.rerun()

# Render chat history
chat_placeholder = st.empty()

def render_chat_history():
    chat_content = ""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            chat_content += f'<div class="user-message"><div>{message["text"]}</div></div>'
        else:
            chat_content += f'<div class="bot-message"><div>{message["text"]}</div><div class="timestamp">Response time: {message["response_time"]} seconds</div></div>'
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

