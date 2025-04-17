import time
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Path to the stored Chroma database
db_dir = "./chroma_db"

# Load the stored Chroma vector database with the same embedding model
embedding = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
try:
    vector_db = Chroma(collection_name="local-rag", persist_directory=db_dir, embedding_function=embedding)
    print("Vector DB loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load vector database: {e}")

# Set up LLM and retrieval
local_model = "llama3.2"  # or whichever model you prefer
llm = ChatOllama(model=local_model)

# Query prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
        You are an AI assistant for a bank. Generate 4 different rephrasings of the user's question
        to help retrieve relevant documents from a vector database.

        Original question: {question}

        Provide 4 alternate versions, each on a new line:
    """
)

# Set up retriever
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt template
template = """
    You are a helpful and intelligent AI customer service assistant for a bank.
    Use the following context if relevant to answer the user’s question. If the context does not contain the answer,
    use your general knowledge of banking to provide a clear and helpful response.

    If the question cannot reasonably be answered, politely suggest contacting the bank’s helpline at 16234.

    Context:
    {context}

    Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Create chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chatbot logic
def get_bot_response(user_message):
    try:
        responses = chain.invoke({"question": user_message})  # Pass input as a dictionary
        time.sleep(1)  # Simulates a delay in bot response
        return responses
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
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