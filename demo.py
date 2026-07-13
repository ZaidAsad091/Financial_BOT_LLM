import langchain_core
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser


import os
import bs4
import lxml
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from pathlib import Path
import gradio as gr
from openai import OpenAI




# ============================================================================
# PART 1: Data Loading, Data Ingestion & Vector Store Setup
# ============================================================================

persist_directory = "./rag_vectorestore"

print("\n Creating Embedding model\n")

embeddings= OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))  # Create embeddings model

print(f"\nEmbedding model created: {os.getenv('OPENAI_EMBEDDING_MODEL')} \n")

if os.getenv("OPENAI_API_KEY"):
    print(f"OPENAI_API_KEY is set: {os.getenv('OPENAI_API_KEY')}")
else: 
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the script.")

def get_sitemap(urls):
    request = Request(
                url= urls,
                headers={'User-Agent': 'Mozilla/5.0'}
                  )
    response = urlopen(request)
    xml = BeautifulSoup(response, 
                        'lxml-xml', 
                        from_encoding=response.info().get_param('charset'))
    
    return xml


def get_url(xml):

    urls = []
    for url in xml.find_all("url"):
        if url.find("loc"):
            urls.append(url.findNext("loc").text)
    return urls

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        formatted_docs.append(doc.page_content)
    return "\n\n---\n\n".join(formatted_docs)

if Path(persist_directory).exists():
    print(f"Loading existing vector store from {persist_directory} \n")
    vectore_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name="rag_Financial_LLM")  # Load existing vector store
    
else:
    print(f"Creating new vector store at {persist_directory} \n")
    xmls = get_sitemap("https://zerodha.com/varsity/chapter-sitemap2.xml")
    list_urls = get_url(xmls)

    #Loading Langchain Documents from the list of URLs
    docs = []
    for i, url in enumerate(list_urls):
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
        if i % 10 ==0:
            print(f"Loaded {i} documents \n")

    #Splitting the documeents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50, separators = ["\n\n", "\n", " ", "" ])  # Create text splitter
    splits = text_splitter.split_documents(docs)

    vectore_store = Chroma.from_documents(documents=splits, 
                                          embedding=embeddings, 
                                          persist_directory=persist_directory, 
                                          collection_name="rag_Financial_LLM", 
                                          collection_metadata={"hnsw:space": "cosine"})  # Create vector store

print(f"Vector store created and persisted at {persist_directory} \n")

# ============================================================================
# PART 2: Creating Retriver
# ============================================================================

retriever = vectore_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Create retriever

print("\n Retriever created \n")

#test retriever
# query = "What is Index?"
# retrieved_docs = retriever.invoke(query)
# for i, doc in enumerate(retrieved_docs, 1):
#     print(f"/n #{i} - {doc.metadata['title']} \n {doc.page_content} \n")


# ============================================================================
# PART 3: Initialize LLM and Prompt Template
# ============================================================================

if os.getenv("OPENAI_API_KEY"):
    print(f"\nOPENAI_API_KEY found. {os.getenv('OPENAI_API_KEY')[:10]}...{os.getenv('OPENAI_API_KEY')[-10:]} \n")
    llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL"), 
                     temperature=0,
                     timeout=120,
                     max_retries=3,
                     streaming=True, 
                     callbacks=[StreamingStdOutCallbackHandler()])  # Create LLM
else:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the script.")

chat_history = []
chat_prompt = ChatPromptTemplate.from_messages([
    "system", """YYou are Financial assistant AI. Answer using the document context below and the chat history.
    Use chat history to resolve references like "that issue" or "that stock".
    For factual claims, prioritize the retrieved context.
    If information is not available in context or history, say "I don't have that information."
    Always cite sources when available.""",
    MessagesPlaceholder(variable_name="history"),
    "human", "{question}"])    

# ============================================================================
# PART 4: RAG Pipeline Creation
# ============================================================================
qachain = chat_prompt | llm | StrOutputParser()  # Create QA chain

def ask_with_history(question: str, history: list[dict]):
    global chat_history
    context = format_docs(retriever.invoke(question))  # Retrieve context
    for item in history:
        if item["role"] == "user":
            chat_history.append(HumanMessage(content=item.get("content", "")))
        elif item["role"] == "assistant":
            chat_history.append(AIMessage(content=item.get("content", "")))
    response = qachain.invoke({"question": question, "history": chat_history, "context": context})
    return response


 
 # ============================================================================
# PART 5: ChatBOT UI with Gradio
# ============================================================================

with gr.Blocks(theme="monochrome") as demo:
    gr.Markdown("## Financial Assistant Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question about finance:")
    with gr.Row():
        clear = gr.Button("Clear")
        undo = gr.Button("Undo Last Message")
        # with gr.Column(scale=0.85):
        #     msg.submit(respond, [msg, chatbot], [msg, chatbot])
        # with gr.Column(scale=0.15, min_width=0):
            
    gr.ChatInterface(fn=ask_with_history, chatbot = chatbot, textbox=msg)
    def undo_last_message(chat_history):
        if chat_history:
            chat_history.pop()  # Remove the last user message
            chat_history.pop()  # Remove the last AI response
        return chat_history
    undo.click(undo_last_message, chatbot, chatbot)
    clear.click(lambda: [], None, chatbot)
    undo.click(undo_last_message, chatbot, chatbot)
demo.launch(share=True, debug=False)