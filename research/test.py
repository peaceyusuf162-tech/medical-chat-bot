import os
from typing import List
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI  # Fixed for Gemini
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Environment Variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMMA_API_KEY  # LangChain Gemini wrapper looks for GOOGLE_API_KEY

# 2. Extract text from PDF
def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",  # Fixed missing period in glob
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Execute PDF loading
extracted_data = load_pdf_files("data")
print(f"Loaded {len(extracted_data)} document pages.")

# 3. Filter metadata to keep things minimal
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

minimal_docs = filter_to_minimal_docs(extracted_data)

# 4. Chunking operation
def text_split(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(docs)
    return texts_chunk

texts_chunk = text_split(minimal_docs)
print(f"Number of chunks: {len(texts_chunk)}")

# 5. Download Hugging Face Embeddings
def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

embedding = download_embeddings()

# 6. Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Matches all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index and upsert documents
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)

# Optional: How to load existing index later without re-uploading
# docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

# Optional: Add standalone document
dswith = Document(
    page_content="peace is a software developer ai engineer",
    metadata={"source": "computer-scientist"}
)
docsearch.add_documents(documents=[dswith])

# 7. Create Retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 8. Setup LLM and RAG Chain
# Using the dedicated Google Gemini class instead of ChatOpenAI
chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

system_prompt = (
    "You are a medical assistant for question-answering tasks.\n"
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),  # Fixed: changed (input) to {input}
    ]
)

question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 9. Run Query
response = rag_chain.invoke({"input": "what is Acne?"})
print("\n--- Answer ---")
print(response["answer"])