
from crewai import Agent, Task, Crew
from pathlib import Path
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from pydantic import PrivateAttr
from crewai.tools import BaseTool 
from dotenv import load_dotenv
load_dotenv() 
# Load and chunk documents
print("HuggingFace:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
loader = TextLoader("docs/knowledge_base.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Use HuggingFace SentenceTransformer embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_path = "faiss_index"
if Path(index_path).exists():
    print("Loading saved FAISS index...")
    vector_store = FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    print("⚙️  Creating FAISS index...")
    loader = TextLoader("docs/knowledge_base.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents
    (chunks, embedding_model)
    vector_store.save_local(index_path)

# Step 3: Define HuggingFaceHub LLM (text generation)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
)

# Step 4: RAG Chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Step 5: Define Agents

class RAGQATool(BaseTool):
    name: str = "rag_qa_tool"
    description: str = "Answers questions using RAG"
    _qa_chain: RetrievalQA = PrivateAttr()

    def __init__(self, qa_chain):
        super().__init__()
        self._qa_chain = qa_chain


    def _run(self, query: str) -> str:
        return self._qa_chain.run(query)

    async def _arun(self, query: str) -> str:
        return self._qa_chain.run(query)

# Wrap the tool
rag_tool = RAGQATool(qa_chain=rag_chain)

# Create the agent
rag_agent = Agent(
    role="RAG Agent",
    goal="Answer user queries with supporting facts using the document base.",
    backstory="You rely on retrieved documents to ground your responses.",
    tools=[rag_tool],
    allow_delegation=False
)

critic_agent = Agent(
    role="Critique Agent",
    goal="Review the RAG Agent's answer for accuracy and faithfulness to the retrieved context",
    backstory="You're a critical reviewer checking for hallucinations and clarity.",
    allow_delegation=False
)

# Step 6: Define Tasks
rag_task = Task(
    description='Answer the question: "What is CrewAI and how does it help with AI workflows?"',
    expected_output="A clear, complete, and factual answer grounded in the knowledge base.",
    agent=rag_agent,
    output_file="rag_output.txt"
)

critic_task = Task(
    description='Read rag_output.txt and provide a critique for factuality, clarity, and alignment with the documents.',
    expected_output="Detailed response based on document knowledge base",
    agent=critic_agent
)

# Step 7: Crew Execution
crew = Crew(
    agents=[rag_agent, critic_agent],
    tasks=[rag_task, critic_task],
    verbose=True
)
crew.kickoff()
