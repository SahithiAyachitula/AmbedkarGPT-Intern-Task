#import the libraries
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama


def main():
    #Load the Ambedkar speech text file
    loader = TextLoader("speech.txt")
    docs = loader.load()
    
    #Split the text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunked_docs = splitter.split_documents(docs)
    
    #Create HuggingFace embeddings
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    #Store embeddings in Chroma vector store
    db = Chroma.from_documents(chunked_docs, embedding, persist_directory="./chroma_db")
    
    #Connect to local Ollama LLM
    llm = ChatOllama(model="mistral", temperature=0.2)
    
    print("\nAmbedkar Speech Q&A System\nType your question and press Enter. Type 'exit' to quit.")
    
    #Interactive Q&A loop
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Retrieve most relevant chunks
        retriever = db.as_retriever(search_kwargs={'k': 3})
        relevant_docs = retriever.invoke(question)
        context = "\n".join(d.page_content for d in relevant_docs)

        # Build prompt for LLM: only use context
        prompt = (
            "You are a helpful assistant. Use ONLY the context below to answer the question. "
            "If the answer is not present, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        # Get answer from LLM
        response = llm.invoke(prompt)
        print("Answer:", response.content)

if __name__ == "__main__":
    main()
