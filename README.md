# AmbedkarGPT-Intern-Task

This project is a Retrieval-Augmented Generation (RAG) command-line Q&A system built with Python  
It answers questions **using only the content of Dr. B.R. Ambedkar's provided speech**.  
All processing is local and open-source.

- **Framework:** LangChain (for RAG pipeline)
- **Vector DB:** ChromaDB (local)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`, local)
- **LLM:** Ollama with Mistral (local LLM, no API key or cloud calls)

Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running
- Sufficient disk space for vector DB and embeddings

Setup Instructions

1. Clone or download this repository

Put these files in the same folder:
- `main.py` (the Q&A script)
- `requirements.txt` (Python dependencies)
- `speech.txt` (the text of the Ambedkar speech; provided in the assignment or this repo)

2. Create and activate a virtual environment

python -m venv myenv

On Windows:
myenv\Scripts\activate

On macOS/Linux:
source myenv/bin/activate

3. Install Python dependencies

pip install --upgrade pip
pip install -r requirements.txt

4. Install and set up Ollama (if not already present)

Download and install ollama:
curl -fsSL https://ollama.ai/install.sh | sh

Then, pull and run the Mistral model:
ollama pull mistral
ollama run mistral

Let ollama be running in the background

5. Run the Q&A program
python main.py
