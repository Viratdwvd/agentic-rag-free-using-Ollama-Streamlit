
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = Path("data")
PERSIST_DIR = ".chroma"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

def load_docs(data_dir: Path):
    docs = []
    for p in data_dir.rglob("*"):
        if p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        elif p.suffix.lower() in {".txt", ".md"}:
            loader = TextLoader(str(p), encoding="utf-8")
            docs.extend(loader.load())
    return docs

def main():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    print(f"Loading from: {DATA_DIR.resolve()}")
    docs = load_docs(DATA_DIR)
    if not docs:
        print("No documents found. Put PDFs or .txt files in the `data/` folder and re-run.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print(f"Ingested {len(chunks)} chunks into Chroma at {PERSIST_DIR!r}.")

if __name__ == "__main__":
    main()
