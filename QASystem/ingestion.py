from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings  # Updated import
import boto3

bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def data_ingestion():
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()
    # Fixed parameter name: chunk_overlap (not chunk_overlaps)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    # Use the bedrock_embeddings instance, not the class
    vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss

if __name__ == "__main__":
    docs = data_ingestion()
    get_vector_store(docs)