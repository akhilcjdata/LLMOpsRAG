from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
import boto3


bedrock = boto3.client(service_name = "bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)
  

def data_ingestion():
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()

    text_spliter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlaps = 1000)
    docs = text_spliter.split_documents(documents)
    return docs


def get_vector_store(doc):
    vector_store_faiss = FAISS.from_documents(doc,BedrockEmbeddings)
    vector_store_faiss.save_local("faiss index")



if __name__ == "__main__":
    docs = data_ingestion()
    get_vector_store(docs)
