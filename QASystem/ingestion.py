from langchain_community.document_loaders import PyPDFDirectoryLoader


def data_ingestion():
    loader = PyPDFDirectoryLoader("data")