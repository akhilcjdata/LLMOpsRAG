import os
import json
import boto3
import sys 
import streamlit as streamlit
from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa
from langchain_community.vectorstores import FAISS


