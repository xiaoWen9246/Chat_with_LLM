import sys 
# sys.path.append("../embedding") 
# sys.path.append("../database") 

from langchain_community.embeddings import OpenAIEmbeddings    # 调用 OpenAI 的 Embeddings 模型
import os
from database.create_db import create_db,load_knowledge_db
from embedding.call_embedding import get_embedding
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import EnsembleRetriever

def get_vectordb(file_path:str=None, persist_path:str=None, embedding = "openai",embedding_key:str=None):
    """
    返回向量数据库对象
    输入参数：
    question：
    llm:
    vectordb:向量数据库(必要参数),一个对象
    template：提示模版（可选参数）可以自己设计一个提示模版，也有默认使用的
    embedding：可以使用zhipuai等embedding，不输入该参数则默认使用 openai embedding，注意此时api_key不要输错
    """
    embedding = get_embedding(embedding=embedding, embedding_key=embedding_key)
    if os.path.exists(persist_path):  #持久化目录存在
        contents = os.listdir(persist_path)
        if len(contents) == 0:  #但是下面为空
            #print("目录为空")
            vectordb = create_db(file_path, persist_path, embedding)
            #presit_knowledge_db(vectordb)
            vectordb, bm25_retriever = load_knowledge_db(persist_path, embedding)
        else:
            #print("目录不为空")
            vectordb, bm25_retriever = load_knowledge_db(persist_path, embedding)
    else: #目录不存在，从头开始创建向量数据库
        vectordb = create_db(file_path, persist_path, embedding)
        #presit_knowledge_db(vectordb)
        vectordb, bm25_retriever = load_knowledge_db(persist_path, embedding)

    return vectordb, bm25_retriever


def get_retriever(vectordb, bm25_retriever, top_k=3):
    Chroma_retriever = vectordb.as_retriever(search_type="similarity",
                                                      search_kwargs={'k': top_k})
    bm25_retriever.k = top_k
    retriever = EnsembleRetriever(
            retrievers=[bm25_retriever,
                        Chroma_retriever], weights=[0.5, 0.5]
        )
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=top_k)
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
    )
    
    return compression_retriever