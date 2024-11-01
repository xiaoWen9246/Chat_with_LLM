from dotenv import load_dotenv, find_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

import tempfile
import os
import sys
import pickle
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from embedding.call_embedding import get_embedding


DEFAULT_DB_PATH = "../knowledge_db"
DEFAULT_PERSIST_PATH = "./vector_db"

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_url = os.getenv('BASE_URL')


class HybridRetriever:
    def __init__(self, vectordb, bm25_retriever):
        self.vectordb = vectordb
        self.bm25_retriever = bm25_retriever

    def search(self, query, top_k=3):
        Chroma_retriever = self.vectordb.as_retriever(search_type="similarity",
                                                      search_kwargs={'k': top_k})
        self.bm25_retriever.k = top_k
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever,
                        Chroma_retriever], weights=[0.5, 0.5]
        )
        result = ensemble_retriever.invoke(query)

        return result
        


def get_files(dir_path):
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            file_list.append(os.path.join(filepath, filename))
    return file_list


def file_loader(file, loaders):
    # 检查file是否临时文件，
    if isinstance(file, tempfile._TemporaryFileWrapper):
        file = file.name
    # 检查file是否是文件，如果是目录进一步处理
    if not os.path.isfile(file):
        [file_loader(os.path.join(file, f), loaders) for f in os.listdir(file)]
        return
    file_type = file.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file))
    elif file_type == 'md':
        pattern = r"不存在|风控"
        match = re.search(pattern, file)
        if not match:
            loaders.append(UnstructuredMarkdownLoader(file))
    elif file_type == 'txt':
        loaders.append(UnstructuredFileLoader(file))
    return


def create_db_info(files=DEFAULT_DB_PATH, embeddings="openai", persist_directory=DEFAULT_PERSIST_PATH):
    if embeddings == 'openai' or embeddings == 'm3e':
        vectordb = create_db(files, persist_directory, embeddings)
    return ""


def create_db(files=DEFAULT_DB_PATH, persist_directory=DEFAULT_PERSIST_PATH, embeddings="openai"):
    """
    该函数用于加载 PDF 文件，切分文档，生成文档的嵌入向量，创建向量数据库。

    参数:
    file: 存放文件的路径。
    embeddings: 用于生产 Embedding 的模型

    返回:
    vectordb: 创建的数据库。
    """
    if files == None:
        return "can't load empty file"
    if type(files) != list:
        files = [files]
    loaders = []
    [file_loader(file, loaders) for file in files]
    docs = []
    for loader in loaders:
        if loader is not None:
            docs.extend(loader.load())
    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",  # 按段落分割
            "\n",   # 按行分割
            " ",   # 按空格分割
            ".",  # 按句号分割
            ",",  # 按逗号分割
            "\uff0c",  # 中文全角逗号
            "\u3001",  # 中文顿号
            "\uff0e",  # 中文全角句号
            "\u3002",  # 中文句号
            "",  # 最后逐字符分割
        ],
        chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    if type(embeddings) == str:
        embeddings = get_embedding(embedding=embeddings)
    # 定义持久化路径
    persist_directory = './vector_db/chroma'
    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
    vectordb.persist()
    print(f"向量库中存储的数量：{vectordb._collection.count()}")

    bm25_retriever = BM25Retriever.from_documents(
        documents = split_docs
    )

    with open('retriever.pkl', 'wb') as f:
      pickle.dump(bm25_retriever, f)

    print('finished')
    return vectordb


def presit_knowledge_db(vectordb):
    """
    该函数用于持久化向量数据库。

    参数:
    vectordb: 要持久化的向量数据库。
    """
    vectordb.persist()


def load_knowledge_db(path, embeddings):
    """
    该函数用于加载向量数据库。

    参数:
    path: 要加载的向量数据库路径。
    embeddings: 向量数据库使用的 embedding 模型。

    返回:
    vectordb: 加载的数据库。
    """
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "retriever.pkl")
    with open(file_path, 'rb') as f:
        bm25_retriever = pickle.load(f)
    return vectordb, bm25_retriever


if __name__ == "__main__":
    create_db(files=DEFAULT_DB_PATH, embeddings="m3e")
