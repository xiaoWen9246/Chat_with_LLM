import sys 
sys.path.append("../")
from llm.wenxin_llm import Wenxin_LLM
from llm.spark_llm import Spark_LLM
from llm.zhipuai_llm import ZhipuAILLM
from langchain_community.chat_models import ChatOpenAI
from llm.call_llm import parse_llm_api_key
from llm.call_llm import parse_llm_base_url

def model_to_llm(model:str=None, temperature:float=0.0):
        """
        星火：model,temperature,appid,api_key,api_secret
        百度问心：model,temperature,api_key,api_secret
        智谱：model,temperature,api_key
        OpenAI：model,temperature,api_key
        """
        if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
            base_url = parse_llm_base_url("openai")
            api_key= parse_llm_api_key("openai")
            llm = ChatOpenAI(model_name = model, temperature = temperature , openai_api_key = api_key, base_url = base_url)
        elif model in ["ernie-3.5", "ernie-3.5-8k", "ernie-lite","ernie-4.0"]:
            # if api_key == None or Wenxin_secret_key == None:
            #     api_key, Wenxin_secret_key = parse_llm_api_key("wenxin")
            llm = Wenxin_LLM(model_name=model, temperature = temperature)
        elif model in ["Spark4.0 Ultra","Spark Max-32K","Spark Max","Spark Pro-128K","Spark Pro","Spark Lite"]:
            api_key, appid, Spark_api_secret = parse_llm_api_key("spark")
            llm = Spark_LLM(model_name=model, temperature = temperature, appid=appid, api_secret=Spark_api_secret, api_key=api_key)
        elif model in ["glm-4-plus", "glm-4-airx"]:
            api_key = parse_llm_api_key("zhipuai")
            api_base = parse_llm_base_url("zhipuai")
            llm = ChatOpenAI(model = model, openai_api_key = api_key, openai_api_base = api_base)
            #llm = ZhipuAILLM(model=model, zhipuai_api_key=api_key, temperature = temperature)
        else:
            raise ValueError(f"model{model} not support!!!")
        return llm