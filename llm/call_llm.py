#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import erniebot.api_types
from openai import OpenAI
import erniebot
import json
import requests
import _thread as thread
import base64
import datetime
from dotenv import load_dotenv, find_dotenv
import hashlib
import hmac
import os
import queue
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from zhipuai import ZhipuAI
from langchain.utils import get_from_dict_or_env
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

import websocket  # 使用websocket_client

def get_completion(prompt :str, model :str, temperature=0.1, max_tokens=2048):
    # 调用大模型获取回复，支持上述三种模型+gpt
    # arguments:
    # prompt: 输入提示
    # model：模型名
    # temperature: 温度系数
    # api_key：如名
    # secret_key, access_token：调用文心系列模型需要
    # appid, api_secret: 调用星火系列模型需要
    # max_tokens : 返回最长序列
    # return: 模型返回，字符串
    # 调用 GPT
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
        return get_completion_gpt(prompt, model, temperature, max_tokens)
    elif model in ["ernie-3.5", "ernie-3.5-8k", "ernie-lite","ernie-4.0"]:
        return get_completion_wenxin(prompt, model, temperature)
    elif model in ["Spark4.0 Ultra","Spark Max-32K","Spark Max","Spark Pro-128K","Spark Pro","Spark Lite"]:
        return get_completion_spark(prompt, model, temperature, max_tokens)
    elif model in ["glm-4-plus", "glm-4-airx"]:
        return get_completion_glm(prompt, model, temperature, max_tokens)
    else:
        return "不正确的模型"
    
def get_completion_gpt(prompt : str, model : str, temperature : float, max_tokens:int, api_key : str = None):
    # 封装 OpenAI 原生接口
    if api_key is None:
        api_key = parse_llm_api_key("openai")
    api_base = parse_llm_base_url("openai")
    client = OpenAI(api_key = api_key, base_url = api_base)
    # 具体调用
    messages = [{"role": "user", "content": prompt}]

   
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = temperature,
        max_tokens = max_tokens
    )

    return response.choices[0].message.content



def get_completion_wenxin(prompt : str, model : str, temperature : float, access_token : str = None):
    
    if access_token is None:
        access_token = parse_llm_api_key("wenxin")
    erniebot.api_type = "aistudio"
    erniebot.access_token = access_token
    messages = [{"role": "user", "content": prompt}]
    response = erniebot.ChatCompletion.create(
        model = model,
        messages=messages,
        temperature=temperature,
    )
    
    return response["result"]

def get_completion_spark(prompt : str, model : str, temperature : float,max_tokens : int, api_key:str = None, appid : str = None, api_secret : str=None):
    if api_key == None or appid == None and api_secret == None:
        api_key, appid, api_secret = parse_llm_api_key("spark")
    url, domain = get_url_domain_spark(model)
    spark = ChatSparkLLM(
        spark_api_url= url,
        spark_app_id=appid,
        spark_api_key=api_key,
        spark_api_secret=api_secret,
        spark_llm_domain=domain,
        streaming=False,
    )
    messages = [ChatMessage(
        role="user",
        content=prompt
    )]
    handler = ChunkPrintHandler()
    response = spark.generate([messages], callbacks=[handler])
    return response.generations[0][0].text

def get_completion_glm(prompt : str, model : str, temperature : float , max_tokens : int, api_key:str = None):
    # 获取GLM回答
    if api_key == None:
        api_key = parse_llm_api_key("zhipuai")
    client = ZhipuAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"



def get_url_domain_spark(model:str):
    url_domain_map = {"Spark4.0 Ultra": {"url":"wss://spark-api.xf-yun.com/v4.0/chat", "domain":"4.0Ultra"}, 
                  "Spark Max-32K":{"url":"wss://spark-api.xf-yun.com/chat/max-32k","domain":"max-32k"},
                  "Spark Max":{"url":"wss://spark-api.xf-yun.com/v3.5/chat","domain":"generalv3.5"},
                  "Spark Pro-128K":{"url":"wss://spark-api.xf-yun.com/chat/pro-128k","domain":"pro-128k"},
                  "Spark Pro":{"url":"wss://spark-api.xf-yun.com/v3.1/chat","domain":"generalv3"},
                  "Spark Lite":{"url":"wss://spark-api.xf-yun.com/v1.1/chat","domain":"general"}
                  }
    return url_domain_map[model]["url"], url_domain_map[model]["domain"]


def parse_llm_api_key(model:str, env_file:dict()=None):
    """
    通过 model 和 env_file 的来解析平台参数
    """   
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    if model == "openai":
        return env_file["OPENAI_API_KEY"]
    elif model == "wenxin":
        return env_file["EB_ACCESS_TOKEN"]
    elif model == "spark":
        return env_file["spark_api_key"], env_file["spark_appid"], env_file["spark_api_secret"]
    elif model == "zhipuai":
        return get_from_dict_or_env(env_file, "zhipuai_api_key", "ZHIPUAI_API_KEY")
        # return env_file["ZHIPUAI_API_KEY"]
    else:
        raise ValueError(f"model{model} not support!!!")
    
def parse_llm_base_url(model:str, env_file:dict()=None):
    """
    通过 model 和 env_file 来解析url
    """
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    if model == "openai":
        return env_file["OPENAI_BASE_URL"]
    elif model == "zhipuai":
        return env_file["ZHIPUAI_API_BASE"]
    else:
        raise ValueError(f"model{model} not support!!!")
