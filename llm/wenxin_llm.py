#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
from pydantic import Field
from llm.self_llm import Self_LLM
import json
from langchain.callbacks.manager import CallbackManagerForLLMRun
import erniebot
from llm.call_llm import parse_llm_api_key



class Wenxin_LLM(Self_LLM):
    # 文心大模型的自定义 LLM
    # URL
    # url : str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={}"
    # Secret_Key
    secret_key: str = None
    # access_token
    access_token: str = None

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        self.access_token = parse_llm_api_key("wenxin")
        # 如果 access_token 为空，初始化 access_token
        if self.access_token == None:
            return "缺少access_token"
        
        erniebot.api_type = "aistudio"
        erniebot.access_token = self.access_token
        # 发起请求
        messages = [{"role": "user", "content": prompt}]
        response = erniebot.ChatCompletion.create(messages=messages,
                                                  model=self.model_name,
                                                  temperature=self.temperature
                                                  )
        return response["result"]

    @property
    def _llm_type(self) -> str:
        return "Wenxin"