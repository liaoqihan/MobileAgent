import argparse
import base64
from typing import List
import requests
import time
from retrying import retry
import base64
import requests
import os
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion
from enum import Enum

from .util import print_execution_time




@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000)
def get_model_response(ak, prompt: str, images: List[str]) -> (bool, str):
    # print(f"QaModel.get_model_response:\n prompt:\n{prompt}\n")
    # base64_imgs = [encode_image(image) for image in images]
    base64_imgs = images
    headers = {
        "Content-Type": "application/json",
        "Authorization": ak
    }
    body ={
        "prompt": prompt,
        "images": base64_imgs
    }
    base_url = "https://fmh.alibaba-inc.com/api/azure/chat/vision/list"
    response = requests.post(url=base_url, headers=headers, json=body).json()
    # print(f"QaModel response:\n {response}")
    # if not response.get("data"):
    #     return response
    if not response.get("data") or not response.get("success"):
        print(f"QaModel.get_model_response Error res:\n{response}")
        raise Exception("Response not successful")
    return response["data"]



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
    
@print_execution_time
@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000)
def inference_chat(chat, model, api_url, token,use_qa=True):
    if not use_qa:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        data = {
            "model": model,
            "messages": [],
            "max_tokens": 2048,
            'temperature': 0.0,
            "seed": 1234
        }

        for role, content in chat:
            data["messages"].append({"role": role, "content": content})

        while True:
            try:
                res = requests.post(api_url, headers=headers, json=data)
                res_json = res.json()
                res_content = res_json['choices'][0]['message']['content']
            except:
                print("Network Error:")
                try:
                    print(res.json())
                except:
                    print("Request Failed")
                time.sleep(1)
            else:
                break
        
        return res_content

    # prompt, images = extract_content(chat)
    # return get_model_response(token,prompt,images)

    return inference_chat_azure(chat, model)
    
def extract_content(obj):
    texts = []
    images = []

    def extract_texts_and_images(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'type':
                    if value == 'text' and 'text' in obj:
                        texts.append(obj['text'])
                    elif value == 'image_url' and 'image_url' in obj:
                        images.append(obj['image_url']["url"])
                else:
                    extract_texts_and_images(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_texts_and_images(item)

    extract_texts_and_images(obj)
    prompt = "\n\n".join(texts)
    return prompt, images



class ICBUGPTModels(Enum):
    GPT_3_5_TURBO = "gpt-35-turbo"
    GPT_3_5_TURBO_16K = "gpt-35-turbo-16k"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_VISION = "gpt-4-vision"
    GPT_4_O = "gpt-4o"

def convert_model_name(input_model_name: str) -> str:
    mapping = {
        "gpt-3.5": ICBUGPTModels.GPT_3_5_TURBO.value,
        "gpt-3.5-turbo": ICBUGPTModels.GPT_3_5_TURBO.value,
        "gpt-3.5-turbo-16k": ICBUGPTModels.GPT_3_5_TURBO.value,
        "gpt-4": ICBUGPTModels.GPT_4_TURBO.value,
        "gpt-4 8K": ICBUGPTModels.GPT_4_TURBO.value,
        "gpt-4-8K": ICBUGPTModels.GPT_4_TURBO.value,
        "gpt-4 32k": ICBUGPTModels.GPT_4_TURBO.value,
        "gpt-4-32k": ICBUGPTModels.GPT_4_TURBO.value,
        "gpt-4-turbo": ICBUGPTModels.GPT_4_TURBO.value,
    }
    return mapping.get(input_model_name, input_model_name)


client = AzureOpenAI(
    default_headers={"empId": os.getenv('empId')},
    azure_endpoint=os.getenv('api_url'),
    api_key = os.getenv('api_key'),
    api_version = "2024-03-01-preview",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000)
def inference_chat_azure(chat, model):

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': 0.0,
        "seed": 1234
    }

    # 将prompt塞进request data的message数组中
    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    chat_completion = client.chat.completions.create(
        model=ICBUGPTModels.GPT_4_O.value,
        messages=data["messages"],
        timeout=600,
        max_tokens=4096
    )
    try:
        res_content = chat_completion.choices[0].message.content
        usage = chat_completion.usage
        tokens_info = {
            "total_tokens": usage.total_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens
        }
        print(f"tokens_info: {tokens_info}")
    except Exception as e:
        print(f"inference_chat_azure Error:{e} \n res:{chat_completion} ")
        raise Exception(f"inference_chat_azure Error:{e} \n res:{chat_completion} ")
    if not res_content:
        print(f"inference_chat_azure res_content empty res:{chat_completion}")
        raise Exception(f"inference_chat_azure res_content empty res:{chat_completion}")
    return res_content
