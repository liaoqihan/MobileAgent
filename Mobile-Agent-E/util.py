



import base64
from dataclasses import dataclass, field, asdict
import mimetypes
import os
from typing import Dict, List, Optional, Union
import uuid
import requests
import json


scene_appCode_map = {
    "体验问题识别": "pDwdGSKuvAs",
    "价格一致性":"QrbnhMwByKA",
    "planning":"jxkRkfQVXsx",
    "actions":"XrpWDXnXIqP",
    "action_reflection": "rrOTihLcWvD",
    "note":"JDBgIgpgifJ"
}
variableMap = {'country': '美国', 'language': '英语', 'currency': 'USD'}


def generate_session_id():
    return str(uuid.uuid4())


def call_austudio_api(aistudio_ak,image_urls, variableMap=variableMap,scene="体验问题识别",appVersion="latest",question="请回答",session_id=None):
    """
    agent for 获取点击坐标
    """
    appCode = scene_appCode_map.get(scene)
    url = f'https://aistudio.alibaba-inc.com/api/aiapp/run/{appCode}/{appVersion}'
    headers = {
        "accept": "*/*",
        "Content-Type": "application/json",
        "X-AK": aistudio_ak
    }
    if not session_id:
        session_id = generate_session_id()

    media_entities_list = gen_media_entities_list_for_call_aistudio(image_urls)


    data = {
        "mediaEntities": media_entities_list,
        "empId": "109547",
        "question": question,
        "sessionId": session_id,
        "stream": "false",
        "variableMap": variableMap
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        ret = json.loads(response.text)
        result = ret['data']
        content = result['content']
        return content
    except Exception as e:
        print("ai_agent error :%s", e.args)
        return None
    

def convert_image_input(image_url):
    """
    调用AI时传入的image_url是个列表，元素为HTTP URL或Base64。
    如果传入的是文件路径，需要解析成Base64返回。
    """
    if type(image_url) is str:
        image_url = [image_url]

    results = []
    for item in image_url:
        # 检查是否是一个HTTP URL
        if item.startswith('http://') or item.startswith('https://'):
            results.append(item)
        
        # 检查是否是Base64字符串（简单验证）
        elif item.startswith('data:image') or len(item) % 4 == 0:
            results.append(item)
        
        # 检查是否是文件路径
        elif os.path.isfile(item):
            # 获取文件的MIME类型
            mime_type, _ = mimetypes.guess_type(item)
            if mime_type and mime_type.startswith('image'):
                with open(item, "rb") as image_file:
                    # 读取文件并编码为Base64
                    encoded_string = base64.b64encode(image_file.read()).decode()
                    # 将Base64编码结果添加 MIME 类型前缀
                    base64_str = f"data:{mime_type};base64,{encoded_string}"
                    results.append(base64_str)
        else:
            raise ValueError(f"Unrecognized input type: {item}")

    return results

def gen_media_entities_list_for_call_aistudio(image_url):
    """
    返回调用aistudio时传入image时需要的格式
    """
    media_entities_list = []
    image_urls = convert_image_input(image_url)

    if isinstance(image_urls, str):
        one_media_entity = {
            "content": image_urls
        }
        media_entities_list.append(one_media_entity)
    elif isinstance(image_urls, list):
        for i in range(0, len(image_urls)):
            one_media_entity = {
                "content": image_urls[i]
            }
            media_entities_list.append(one_media_entity)
    return media_entities_list


def get_user_prompt_from_origin_mes(chat):
    """
    为使源代码更好兼容aistudio调用方式，从接口请求中提取最新的用户输入。
    """
    text = ""
    image_urls = []
    for message in reversed(chat):
        role, contents = message
        if role == "user" and isinstance(contents, list) and len(contents) > 0:
            text = contents[0].get("text")
            if len(contents) > 1:
                image_urls = [contents[i].get("image_url").get("url") for i in range(1, len(contents))]
    return text, image_urls

    
