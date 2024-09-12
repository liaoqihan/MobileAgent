from dataclasses import dataclass, field, asdict
import os
from typing import Dict, List, Optional, Union
import uuid
import requests
import json
def generate_session_id():
    return str(uuid.uuid4())

def generate_unique_id(len=64):
    return uuid.uuid4().int >> len  # 取前64位

image_urls = [
"http://b2b-algo-test.oss-cn-hangzhou-zmf.aliyuncs.com/lqh/%E9%A6%96%E9%A1%B5%E8%BF%9B%E5%85%A5'Supplier%20leaderboard'%E4%B9%9D%E6%9C%88%E5%A4%A7%E4%BF%83%E4%BC%9A%E5%9C%BA%20%E9%80%89%E6%8B%A9%E4%B8%80%E4%B8%AA%E5%95%86%E5%93%81%E4%B8%8B%E5%8D%95%EF%BC%88%E5%A6%82%E6%9E%9C%E6%97%A0%E6%B3%95%E4%B8%8B%E5%8D%95%20%E5%88%99%E6%B2%9F%E9%80%9A%E8%AF%A2%E7%9B%98%EF%BC%89_20240906104205/iter_1_screenshot.png"
]
aistudio_ak = os.getenv('aistudio_ak')
def ai_agent_rec_bug(image_urls=image_urls, country="美国", language="英语", currency="USD"):
    """
    agent for 获取点击坐标
    """
    url = f'https://aistudio.alibaba-inc.com/api/aiapp/run/pDwdGSKuvAs/latest'
    headers = {
        "accept": "*/*",
        "Content-Type": "application/json",
        "X-AK": aistudio_ak
    }
    session_id = generate_session_id()

    media_entities_list = []
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

    data = {
        "mediaEntities": media_entities_list,
        "empId": "109547",
        "question": "检测图片的体验问题",
        "sessionId": session_id,
        "stream": "false",
        "variableMap": {
            "country": country,
            "language": language,
            "currency": currency
        }
    }
    print("agent api:", url)
    print("agent data:", data)
    response = requests.post(url, json=data, headers=headers)
    print(response.text)
    try:
        ret = json.loads(response.text)
        result = ret['data']
        content = result['content']
        print(content)
        # result = json.loads(content)
        return content
    except Exception as e:
        print("ai_agent error :%s", e.args)
        return None
    

@dataclass
class BizNodeResult:

    description: str
    actionName: str
    imgList: List[str]
    bizTag: str = "DEFAULT"
    id: int = generate_unique_id()
    actionType: str = "NORMAL_CLICK"
    driverAssert: str = ""
    result: str = "INIT"
    bizDesc: Optional[str] = None 
    dataList: Optional[List] = None
    exception: str = ""
    note: Optional[str] = None
    actionFlag: Optional[str] = None
    testCaseHistoryId: str = ""

@dataclass
class UploadChatResultRequest:
    instruction: str
    bizNodesResult: List[BizNodeResult]
    takeTime: str = "300.00"
    exception: str = ""
    taskRecordId: int = 150966 # https://pre-xl.alibaba-inc.com/#/task/taskDetail?id=2886&tab=taskRecord&taskRecordId=150966
    testCaseHistoryId: Optional[str] = None


def upload_chat_result(uploadChatResultRequest):

    url = "http://pre-xl.alibaba-inc.com/api/uploadChatResult"
    req_dict = asdict(uploadChatResultRequest)
    print(f"upload_chat_result req:\n {json.dumps(req_dict,ensure_ascii=False)}")
    response = requests.post(url, json=req_dict)
    is_success = False
    try:
        is_success = response.json().get("success")
        print(f"upload_chat_result response:\n {response.json()}")
    except Exception as e:
        pass

    if not is_success:
        raise Exception(f"upload_chat_result failed response:{response.text}")
    historyId = response.json().get("data")
    print(f"upload_chat_result url: https://pre-xl.alibaba-inc.com/#/task/testCaseHistoryDetail?historyId={historyId}")
    return response


if __name__ == "__main__":
    results = {}
    for i in range(1,10):
        image_urls = [f"http://b2b-algo-test.oss-cn-hangzhou-zmf.aliyuncs.com/lqh/%E9%A6%96%E9%A1%B5%E8%BF%9B%E5%85%A5'Supplier%20leaderboard'%E4%B9%9D%E6%9C%88%E5%A4%A7%E4%BF%83%E4%BC%9A%E5%9C%BA%20%E9%80%89%E6%8B%A9%E4%B8%80%E4%B8%AA%E5%95%86%E5%93%81%E4%B8%8B%E5%8D%95%EF%BC%88%E5%A6%82%E6%9E%9C%E6%97%A0%E6%B3%95%E4%B8%8B%E5%8D%95%20%E5%88%99%E6%B2%9F%E9%80%9A%E8%AF%A2%E7%9B%98%EF%BC%89_20240906104205/iter_{i}_screenshot.png"]
        res = ai_agent_rec_bug(image_urls=image_urls, country="美国", language="英语", currency="USD")
        results[i] = res
    with open("tmp.json", "w") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)


@dataclass
class WorkspaceAiReqBody:
    appCode: str  # 应用code，必填
    pageNo: int  # 页码，必填
    pageSize: int  # 数量，必填

    empIds: Optional[List[str]] = None  # 员工工号，非必填
    endTime: Optional[str] = None  # 结束时间，非必填
    startTime: Optional[str] = None  # 开始时间，格式：yyyy-MM-dd HH:mm:ss，非必填
    messageIds: Optional[List[str]] = None  # 一个对话的消息id或者traceId，非必填
    source: Optional[List[str]] = None  # 来源（DEBUG：调试，API：API接口，WEB：分享页面，IDEAS：IDEAs分享页面，DING：钉钉机器人），非必填
    sessionId: Optional[str] = None  # 会话记录id，非必填
    orderBy: str = "DESC"  # 排序，非必填，默认倒序 ASC:升序 DESC:倒序


def workspace_ai_request(appCode,appVersion):
    url = f"https://aistudio.alibaba-inc.com/api/aiapp/run/{appCode}/{appVersion}"
    header = {
        "X-AK":aistudio_ak
    }
    WorkspaceAiReqBody