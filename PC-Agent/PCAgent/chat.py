import copy
from PCAgent.api import encode_image


def init_action_chat(in_cn=True):
    operation_history = []
    sysetm_prompt = "You are a helpful AI PC operating assistant. You need to help me operate the PC to complete the user\'s instruction."
    if in_cn:
        sysetm_prompt += "All your response should be in Chinese."
    operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
    return operation_history


def init_reflect_chat():
    operation_history = []
    sysetm_prompt = "You are a helpful AI PC operating assistant."
    operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
    return operation_history


def init_memory_chat():
    operation_history = []
    sysetm_prompt = "You are a helpful AI PC operating assistant."
    operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
    return operation_history


def init_xl_chat(task,width,height,image_num,in_json=False,in_cn=True):
    operation_history = []
    # sysetm_prompt = "You are a helpful AI test engineer for a B2B ecommerce platform named 'Alibaba.com'."
    # if in_json:
    #     sysetm_prompt +="Your all responses will be in JSON format."
    # if in_cn:
    #     sysetm_prompt += "All your response should be in Chinese."
    sysetm_prompt = f"""
您是一个专业的测试工程师 负责测试一个名为'Alibaba.com'的B2B跨境电商平台.
用户刚刚在这个平台上执行完了一个任务,任务名称是'{task}'.用户每次提问都会向您提供{2*image_num}张截图.前{image_num}张是执行过程中每一个步骤执行完之后的截图,后{image_num}张为是在前{image_num}张的基础上进行标注的,所谓标注就是将截图上的icon和文本都框起来且在框的旁边有对应的数字.提供后{image_num}张的目的是为了您能更好的识别图像。每张截图的宽都是{width} pixels,高都是{height} pixels.
请您帮助用户:基于执行步骤的截图，识别此操作链中可能存在的问题。请特别关注“价格一致性问题”。价格一致性问题是指同一产品的原价包括折扣价在不同页面上应保持一致的要求。
接下来我会提供几个示例供您参考,如果遇到例子中未提到的情况,您需要自行判断.示例:
    属于价格一致性问题的情况：
            1. 价格显示不一致：同一产品在一个页面有划线线价格显示，但在另一个页面没有;同一产品在一个页面有价格显示，但在另一个页面没有;同一产品在一个页面的原价或划线价价格显示与其他页面不同。
            2. 价格表达不一致:同一产品在一个页面的价格显示为1,000.00,而在另一个页面显示为1000.00;同一产品在一个页面有货币符号，但在另一个页面没有（导致价格表示不一致）
            3. 多价格显示不一致：同一产品在商品详情页基于购买数量显示多个价格，但在其他页面只显示一个价格，而且这个价格并不是商品详情页中最低购买数量对应的价格。
        不属于价格一致性问题的情况：
            1. 一致的最低价格显示：同一产品在商品详情页基于购买数量显示多个价格，在其他页面只显示一个价格，而这个价格是商品详情页中最低购买数量对应的价格

请您用中文回答,并以Json字符串提供输出（无需markdown格式）。
"""
    if in_json:
        sysetm_prompt += "\n\nJson有两个键值对，第一个是Thought，其值是您关于价格一致性问题的分析（您可以按照以下步骤来思考回答: 步骤1-详细分析用户提供给您的每一张截图,提取出商品的价格表达。步骤2-先判断题去的商品价格表达的情况属不属于上面提供的例子，步骤三-如果不属于，你再自己综合判断，如果属于，那么请详细说明。）。总之，如果你识别到任何价格一致性问题，需要在此详细解释，第二个键值对是Answer，其值是一个数字或'无，它代表您在哪个图片上检测出了有问题"
    operation_history.append(["system", [{"type": "text", "text": sysetm_prompt}]])
    return operation_history


def add_response_old(role, prompt, chat_history, image=None):

    new_chat_history = copy.deepcopy(chat_history)
    if image:
        base64_image = encode_image(image)
        content = [
            {
                "type": "text", 
                "text": prompt
            },
            {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            },
        ]
    else:
        content = [
            {
            "type": "text", 
            "text": prompt
            },
        ]
    new_chat_history.append([role, content])
    return new_chat_history


def add_response(role, prompt, chat_history, image=[]):
    new_chat_history = copy.deepcopy(chat_history)
    content = [
        {
        "type": "text", 
        "text": prompt
        },
    ]
    for i in range(len(image)):
        base64_image = encode_image(image[i])
        content.append(
            {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        )
    new_chat_history.append([role, content])
    return new_chat_history


def add_response_two_image(role, prompt, chat_history, image):
    new_chat_history = copy.deepcopy(chat_history)

    base64_image1 = encode_image(image[0])
    base64_image2 = encode_image(image[1])
    content = [
        {
            "type": "text", 
            "text": prompt
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image1}"
            }
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image2}"
            }
        },
    ]

    new_chat_history.append([role, content])
    return new_chat_history


def print_status(chat_history):
    print("*"*100)
    for chat in chat_history:
        print("role:", chat[0])
        print(chat[1][0]["text"] + "<image>"*(len(chat[1])-1) + "\n")
    print("*"*100)