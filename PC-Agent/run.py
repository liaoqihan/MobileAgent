import os
import time
import copy
import torch
import shutil
import argparse
from PIL import Image, ImageDraw
from retrying import retry

from utils.oss import upload_oos_file
from utils.xl import BizNodeResult, UploadChatResultRequest, ai_agent_rec_bug, upload_chat_result,scene_appCode_map
parser = argparse.ArgumentParser(description="PC Agent")
parser.add_argument('--instruction', type=str, default='default')
parser.add_argument('--icon_caption', type=int, default=1) # 0: w/o icon_caption
parser.add_argument('--location_info', type=str, default='center') # center or bbox or icon_centor; icon_center: only icon center
parser.add_argument('--use_som', type=int, default=1) # for action
parser.add_argument('--draw_text_box', type=int, default=1, help="whether to draw text boxes in som.")
parser.add_argument('--font_path', type=str, default="/System/Library/Fonts/Times.ttc")
parser.add_argument('--pc_type', type=str, default="mac") # windows or mac
parser.add_argument('--api_url', type=str, default="https://api.openai.com/v1/chat/completions", help="GPT-4o api url.")
parser.add_argument('--api_token', type=str, help="Your GPT-4o api token.")
parser.add_argument('--qwen_api', type=str, default='', help="Input your Qwen-VL api if icon_caption=1.")
parser.add_argument('--add_info', type=str, default='')
parser.add_argument('--disable_reflection', action='store_true')
parser.add_argument("--empId", type=str)
parser.add_argument("--api_key", type=str)
parser.add_argument("--caption_call_method", type=str)
parser.add_argument("--caption_model", type=str)
parser.add_argument("--aistudio_ak", type=str)
args = parser.parse_args()

os.environ['empId'] = args.empId
os.environ['api_key'] = args.api_key
os.environ['api_url'] = args.api_url
os.environ['aistudio_ak'] = args.aistudio_ak

from PCAgent.api import inference_chat
from PCAgent.text_localization import ocr
from PCAgent.icon_localization import det
from PCAgent.prompt import PriceValidateResp, get_action_prompt, get_price_prompt_cn, get_price_validate_json_prompt, get_price_validate_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt,price_validate_response_format
from PCAgent.chat import init_action_chat, init_reflect_chat, init_memory_chat, add_response, init_xl_chat

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dashscope import MultiModalConversation
import dashscope
import concurrent

from pynput.mouse import Button, Controller
import pyautogui
import pyperclip
from PCAgent.merge_strategy import merge_boxes_and_texts, merge_all_icon_boxes, merge_boxes_and_texts_new

import re
from PCAgent.util import print_execution_time

def contains_chinese(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    match = chinese_pattern.search(text)
    return match is not None


import random
from PIL import ImageFont

def cmyk_to_rgb(c, m, y, k):
    r = 255 * (1.0 - c / 255) * (1.0 - k / 255)
    g = 255 * (1.0 - m / 255) * (1.0 - k / 255)
    b = 255 * (1.0 - y / 255) * (1.0 - k / 255)
    return int(r), int(g), int(b)

def draw_coordinates_boxes_on_image(image_path, coordinates, output_image_path, font_path):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    total_boxes = len(coordinates)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
              range(total_boxes)]
    # padding = height * 0.005

    for i, coord in enumerate(coordinates):
        # color = generate_color_from_hsv_pil(i, total_boxes)
        c, m, y, k = colors[i]
        color = cmyk_to_rgb(c, m, y, k)
        # print(color)

        # coord[0] = coord[0] - padding
        # coord[1] = coord[1] - padding
        # coord[2] = coord[2] + padding
        # coord[3] = coord[3] + padding
        try:
            draw.rectangle(coord, outline=color, width=int(height * 0.0025))
        except Exception as e:
            print(f"draw.rectangle error: {e} i, coord:{(i, coord)}")

        font = ImageFont.truetype(font_path, int(height * 0.012))
        text_x = coord[0] + int(height * 0.0025)
        text_y = max(0, coord[1] - int(height * 0.013))
        draw.text((text_x, text_y), str(i + 1), fill=color, font=font)
    # image.show()
    image = image.convert('RGB')
    image.save(output_image_path)



caption_call_method = args.caption_call_method

if args.pc_type == "mac":
    ctrl_key = "command"
    search_key = ["command", "space"]
    ratio = 2
else:
    ctrl_key = "ctrl"
    search_key = ["win", "s"]
    ratio = 1
    args.font_path = "C:\Windows\Fonts\\times.ttf"

vl_model_version = 'gpt-4o'


def extract_x_y(action_string):
    numbers = re.findall(r'\d+', action_string)
    if len(numbers) >= 2:
        x = int(numbers[0])
        y = int(numbers[1])
        return x, y
    else:
        raise ValueError("The given string does not contain enough numbers.")
    
def get_screenshot(path='screenshot/screenshot.png'):
    screenshot = pyautogui.screenshot()
    screenshot.save(path)
    return

def open_app(name):
    print('Action: open %s' % name)
    pyautogui.keyDown(search_key[0])
    pyautogui.keyDown(search_key[1])
    pyautogui.keyUp(search_key[1])
    pyautogui.keyUp(search_key[0])
    if contains_chinese(name):
        pyperclip.copy(name)
        pyautogui.keyDown(ctrl_key)
        pyautogui.keyDown('v')
        pyautogui.keyUp('v')
        pyautogui.keyUp(ctrl_key)
    else:
        pyautogui.typewrite(name)
    time.sleep(1)
    pyautogui.press('enter')

def tap(x, y, count=1):
    x, y = x//ratio, y//ratio
    print('Action: click (%d, %d) %d times' % (x, y, count))
    mouse = Controller()
    pyautogui.moveTo(x,y)
    mouse.click(Button.left, count=count)
    return

def shortcut(key1, key2):
    if key1 == 'command' and args.pc_type != "mac":
        key1 = 'ctrl'
    print('Action: shortcut %s + %s' % (key1, key2))
    pyautogui.keyDown(key1)
    pyautogui.keyDown(key2)
    pyautogui.keyUp(key2)
    pyautogui.keyUp(key1)
    return

def presskey(key):
    print('Action: press %s' % key)
    pyautogui.press(key)

def tap_type_enter(x, y, text):
    x, y = x//ratio, y//ratio
    print('Action: click (%d, %d), enter %s and press Enter' % (x, y, text))
    pyautogui.click(x=x, y=y)
    if contains_chinese(text):
        pyperclip.copy(text)
        pyautogui.keyDown(ctrl_key)
        pyautogui.keyDown('v')
        pyautogui.keyUp('v')
        pyautogui.keyUp(ctrl_key)
    else:
        pyautogui.typewrite(text)
    time.sleep(1)
    pyautogui.press('enter')
    return

def delete_part_png(directory):
    files = os.listdir(directory)
    for file_name in files:
        if "_part_" in file_name:
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


def get_file_list(parent_dir,key,key2=""):
    file_list = []
    files = sorted(os.listdir(parent_dir))
    for file_name in files:
        if f"{key}" in file_name and f"{key2}" in file_name:
            file_path = os.path.join(parent_dir, file_name)
            if os.path.isfile(file_path):
                file_list.append(file_path)
    return file_list

####################################### Edit your Setting #########################################

if args.instruction != 'default':
    instruction = args.instruction
else:
    # Your default instruction
    instruction = "Create a new doc on Word, write a brief introduction of Alibaba, and save the document."
    # instruction = "Help me download the pdf version of the 'Mobile Agent v2' paper on Chrome."

# Your GPT-4o API URL
API_url = args.api_url

# Your GPT-4o API Token
token = args.api_token

# Choose between "api" and "local". api: use the qwen api. local: use the local qwen checkpoint

# Choose between "qwen-vl-plus" and "qwen-vl-max" if use api method. Choose between "qwen-vl-chat" and "qwen-vl-chat-int4" if use local method.
# caption_model = "qwen-vl-max"
caption_model = args.caption_model

# If you choose the api caption call method, input your Qwen api here
qwen_api = args.qwen_api

# You can add operational knowledge to help Agent operate more accurately.
if args.add_info == '':
    add_info = '''
    When searching in the browser, click on the search bar at the top.
    The input field in WeChat is near the send button.
    When downloading files in the browser, it's preferred to use keyboard shortcuts.
    '''
else:
    add_info = args.add_info

# Reflection Setting: If you want to improve the operating speed, you can disable the reflection agent. This may reduce the success rate.
# reflection_switch = True if not args.disable_reflection else False
reflection_switch = False

# Memory Setting: If you want to improve the operating speed, you can disable the memory unit. This may reduce the success rate.
memory_switch = False # default: False
###################################################################################################


def get_all_files_in_folder(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        file_list.append(file_name)
    return file_list


def draw_coordinates_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    point_size = 10
    for coord in coordinates:
        draw.ellipse((coord[0] - point_size, coord[1] - point_size, coord[0] + point_size, coord[1] + point_size), fill='red')
    output_image_path = './screenshot/output_image.png'
    image.save(output_image_path)
    return output_image_path


def draw_rectangles_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for coord in coordinates:
        draw.rectangle([coord[0], coord[1]], outline="red", width=2)
    output_image_path = './screenshot/output_image.png'
    image.save(output_image_path)
    return output_image_path


def crop(image, box, i):
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2-10 or y1 >= y2-10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(f"./temp/{i}.png")


def generate_local(tokenizer, model, image_file, query):
    query = tokenizer.from_list_format([
        {'image': image_file},
        {'text': query},
    ])
    # create a "dummy" attention mask to pass to the model
    encoded_query = tokenizer(query, return_tensors='pt')
    input_ids = encoded_query['input_ids']
    attention_mask = encoded_query['attention_mask']
    # dummy_attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response


def process_image(image, query):
    dashscope.api_key = qwen_api
    image = "file://" + image
    messages = [{
        'role': 'user',
        'content': [
            {
                'image': image
            },
            {
                'text': query
            },
        ]
    }]
        
    try:
        call_res = MultiModalConversation.call(model=caption_model, messages=messages)
        response = call_res['output']['choices'][0]['message']['content'][0]["text"]
    except Exception as e:
        print(f"process_image  e:{e}")
        response = "An icon."
    
    return response


# def process_image(image, query):
#     dashscope.api_key = qwen_api
#     image = "file://" + image
#     messages = [{
#         'role': 'user',
#         'content': [
#             {
#                 'image': image
#             },
#             {
#                 'text': query
#             },
#         ]
#     }]
#     @retry(stop_max_attempt_number=5, wait_fixed=1000)
#     def call_api():
#         response = MultiModalConversation.call(model=caption_model, messages=messages)
#         try:
#             response = response['output']['choices'][0]['message']['content'][0]["text"]
#             return response
#         except Exception as e:
#             print(f"response:{response} e:{e}")
#             raise Exception(f"response:{response} e:{e}")

#     try:
#         response = call_api()
#         return response
#     except Exception as e:
#         return "An icon."



def generate_api(images, query):
    icon_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image, query): i for i, image in enumerate(images)}
        
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            response = future.result()
            icon_map[i + 1] = response
    
    return icon_map


def split_image_into_4(input_image_path, output_image_prefix):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 2
    sub_height = height // 2

    # crop into 4 sub images
    quadrants = [
        (0, 0, sub_width, sub_height),
        (sub_width, 0, width, sub_height),
        (0, sub_height, sub_width, height),
        (sub_width, sub_height, width, height)
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

def ocr_parallel(img, ocr_detection, ocr_recognition, img_x_list, img_y_list, padding, i):
    width, height = Image.open(img).size
    sub_text, sub_coordinates = ocr(img, ocr_detection, ocr_recognition)
    for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(width*2, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(height*2,img_y_list[i] + coordinate[3] + padding))
    sub_text_merge, sub_coordinates_merge = merge_boxes_and_texts_new(sub_text, sub_coordinates)
    print('parallel end')
    return sub_text_merge, sub_coordinates_merge

def icon_parallel(img, det, img_x_list, img_y_list, padding, i):
    width, height = Image.open(img).size
    sub_coordinates = det(img, "icon", groundingdino_model)
    for coordinate in sub_coordinates:
        coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
        coordinate[2] = int(min(width*2, img_x_list[i] + coordinate[2] + padding))
        coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
        coordinate[3] = int(min(height*2, img_y_list[i] + coordinate[3] + padding))
    sub_coordinates = merge_all_icon_boxes(sub_coordinates)
    return sub_coordinates


@print_execution_time
def get_perception_infos(screenshot_file, screenshot_som_file, font_path):
    get_screenshot(path=screenshot_file)
    
    total_width, total_height = Image.open(screenshot_file).size
    part_path = screenshot_file.replace(".png","")

    # Partition Image into 4 parts
    split_image_into_4(screenshot_file, part_path)
    img_list = [f"{part_path}_part_{i+1}.png" for i in range(4)]
    img_x_list = [0, total_width / 2, 0, total_width / 2]
    img_y_list = [0, 0, total_height / 2, total_height / 2]
    coordinates = []
    texts = []
    padding = total_height * 0.0025  # 10

    for i, img in enumerate(img_list):
        width, height = Image.open(img).size

        sub_text, sub_coordinates = ocr(img, ocr_detection, ocr_recognition)
        for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(total_height, img_y_list[i] + coordinate[3] + padding))

        sub_text_merge, sub_coordinates_merge = merge_boxes_and_texts_new(sub_text, sub_coordinates)
        coordinates.extend(sub_coordinates_merge)
        texts.extend(sub_text_merge)
    merged_text, merged_text_coordinates = merge_boxes_and_texts(texts, coordinates)

    coordinates = []
    for i, img in enumerate(img_list):
        width, height = Image.open(img).size
        sub_coordinates = det(img, "icon", groundingdino_model)
        for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(total_height, img_y_list[i] + coordinate[3] + padding))

        sub_coordinates = merge_all_icon_boxes(sub_coordinates)
        coordinates.extend(sub_coordinates)
    merged_icon_coordinates = merge_all_icon_boxes(coordinates)

    if args.draw_text_box == 1:
        rec_list = merged_text_coordinates + merged_icon_coordinates
        draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(rec_list), screenshot_som_file, font_path)
    else:
        draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(merged_icon_coordinates), screenshot_som_file, font_path)

    mark_number = 0
    perception_infos = []

    for i in range(len(merged_text_coordinates)):
        if args.use_som == 1 and args.draw_text_box == 1:
            mark_number += 1
            perception_info = {"text": "mark number: " + str(mark_number) + " text: " + merged_text[i], "coordinates": merged_text_coordinates[i]}
        else:
            perception_info = {"text": "text: " + merged_text[i], "coordinates": merged_text_coordinates[i]}
        perception_infos.append(perception_info)

    for i in range(len(merged_icon_coordinates)):
        if args.use_som == 1:
            mark_number += 1
            perception_info = {"text": "mark number: " + str(mark_number) + " icon", "coordinates": merged_icon_coordinates[i]}
        else:
            perception_info = {"text": "icon", "coordinates": merged_icon_coordinates[i]}
        perception_infos.append(perception_info)
    
    if args.icon_caption == 1:
        image_box = []
        image_id = []
        for i in range(len(perception_infos)):
            # if perception_infos[i]['text'] == 'icon':
            if 'icon' in perception_infos[i]['text']: # TODO
                image_box.append(perception_infos[i]['coordinates'])
                image_id.append(i)

        for i in range(len(image_box)):
            crop(screenshot_file, image_box[i], image_id[i])

        images = get_all_files_in_folder(temp_file)
        if len(images) > 0:
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
            icon_map = {}
            prompt = 'This image is an icon from a computer screen. Please briefly describe the shape and color of this icon in one sentence.'
            if caption_call_method == "local":
                for i in range(len(images)):
                    image_path = os.path.join(temp_file, images[i])
                    icon_width, icon_height = Image.open(image_path).size
                    if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                        des = "None"
                    else:
                        des = generate_local(tokenizer, model, image_path, prompt)
                    icon_map[i + 1] = des
            else:
                for i in range(len(images)):
                    images[i] = os.path.join(temp_file, images[i])
                icon_map = generate_api(images, prompt)
            for i, j in zip(image_id, range(1, len(image_id) + 1)):
                if icon_map.get(j):
                    perception_infos[i]['text'] += ": " + icon_map[j]

    if args.location_info == 'center':
        for i in range(len(perception_infos)):
            perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0] + perception_infos[i]['coordinates'][2]) / 2), int((perception_infos[i]['coordinates'][1] + perception_infos[i]['coordinates'][3]) / 2)]
    elif args.location_info == 'icon_center':
        for i in range(len(perception_infos)):
            if 'icon' in perception_infos[i]['text']:
                perception_infos[i]['coordinates'] = [
                    int((perception_infos[i]['coordinates'][0] + perception_infos[i]['coordinates'][2]) / 2),
                    int((perception_infos[i]['coordinates'][1] + perception_infos[i]['coordinates'][3]) / 2)]
    # 清空四分图
    for img in img_list:
        os.remove(img)
    return perception_infos, total_width, total_height

### Load caption model ###
device = "mps"
torch.manual_seed(1234)
local_cache_dir = "./model_cache"

if not os.path.exists(local_cache_dir):
    os.makedirs(local_cache_dir)

if caption_call_method == "local":
    if caption_model == "qwen-vl-chat":
        model_dir = os.path.join(local_cache_dir, "Qwen-VL-Chat")
        if not os.path.exists(model_dir):
            print("Downloading Qwen-VL-Chat model...")
            model_dir = snapshot_download('qwen/Qwen-VL-Chat', revision='v1.1.0', cache_dir=local_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    elif caption_model == "qwen-vl-chat-int4":
        model_dir = os.path.join(local_cache_dir, "Qwen-VL-Chat-Int4")
        if not os.path.exists(model_dir):
            print("Downloading Qwen-VL-Chat-Int4 model...")
            model_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision='v1.0.0', cache_dir=local_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True, use_safetensors=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True, do_sample=False)
    else:
        print("If you choose local caption method, you must choose the caption model from \"Qwen-vl-chat\" and \"Qwen-vl-chat-int4\"")
        exit(0)
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

elif caption_call_method == "api":
    pass
else:
    print("You must choose the caption model call function from \"local\" and \"api\"")
    exit(0)


### Load ocr and icon detection model ###
# groundingdino_dir = os.path.join(local_cache_dir, "AI-ModelScope")
# if not os.path.exists(groundingdino_dir):
#     print("Downloading AI-ModelScope/GroundingDINO model...")
#     groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0', cache_dir=local_cache_dir)
# groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')


thought_history = []
summary_history = []
action_history = []
reflection_thought = ""
summary = ""
action = ""
completed_requirements = ""
memory = ""
insight = ""
temp_file = "temp"
screenshot = "screenshot"

if os.path.exists(temp_file):
    shutil.rmtree(temp_file)
os.mkdir(temp_file)
if not os.path.exists(screenshot):
    os.mkdir(screenshot)
error_flag = False


import os
import time
import shutil
import json
import copy
from PIL import Image, ImageDraw, ImageFont

def mark_coordinate_on_image(image_path, coordinates, action,output_path=None):
    # 打开图像
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        # 设置字体
        try:
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 50)
        except IOError:
            font = ImageFont.load_default()
        
        # 画出坐标点
        for coord in coordinates:
            x, y = coord
            # 画一个红色的十字，标记坐标
            cross_size = 2000
            # 水平线
            draw.line((x - cross_size, y, x + cross_size, y), fill='red', width=5)
            # 垂直线
            draw.line((x, y - cross_size, x, y + cross_size), fill='red', width=5)
            # 在坐标旁边写出坐标值
            draw.text((x + 10, y + 10), action, fill='red', font=font)

            # 画一个红色的圆圈，标记坐标
            circle_radius = 30
            left_up_point = (x - circle_radius, y - circle_radius)
            right_down_point = (x + circle_radius, y + circle_radius)
            draw.ellipse([left_up_point, right_down_point], outline='red', width=5)
        # 保存新的图像
        if output_path is None:
            output_path = image_path.replace("screenshot",f"screenshot_{action}")
        img.save(output_path)

# Function to create unique log folder
def create_unique_log_folder(base_folder,instruction=""):
    timestamp = instruction + "_" + time.strftime("%Y%m%d%H%M%S")
    log_folder = os.path.join(base_folder, timestamp)
    os.makedirs(log_folder, exist_ok=True)
    return log_folder

# Function to save response to log file
# def save_response_to_log(log_folder, iter, prompt, response, screenshot_file, screenshot_som_file=None):
#     log_file = os.path.join(log_folder, f"gpt_log_{iter}.json")
#     data = {
#         "iteration": iter,
#         "prompt": prompt,
#         "response": response,
#         "screenshot_file": screenshot_file,
#         "screenshot_som_file": screenshot_som_file
#     }
#     with open(log_file, "w") as f:
#         json.dump(data, f, indent=4)

def save_response_to_log(log_folder, iter, prompt, response,type="",images=[],log_name=None):
    if not log_name:
        log_name = f"iter_{iter}_{type}_gpt_log.txt"
    log_file = os.path.join(log_folder, log_name)
    split_str = "="*30
    content = (
        f"{split_str}Iteration{split_str}\n{iter}\n"
        f"{split_str}Prompt{split_str}\n{prompt}\n"
        f"{split_str}Response{split_str}\n{response}\n"
        f"{split_str}images{split_str}\n{images}\n"
    )

    # Write the content to the log file
    with open(log_file, "w") as f:
        f.write(content)



# Create base folders if not already present
# base_temp_folder = "temp"

# if os.path.exists(base_temp_folder):
#     shutil.rmtree(base_temp_folder)
# os.mkdir(base_temp_folder)
base_log_folder = "log"
if not os.path.exists(base_log_folder):
    os.mkdir(base_log_folder)

# Create a unique log folder
log_folder = create_unique_log_folder(base_log_folder,instruction=instruction)

iter = 1
error_flag = False

start = time.time()
screenshot_file = f"{log_folder}/iter_{iter}_screenshot.png"
screenshot_som_file = f"{log_folder}/iter_{iter}_screenshot_som.png"

perception_infos, width, height = get_perception_infos(screenshot_file, screenshot_som_file, font_path=args.font_path)
shutil.rmtree(temp_file)
os.mkdir(temp_file)

need_stop = False
summary_list = []
action_list = []
thought_list = []
# chat_action = init_action_chat()
while True:
    
    prompt_action = get_action_prompt(instruction, perception_infos, width, height, thought_history, summary_history, action_history, summary, action, reflection_thought, add_info, error_flag, completed_requirements, memory, args.use_som, args.icon_caption, args.location_info)
    chat_action = init_action_chat()
    
    if args.use_som == 1:
        images = [screenshot_file, screenshot_som_file]
    else:
        images = [screenshot_file]
    chat_action = add_response("user", prompt_action, chat_action, images)
    output_action = inference_chat(chat_action, vl_model_version, API_url, token)
    
    save_response_to_log(log_folder, iter, prompt_action, output_action,images=images)
    thought = output_action.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":", "").replace("  ", " ").strip()
    thought_list.append(thought)
    summary = output_action.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
    action = output_action.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace("  ", " ").strip()
    chat_action = add_response("assistant", output_action, chat_action)
    status = "#" * 50 + " Decision " + "#" * 50
    print(status)
    print(output_action)
    print('#' * len(status))


    # debug
    time.sleep(5)
    x_mark,y_mark = None,None
    if "marked with number" in thought:
        pattern = r'coordinates \[(\d+), (\d+)\]'
        matches = re.search(pattern, thought)
        if matches: 
            print(f"优先使用标注过的元素的坐标 {x_mark},{y_mark}")
            x_mark,y_mark = int(matches.group(1)),int(matches.group(2))

    if "Double Tap" in action:
        coordinate = action.split("(")[-1].split(")")[0].split(", ")
        x, y = int(coordinate[0]), int(coordinate[1])
        if x_mark and y_mark:
            if x!=x_mark or y!=y_mark:
                x, y = x_mark, y_mark
                print(f"gpt又一次不使用标注过的元素的坐标 {x},{y}  {x_mark},{y_mark}")
        tap(x, y, 2)

    elif "Triple Tap" in action:
        coordinate = action.split("(")[-1].split(")")[0].split(", ")
        x, y = int(coordinate[0]), int(coordinate[1])
        if x_mark and y_mark:
            if x!=x_mark or y!=y_mark:
                x, y = x_mark, y_mark
                print(f"gpt又一次不使用标注过的元素的坐标 {x},{y}  {x_mark},{y_mark}")
        tap(x, y, 3)

    elif "Tap" in action:
        # coordinate = action.split("(")[-1].split(")")[0].split(", ")
        coordinate = action.split(")")[0].split("(")[-1].split(",")
        x, y = int(coordinate[0]), int(coordinate[-1])
        if x_mark and y_mark:
            if x!=x_mark or y!=y_mark:
                x, y = x_mark, y_mark
                print(f"gpt又一次不使用标注过的元素的坐标 {x},{y}  {x_mark},{y_mark}")
        tap(x, y, 1)

    elif "Shortcut" in action:
        keys = action.split("(")[-1].split(")")[0].split(", ")
        key1, key2 = keys[0].lower(), keys[1].lower()
        shortcut(key1, key2)
    
    elif "Press" in action:
        key = action.split("(")[-1].split(")")[0]
        presskey(key)

    elif "Open App" in action:
        app = action.split("(")[-1].split(")")[0]
        open_app(app)

    elif "Type" in action:
        try:
            coordinate = action.split("(")[1].split(")")[0].split(", ")
            x, y = int(coordinate[0]), int(coordinate[1])
        except:
            x, y = extract_x_y(action)
        if x_mark and y_mark:
            if x!=x_mark or y!=y_mark:
                x, y = x_mark, y_mark
                print(f"gpt又一次不使用标注过的元素的坐标 {x},{y}  {x_mark},{y_mark}")
        if "[text]" not in action:
            text = action.split("[")[-1].split("]")[0]
        else:
            text = action.split(" \"")[-1].split("\"")[0]

        tap_type_enter(x, y, text)
        
    elif "Stop" in action:
        need_stop = True
    if "Type" in action or "Tap" in action:
        # mark_coordinate_on_image(screenshot_som_file,[(x,y)],action)
        mark_coordinate_on_image(screenshot_som_file,[(x,y)],action,screenshot_som_file)
    print(f"{action} 已完成")


    summary_list.append(summary)
    action_list.append(action)
    # upload_som_img_path_list = get_file_list(parent_dir=log_folder,key=f"iter_{iter}_screenshot_som.png")
    # upload_som_img_url_list = upload_oos_file(upload_som_img_path_list)

    # upload_origin_img_path_list = get_file_list(parent_dir=log_folder,key=f"iter_{iter}_screenshot.png")
    # upload_origin_img_url_list = upload_oos_file(upload_som_img_path_list)

    # all_origin_images_url.append(upload_som_img_url_list[0])
    # all_som_images_url.append(upload_origin_img_url_list[0])
    # bizNodeResult = BizNodeResult(
    #     description = summary,
    #     actionName = action,
    #     imgList = upload_som_img_url_list,
    #     bizDesc=thought
    # )
    # bizNodeResultList.append(bizNodeResult)

    if need_stop:
        break

    if memory_switch:
        prompt_memory = get_memory_prompt(insight)
        chat_action = add_response("user", prompt_memory, chat_action)
        output_memory = inference_chat(chat_action, vl_model_version, API_url, token)
        save_response_to_log(log_folder, iter, prompt_memory, output_action,type="memory")
        chat_action = add_response("assistant", output_memory, chat_action)
        status = "#" * 50 + " Memory " + "#" * 50
        print(status)
        print(output_memory)
        print('#' * len(status))
        output_memory = output_memory.split("### Important content ###")[-1].split("\n\n")[0].strip() + "\n"
        if "None" not in output_memory and output_memory not in memory:
            memory += output_memory
    
    last_perception_infos = copy.deepcopy(perception_infos)
    # last_screenshot_file = f"{log_folder}/iter_{iter}_last_screenshot.png"
    # if os.path.exists(last_screenshot_file):
    #     os.remove(last_screenshot_file)
    # os.rename(screenshot_file, last_screenshot_file)
    # if args.use_som == 1:
    #     last_screenshot_som_file = f"{log_folder}/iter_{iter}_screenshot_som.png"
    #     if os.path.exists(last_screenshot_som_file):
    #         os.remove(last_screenshot_som_file)
    #     os.rename(screenshot_som_file, last_screenshot_som_file)
    
    new_timestamp = time.strftime("%Y%m%d%H%M%S")
    screenshot_after_file = f"{log_folder}/iter_{iter}_screenshot_after_{new_timestamp}.png"
    screenshot_som_after_file = f"{log_folder}/iter_{iter}_screenshot_som_after_{new_timestamp}.png"
    perception_infos, width, height = get_perception_infos(screenshot_after_file, screenshot_som_after_file, font_path=args.font_path)
    shutil.rmtree(temp_file)
    os.mkdir(temp_file)

    
    if reflection_switch:
        prompt_reflect = get_reflect_prompt(instruction, last_perception_infos, perception_infos, width, height, summary, action, add_info)
        chat_reflect = init_reflect_chat()
        chat_reflect = add_response("user", prompt_reflect, chat_reflect, [screenshot_file, screenshot_after_file])

        output_reflect = inference_chat(chat_reflect, vl_model_version, API_url, token)
        save_response_to_log(log_folder, iter, prompt_reflect, output_action,type="reflection",images=[screenshot_file, screenshot_after_file])
        reflection_thought = output_reflect.split("### Thought ###")[-1].split("### Answer ###")[0].replace("\n", " ").strip()
        reflect = output_reflect.split("### Answer ###")[-1].replace("\n", " ").strip()
        chat_reflect = add_response("assistant", output_reflect, chat_reflect)
        status = "#" * 50 + " Reflection " + "#" * 50
        print(status)
        print(output_reflect)
        print('#' * len(status))
    
        if 'A' in reflect:
            thought_history.append(thought)
            summary_history.append(summary)
            action_history.append(action)
            
            prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info)
            chat_planning = init_memory_chat()
            chat_planning = add_response("user", prompt_planning, chat_planning)
            output_planning = inference_chat(chat_planning, 'gpt-4o', API_url, token)
            save_response_to_log(log_folder, iter, prompt_planning, output_action,type="A_reflection")
            chat_planning = add_response("assistant", output_planning, chat_planning)
            status = "#" * 50 + " Planning " + "#" * 50
            print(status)
            print(output_planning)
            print('#' * len(status))
            completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
            
            error_flag = False
        
        elif 'B' in reflect:
            error_flag = True
            # presskey('esc')
            
        elif 'C' in reflect:
            error_flag = True
            # presskey('esc')
    
    else:
        thought_history.append(thought)
        summary_history.append(summary)
        action_history.append(action)
        
        prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info)
        chat_planning = init_memory_chat()
        chat_planning = add_response("user", prompt_planning, chat_planning)
        output_planning = inference_chat(chat_planning, 'gpt-4o', API_url, token)
        save_response_to_log(log_folder, iter, prompt_planning, output_planning,type="else_reflection")
        chat_planning = add_response("assistant", output_planning, chat_planning)
        status = "#" * 50 + " Planning " + "#" * 50
        print(status)
        print(output_planning)
        print('#' * len(status))
        completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
         
    # os.remove(last_screenshot_file)
    # if args.use_som == 1:
    #     os.remove(last_screenshot_som_file)


    iter += 1
    screenshot_file = f"{log_folder}/iter_{iter}_screenshot.png"
    screenshot_som_file = f"{log_folder}/iter_{iter}_screenshot_som.png"

    os.rename(screenshot_after_file, screenshot_file)
    if args.use_som == 1:
        os.rename(screenshot_som_after_file, screenshot_som_file)

upload_som_img_path_list = get_file_list(parent_dir=log_folder,key=f"_screenshot_som.png")
upload_som_img_url_list = upload_oos_file(upload_som_img_path_list)

upload_origin_img_path_list = get_file_list(parent_dir=log_folder,key=f"_screenshot.png")
upload_origin_img_url_list = upload_oos_file(upload_origin_img_path_list)


all_origin_images = get_file_list(parent_dir=log_folder,key="_screenshot.png")
all_som_images = get_file_list(parent_dir=log_folder,key="_screenshot_som.png")
req_images = [*upload_origin_img_path_list,*upload_som_img_path_list]
req_images_url = [*upload_origin_img_url_list,*upload_som_img_url_list]
price_validate_prompt = get_price_prompt_cn(img_num=iter)

# 框架ai方式
# xl_chat = init_xl_chat(task=instruction,width=width,height=height,image_num=iter)
# prompt_reqbody =  add_response("user", price_validate_prompt, xl_chat,all_origin_images)
# res = inference_chat(prompt_reqbody, 'gpt-4o', API_url, token,response_format=price_validate_response_format)

#ai studio方式

bizNodeResultList = []
for i in range(iter):
    bizNodeResult = BizNodeResult(
        description = summary_list[i],
        actionName = action_list[i],
        imgList = [upload_som_img_url_list[i]],
        bizDesc=thought
    )
    bizNodeResultList.append(bizNodeResult)

variableMap = {
    'task': instruction,
    'width': width,
    'height': height,
    'image_num': iter
}


# 通用体验问题校验
for i,url in enumerate(upload_origin_img_url_list):
    validate_str = "#######\n\n"

    res = ai_agent_rec_bug(image_urls=url)
    has_issue = "没有体验问题" not in res
    validate_str += f"是否有其他问题: {has_issue}\n\n"
    validate_str += f"{res}\n"
    bizNodeResultList[i].driverAssert = validate_str
    debug_bizNodeResultList = copy.deepcopy(bizNodeResultList)


# 
def debug():
    res = ai_agent_rec_bug(image_urls=req_images_url,variableMap=variableMap,scene="价格一致性",prompt=price_validate_prompt)
    try:
        res_dict = json.loads(res)
    except Exception as e:
        res = res.replace("```json\n","").replace("\n```","")
        res_dict = json.loads(res)
    has_price_issue = res_dict.get("has_price_issue")
    thought = res_dict.get("thought")

    price_validate_str = "#######\n\n"
    price_validate_str += f"是否有价格一致性问题: {has_price_issue}\n\n"
    price_validate_str += f"分析:\n{thought}"
    price_validate_str += "#######\n"

    # index = int(answer)-1
    index = -1 # 不管有没有问题 分析都放在最后一张
    debug_driverAssert = '#######\n\n是否有其他问题: False\n\n问题详情：\n1. 本地化币种\n   - 当前页面显示的价格为 "$203.00", "$201.00", "$192.00" 和 "$188.00"，这些价格符号和格式是符合美国地区的 USD 格式，没有问题。\n2. 本地化数字\n   - 数字格式是符合美国地区的格式的，没有问题。\n3. 页面显示语言\n   - 页面显示语言为英语，符合美国地区的语言习惯，没有问题。\n4. 推荐内容\n   - 页面推荐内容与产品相关，符合电商业务习惯，没有问题。\n5. 其他\n   - 页面没有白屏，没有漏翻问题。\n\n综上所述：\n没有体验问题。\n'
    bizNodeResultList[index].driverAssert = debug_driverAssert + price_validate_str #debug
    # uploadChatResultRequest.bizNodesResult[index].driverAssert +=  price_validate_str



    takeTime = time.time()-start
    takeTime = f"{takeTime:.2f}"
    uploadChatResultRequest = UploadChatResultRequest(
        instruction = instruction,
        bizNodesResult = bizNodeResultList,
        takeTime=takeTime
    )
    upload_chat_result_res = upload_chat_result(uploadChatResultRequest)
    return upload_chat_result_res,has_price_issue

upload_chat_result_res,has_price_issue = debug()

def debug_api(e_has_price_issue=True,num_tasks=5):
    count = 0
    upload_chat_result_url_list = [None] * num_tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
        futures = {executor.submit(debug): i for i in range(num_tasks)}
        
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                upload_chat_result_url_list[i], has_price_issue = future.result()
                if has_price_issue == e_has_price_issue:
                    count += 1
            except Exception as exc:
                print(f"Task {i} generated an exception: {exc}")
    suc = count/num_tasks
    return upload_chat_result_url_list,suc

upload_chat_result_url_list,suc = debug_api(num_tasks=1)


