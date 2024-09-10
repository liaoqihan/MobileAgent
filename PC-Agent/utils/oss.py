from dataclasses import dataclass, field, asdict
from typing import List, Optional
import uuid
import requests
import json
from dataclasses import dataclass, asdict


def upload_oos_file(file_paths:list,return_files=True):

    url = "http://xl.alibaba-inc.com/open/api/uploadFile"

    files = [('file', open(file_path, 'rb')) for file_path in file_paths]
    response = requests.post(url, files=files)

    if response.status_code == 200 and response.json().get("success"):
        # 解析响应的 JSON 数据
        result = response.json()
        if return_files:
            return result.get("files")
        return result
    else:
        print("upload_oos_file Failed. Status code:", response.status_code)
        print("Response:", response.text)

if __name__ == '__main__':
    res = upload_oos_file("/Users/juexin/MobileAgent/log/首页有Top deals会场 一直点击第一个品直至进入商详页（这里商品会不断穿透）如果这个商品在各个页面出现了价格不一致 则Stop_20240909185032/iter_1_screenshot.png")
