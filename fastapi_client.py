import requests
import base64
import sys
from PIL import Image
import io

# API的URL
url = "http://10.144.129.134:8899/process-image/"

# 获取图像文件名
image_file = sys.argv[1]

# 读取图像并编码为base64
with open(image_file, "rb") as image:
    image_base64 = base64.b64encode(image.read()).decode()

# 请求体
payload = {
    "image_base64": image_base64,
    "box_threshold": 0.05,
    "iou_threshold": 0.1,
    "use_paddleocr": True,
    "imgsz": 1920
}

# 发送POST请求
response = requests.post(url, json=payload)

# 打印响应
if response.status_code == 200:
    print("成功:", response.json())
else:
    print("失败:", response.status_code, response.text)