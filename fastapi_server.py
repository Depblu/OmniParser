from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from PIL import Image
import io
from pydantic import BaseModel, Field
import base64



########################
from typing import Optional
import numpy as np
import torch
from PIL import Image
import io


import base64, os
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image

yolo_model_orig = get_yolo_model(model_path='weights/icon_detect/best.pt')
yolo_model = get_yolo_model(model_path='weights/icon_detect/yolo11x.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2")



MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda')

# @spaces.GPU
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz
) -> Optional[Image.Image]:

    image_save_path = 'imgs/saved_image_demo.png'
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    # import pdb; pdb.set_trace()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    
    # print('prompt:', prompt)
    dino_labled_img_orig, label_coordinates_orig, parsed_content_list_orig = get_som_labeled_img(image_save_path, yolo_model_orig, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz)  
    image_orig = Image.open(io.BytesIO(base64.b64decode(dino_labled_img_orig)))
    print('finish processing')
    #parsed_content_list_orig = '\n'.join(parsed_content_list_orig)

    # print('prompt:', prompt)
    
    # dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_save_path, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz)  
    # image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    # print('finish processing')
    
    #parsed_content_list = '\n'.join(parsed_content_list)
    
    image = image_orig
    label_coordinates = label_coordinates_orig
    parsed_content_list = parsed_content_list_orig
    
    return image, label_coordinates, parsed_content_list

########################




app = FastAPI(
    title="OmniParser API",
    description="使用 FastAPI 提供的 OmniParser RESTful API 服务",
    version="1.0.0"
)

class ResponseItem(BaseModel):
    item_id: str = Field(..., description="数值类型，表示解析框序号，从0开始")
    item_type: str = Field(..., description="字符串枚举，表示解析框类型， 取值范围为'text'或'icon'")
    info: str = Field(..., description="信息，表示解析框内的信息， 类型为字符串")
    cxcywh: list[int] = Field(..., description="包含4个数值型元素的list，标识一个box, 四个数值分别表示box的中心坐标xy和box的宽高wh")

class ProcessImageResponse(BaseModel):
    """
    解析图像的响应
    """
    image_with_bbox: str = Field(..., description="包含边界框的图像，类型为base64编码的字符串")
    parsed_content_list: list[ResponseItem] = Field(
        ...,
        description="解析后的内容列表，每个元素为ResponseItem类型"
    )



@app.post("/process-image/", response_model=ProcessImageResponse)
async def process_image(
    image_base64: str = Body(..., description="类型为字符串，表示图像的base64编码数据"),
    box_threshold: float = Body(0.05, description="类型为浮点数，表示用于过滤低置信度边界框的阈值"),
    iou_threshold: float = Body(0.1, description="类型为浮点数，表示用于过滤重叠边界框的阈值"),
    use_paddleocr: bool = Body(True, description="类型为布尔值，表示是否使用PaddleOCR进行OCR识别"),
    imgsz: int = Body(640, description="类型为整数，表示用于icon detection的图像尺寸")
):
    try:
        # 解码 base64 图像数据
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))

        # 调用 process 函数
        processed_image, label_coordinates, parsed_content_list = process(
            image_input=image,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz
        )
        '''
        label_coordinates:是一个字典，key是idx，取值"0", "1", "2", ... "n" 表示 parsed_content_list的下标。
        value是list，包含4个数值型元素，分别为左上x,y和右下x,y.注意，x,y的取值范围为[0,1]，分别表示图像宽度和高度的比例位置。
        parsed_content_list：是一个字符串列表，每个元素为解析后的内容。
        有两种前缀开头：
        1. "Text Box ID"， 表示文本框
        2. "Icon Box ID"， 表示图标框
        '''
        # 构造 parsed_content_list
        parsed_content_list_response = []
        for idx, content in enumerate(parsed_content_list):
            if content.startswith("Text Box ID"):
                item_type = "text"
            elif content.startswith("Icon Box ID"):
                item_type = "icon"
            else:
                continue

            cxcywh = label_coordinates[str(idx)]
            # 提取信息
            # 获取图像的宽度和高度
            image_width, image_height = image.size

            # 计算像素坐标
            center_x = int(cxcywh[0] * image_width)
            center_y = int(cxcywh[1] * image_height)
            width = int(cxcywh[2] * image_width)
            height = int(cxcywh[3] * image_height)

            # 更新cxcywh为像素坐标
            cxcywh = [center_x, center_y, width, height]
            
            info = content.split(f"Box ID {idx}: ")[1] if ": " in content else content

            parsed_content_list_response.append({
                "item_id": str(idx),
                "item_type": item_type,
                "info": info,
                "cxcywh": cxcywh
            })


        # 将图像转换为base64编码的字符串
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        image_with_bbox_base64 = base64.b64encode(buffered.getvalue()).decode()

        # 构造 ProcessImageResponse
        response = ProcessImageResponse(
            image_with_bbox=image_with_bbox_base64,
            parsed_content_list=parsed_content_list_response
        )

        return JSONResponse(content=response.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动命令: uvicorn fastapi_server:app --reload --port 8899