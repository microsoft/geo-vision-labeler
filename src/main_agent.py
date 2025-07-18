import argparse
import glob
import os
import torch
import logging
import random
import numpy as np
import tqdm
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

from collections import Counter
from PIL import Image
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
from vision_agent import describe_image, classify_with_huggingface


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("labeling.log"), logging.StreamHandler()],
)

dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    logging.warning(
        "No .env file found. Make sure to set environment variables manually."
    )
logging.info("logining Hugging face...")
login(token=os.environ.get("HF_TOKEN", ""))

def create_directories():
    """创建保存结果的文件夹"""
    os.makedirs("data/labels_shuffled", exist_ok=True)
    os.makedirs("data/label_list", exist_ok=True)

def save_results_with_timestamp(image_labels_shuffled, label_list):
    """用时间戳保存结果到指定文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存真实标签
    labels_shuffled_file = f"data/labels_shuffled/labels_shuffled_{timestamp}.txt"
    with open(labels_shuffled_file, "w", encoding="utf-8") as f:
        for label in image_labels_shuffled:
            f.write(f"{label}\n")
    logging.info(f"真实标签已保存到: {labels_shuffled_file}")
    
    # 保存预测标签
    label_list_file = f"data/label_list/label_list_{timestamp}.txt"
    with open(label_list_file, "w", encoding="utf-8") as f:
        for label in label_list:
            f.write(f"{label}\n")
    logging.info(f"预测标签已保存到: {label_list_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Label images using a vision LLM and a text classifier.'
    )
    
    parser.add_argument(
        '--simple_dir',
        type=str,
        required=True,
        help='Directory with simple images'
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with images to label.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Directory to save labeled images.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Name of the classifier model.",
    )
    parser.add_argument(
        "--vision_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Vision model to use for image description.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="这是一张卫星遥感图像. ",
        help="Meta prompt to guide the vision or CLIP model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="用长文本描述图像内容: ",
        help="Prompt to guide the vision model.",
    )
    parser.add_argument(
        "--classes_file",
        type=str,
        default="data/simple_classes.txt",
        help="File containing the classes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run the Hugging Face models on.",
    )

    args = parser.parse_args()

    # 创建保存结果的文件夹
    create_directories()


    # Load simple images
    if not os.path.exists(args.simple_dir):
        raise FileNotFoundError(f"simple images dir {args.simple_dir} not found.")
    else:
        images_paths = glob.glob(os.path.join(args.simple_dir, "*"))
    if not images_paths:
        logging.info("No simple images found in the specified input directory.")
        return
    
    # Load classes
    if not os.path.exists(args.classes_file):
        raise FileNotFoundError(f"Classes file {args.classes_file} not found.")
    else:
        with open(args.classes_file, "r") as f:
            classes_flat = [line.strip() for line in f.readlines()]
    if not classes_flat:
        raise ValueError("No classes found in the classes file.")
    
    # Load input images
    categories = ["城市区域", "目标区域"]
    # ------------------------------
    # 1. 读取图片名和对应的标签
    # ------------------------------
    image_paths = []
    image_labels = []
    for label in categories:
        folder_path = os.path.join(args.input_dir, label)
        for img_name in os.listdir(folder_path):
            if img_name.endswith(".png"):  # 仅处理png图片
                img_path = os.path.join(folder_path, img_name)
                image_paths.append(img_path)
                image_labels.append(label)
    label_counts = Counter(image_labels)
    print("Before shuffling:", label_counts)

    # ------------------------------
    # 2. 以相同随机顺序打乱两个列表
    # ------------------------------
    paired = list(zip(image_paths, image_labels))
    random.shuffle(paired)
    image_paths_shuffled, image_labels_shuffled = zip(*paired)
    # 转换为列表（zip返回的是元组）
    image_paths_shuffled = list(image_paths_shuffled)
    image_labels_shuffled = list(image_labels_shuffled)

    # Initialize models
    processor = AutoProcessor.from_pretrained(args.vision_model)
    model = AutoModelForVision2Seq.from_pretrained(args.vision_model).to(args.device)

    
    # Describe images
    desc_list = []
    for i, images in enumerate(images_paths):
        image = Image.open(images)
        desc = describe_image(
            image = image, 
            context = args.context,
            prompt = args.prompt,
            classes = classes_flat[i],
            include_classes = True,
            processor = processor, 
            model = model,
            )
        desc_list.append([desc])    

    # Initialize classifier pipeline
    gen_pipeline = None
    tokenizer = AutoTokenizer.from_pretrained(args.classifier)
    llm_model = AutoModelForCausalLM.from_pretrained(args.classifier).to(args.device)
    
    
    gen_pipeline = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        device=args.device,
    )

    label_list = []
    # 生成描述
    for input_images in tqdm.tqdm(image_paths_shuffled):
        images = Image.open(input_images)
        description = describe_image(
            image=images,
            context=args.context,
            prompt=args.prompt,
            classes=classes_flat,
            include_classes=False,
            processor=processor,
            model=model,
        )
    
        # 处理分类
        label = classify_with_huggingface(
            description=description,
            desc_list=desc_list,
            class_list=classes_flat,
            pipeline_instance=gen_pipeline,
            tokenizer=tokenizer,
            max_new_tokens=28,
        )
        label_list.append(label)
    
    #print(label_list)    
    with open(args.output_file, "w", encoding="utf-8") as file:
        file.writelines(f"{item}\n" for item in desc_list)
    
    print(f"End!\n")

if __name__ == '__main__':
    main()
    