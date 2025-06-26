# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import csv
import glob
import logging
import os

import numpy as np
import torch
import tqdm
import rasterio
import yaml
from PIL import Image
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
import openai
from openai import AzureOpenAI

from vision import describe_image
from classifier import classify_with_openai, classify_with_huggingface
from clip import CLIPClassifier

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
logging.info("begin...")

login(token=os.environ.get("HF_TOKEN", ""))
logging.info("Logeding in to Hugging Face Hub.")
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-large-patch14")
logging.info("Loged in to Hugging Face Hub.")

def split_image(image_path, split_height_by, split_width_by):
    # unchanged...
    logging.info("Splitting image...")
    with rasterio.open(image_path) as src:
        image = src.read()
        if len(image.shape) == 3:
            image = np.moveaxis(image, 0, -1)
        else:
            raise ValueError("Image must have 3 dimensions (height, width, channels).")
        if image.shape[2] != 3:
            logging.warning(
                f"Image has {image.shape[2]} channels. Keeping the first 3 channels."
            )
            image = image[:, :, :3]

    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image, mode="RGB")
    if min(image.size) < 224:
        logging.warning(
            f"Image size {image.size} is smaller than 224. Resizing to 224x224."
        )
        image = image.resize((224, 224), Image.BILINEAR)

    width, height = image.size
    chunk_width = width // split_width_by
    chunk_height = height // split_height_by
    chunks = []
    coords_dict_list = []
    for i in range(split_height_by):
        for j in range(split_width_by):
            left = j * chunk_width
            lower = i * chunk_height
            right = left + chunk_width
            upper = lower + chunk_height
            coords = (left, lower, right, upper)
            chunk = image.crop(coords)
            chunks.append(chunk)
            coords_dict_list.append(
                {"row": i, "col": j, "left": left, "upper": upper, "right": right, "lower": lower}
            )
    logging.info(f"Split image into {len(chunks)} chunks.") # Save chunks to disk
    return chunks, coords_dict_list


def init_openai_client(variant):
    logging.info(f"Initializing OpenAI client for variant {variant}.") 
    if variant.lower() == "azure":
        client = AzureOpenAI(
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2023-03-15-preview"
            ),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        )
    else:
        openai.api_key = os.environ.get("OPENAI_API_KEY", "")
        client = openai
    return client


def get_hierarchy_depth(nested_dict):
    depth = 0
    if isinstance(nested_dict, dict):
        for v in nested_dict.values():
            depth = max(depth, 1 + get_hierarchy_depth(v))
    return depth

def fallback_label(label, classes, chunk, filename, coords, clip_classifier, args):
    """
    Fallback mechanism to handle cases where the label is not found in the current hierarchy.
    """
    if label not in classes:
        # Log a warning and try to find the closest match
        logging.warning(
            f"Label '{label}' not found in classes for file {filename}_r{coords['row']}_c{coords['col']}. "
            "Trying to find the closest match."
        )
        matched_classes = sorted(
            [
                (class_name, label.find(class_name))
                for class_name in classes
                if class_name in label
            ]
            + [
                (class_name, class_name.find(label))
                for class_name in classes
                if label in class_name
            ],
            key=lambda x: x[1],
        )

        if matched_classes:
            # Use the closest match
            label = matched_classes[0][0]
        else:
            # Fall back to CLIP classification
            logging.warning(
                f"No closest match for label '{label}' for file {filename}_r{coords['row']}_c{coords['col']}. "
                "Falling back to CLIP."
            )
            label, _ = clip_classifier.classify_image(
                image=chunk,
                classes=classes,
                context=args.context,
            )
    return label

def main():
    logging.info("Starting main function")
    parser = argparse.ArgumentParser(
        description="Label images using a vision LLM and a text classifier."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/labels.csv",
        help="CSV file to write the results.",
    )
    parser.add_argument(
        "--classes_file",
        type=str,
        default="data/classes.txt",
        help="File containing the classes or hierarchy for classification. Can be a text file with flat classes or a YAML file with hierarchical classes.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["flat", "hierarchical"],
        default="flat",
        help="Classification mode: flat or hierarchical.",
    )
    # existing args...
    parser.add_argument(
        "--vision_model",
        type=str,
        default="microsoft/kosmos-2-patch14-224",
        help="Vision model to use for image description.",
    )
    parser.add_argument(
        "--apply_vision_template",
        action="store_true",
        help="Use 'role'/'content' template for the vision model.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Name of the classifier model.",
    )
    parser.add_argument(
        "--classifier_type",
        type=str,
        choices=["openai", "huggingface", "clip"],
        default="huggingface",
        help="Classifier type: openai, huggingface, or clip.",
    )
    parser.add_argument(
        "--openai_variant",
        type=str,
        choices=["azure", "openai"],
        default="azure",
        help="For OpenAI classifier: choose Azure or OpenAI API.",
    )
    parser.add_argument(
        "--split_height_by",
        type=int,
        default=1,
        help="Number of vertical splits per image.",
    )
    parser.add_argument(
        "--split_width_by",
        type=int,
        default=1,
        help="Number of horizontal splits per image.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="This is a satellite image. ",
        help="Meta prompt to guide the vision or CLIP model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Detailed long description of the image: ",
        help="Prompt to guide the vision model.",
    )
    parser.add_argument(
        "--include_filename",
        action="store_true",
        help="Include the filename in the prompt for the vision/CLIP model.",
    )
    parser.add_argument(
        "--include_classes",
        action="store_true",
        help="Include classes in the vision model prompt.",
    )
    parser.add_argument(
        "--test_time_augmentation",
        type=list,
        default="",
        help="Test time augmentation strategies for rotation [x, y, xy].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run the Hugging Face models on.",
    )
    args = parser.parse_args()

    # Load classes or hierarchy
    if not os.path.exists(args.classes_file):
        raise FileNotFoundError(f"Classes file {args.classes_file} not found.")
    if (args.mode == "hierarchical") and "yaml" in args.classes_file:
        with open(args.classes_file, "r") as f:
            hierarchy = yaml.safe_load(f)
            classes_flat = list(hierarchy.keys())     
            depth = get_hierarchy_depth(hierarchy)
    else:
        with open(args.classes_file, "r") as f:
            classes_flat = [line.strip() for line in f.readlines()]
            hierarchy = None
        depth = 0

    logging.info(f"Detected hierarchy depth = {depth}")

    if not classes_flat:
        raise ValueError("No classes found in the classes file.")

    logging.info(f"Loaded {len(classes_flat)} top-level classes.")

    # Initialize models
    if args.classifier_type != "clip":
        processor = AutoProcessor.from_pretrained(args.vision_model)
        model = AutoModelForVision2Seq.from_pretrained(args.vision_model).to(
            args.device
        )
    clip_classifier = CLIPClassifier(
        model_name=CLIP_MODEL_NAME,
        device=args.device,
    )

    # Setup classifier pipelines
    client = None
    gen_pipeline = None
    if args.classifier_type == "openai":
        client = init_openai_client(args.openai_variant)
    elif args.classifier_type == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(args.classifier)
        llm_model = AutoModelForCausalLM.from_pretrained(args.classifier).to(
            args.device
        )
        gen_pipeline = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=tokenizer,
            device=args.device,
        )

    filepaths = glob.glob(os.path.join(args.input_dir, "*"))
    if not filepaths:
        logging.info("No images found in the specified input directory.")
        return

    results = []
    for filepath in tqdm.tqdm(filepaths):
        chunks, coords_list = split_image(
            filepath, args.split_height_by, args.split_width_by
        )
        for i, chunk in enumerate(chunks):
            filename = os.path.basename(filepath)
            coords = coords_list[i]
            desc = None

            # get description if not CLIP-only
            if args.classifier_type != "clip":
                desc = describe_image(
                    image_chunk=chunk,
                    image_path=filepath,
                    context=args.context,
                    prompt=args.prompt,
                    classes=classes_flat,
                    include_classes=args.include_classes,
                    include_filename=args.include_filename,
                    test_time_augmentation=args.test_time_augmentation,
                    processor=processor,
                    model=model,
                    apply_template=args.apply_vision_template,
                    device=args.device,
                )

            # Hierarchical or flat classification
            current_candidates = classes_flat
            label_tree = []
            temp_hierarchy = hierarchy.copy() if hierarchy else None
            for level in range(depth+1):
                if args.classifier_type == "clip":
                    label, _ = clip_classifier.classify_image(
                        image=chunk,
                        classes=current_candidates,
                        context=args.context,
                    )
                else:
                    if args.classifier_type == "openai":
                        label = classify_with_openai(
                            desc,
                            args.include_filename,
                            f"{filename}_r{coords['row']}_c{coords['col']}",
                            current_candidates,
                            client,
                            args.classifier,
                        )
                    else:
                        label = classify_with_huggingface(
                            desc,
                            args.include_filename,
                            f"{filename}_r{coords['row']}_c{coords['col']}",
                            current_candidates,
                            gen_pipeline,
                            tokenizer,
                            28,
                        )
                # Apply fallback if label is not in the current hierarchy
                label = fallback_label(
                    label, current_candidates, chunk, filename, coords, clip_classifier, args
                )

                label_tree.append(label)
                # update candidates for next level
                if temp_hierarchy is not None and label in temp_hierarchy:
                    current_candidates = temp_hierarchy[label] if isinstance(temp_hierarchy, dict) else temp_hierarchy
                    current_candidates = list(current_candidates.keys()) if isinstance(current_candidates, dict) else current_candidates
                    temp_hierarchy = temp_hierarchy[label] if isinstance(temp_hierarchy, dict) else None
                else:
                    break

            results.append([
                filename,
                coords['row'], coords['col'], coords['left'], coords['upper'], coords['right'], coords['lower'],
                args.mode,
                desc,
                ";".join(label_tree),
                label_tree[-1]
            ])

    # write CSV
    with open(args.output_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "filename","row","col","left","upper","right","lower", "classification_mode", "description", "classification_tree", "classification"
        ])
        writer.writerows(results)

    logging.info("Done.")


if __name__ == "__main__":
    main()
