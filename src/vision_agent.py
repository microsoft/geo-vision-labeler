# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import re

def describe_image(
    image,
    context,
    prompt,
    classes,
    include_classes,
    processor,
    model,
):
    """
    Opens an image, processes it through the vision LLM to generate a detailed description,
    and returns a processed text label.
    """
    max_new_tokens = 128

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{image}" },
                {"type": "text", "text": f"{context}. "},
            ],
        }
    ]
            
    if include_classes:
        messages[0]["content"].append(
            {
                "type": "text",
                #"text": f"The category of this image is: {classes}. ",
                "text": f"图片的类别是: {classes}. ",
            }
        )
    messages[0]["content"].append({"type": "text", "text": prompt})

    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
            
    inputs = processor(
        image, input_text, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)                     
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    # Cleaning the generated text from any tags
    generated_text = re.sub(r"<.*?>.*?<.*?>", "", generated_text)
    generated_text = generated_text.replace("user\n\n", "").replace(
        "assistant\n\n", ""
    )
    # Remove the prompt from the description
    description = re.sub(
        rf"^.*?{re.escape(prompt.lower())}", "", generated_text.lower()
    ).strip()
    return description

def classify_with_huggingface(
    description,
    desc_list,
    class_list,
    pipeline_instance,
    tokenizer,
    max_new_tokens=10,
):
    """
    Uses an open-source language model via a Hugging Face text-generation pipeline to classify the image description.

    Args:
        description (str): The description of the image.
        filename (str): The filename of the image.
        classes (list): The list of classes to classify into.
        pipeline_instance: The Hugging Face pipeline instance.
        tokenizer: The tokenizer for the model.
        max_new_tokens (int): The maximum number of tokens to generate.
    Returns:
        str: The classification result.
    """

    messages = [
        {"role": "system", "content": "You are a text classifier."},
        {
            "role": "user",
            "content": (
                #f"Given the following list of text descriptions: {desc_list}.Their corresponding categories are: {class_list}. "
                f"对于给定的一系列文本描述: {desc_list}.他们的对应类别分别是: {class_list}. "
                #f"Now, Given a new text description, determine whether the category of the description is public_area or military_area. Note that only these two categories can be generated and no other categories can be generated. "
                f"现给定一个文本描述，给出该描述的分类类别（分类结果只能是‘城市区域’或‘目标区域’，若出现其他分类结果则扣1分并重新分类）. "
                f"除分类结果外不输出其他内容，描述：{description}. 分类："
                #f"Do not output any additional content other than one of the specified categories. Description: {description}. Category: "
            ),
        },
    ]
    response = pipeline_instance(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=1.0,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    classification = response[0]["generated_text"]

    if classification[-1] in [".", ",", "!", "?"]:
        classification = classification[:-1]
    classification = classification.strip()

    return classification
