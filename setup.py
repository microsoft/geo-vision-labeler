from setuptools import setup, find_packages

setup(
    name="geo_vision_labeler",
    version="0.1.0",
    description="A flexible tool to label images using vision LLMs and text classification via OpenAI or open-source LLMs.",
    author="Gilles Quentin Hacheme",
    packages=find_packages(),
    install_requires=[
        "Pillow==11.1.0",
        "rasterio==1.4.3",
        "transformers==4.50.0",
        "torch==2.7.0",
        "tqdm==4.67.1",
        "numpy==2.2.4",
        "openai==1.65.2",
        "torchvision==0.22.0",
        "python-dotenv==1.2.2",
        "pytest==8.3.5",
        "pandas==2.2.3",
        "pyyaml==6.0.2"
    ],
    entry_points={
        "console_scripts": [
            "label-images=src.main:main"
        ]
    },
    python_requires="==3.10",
)
