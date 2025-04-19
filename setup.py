from setuptools import setup, find_packages

setup(
    name="chatbot-finetuning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.7",
        "transformers>=4.44.0",
        "datasets>=2.21.0",
        "pyyaml>=6.0.2",
        "torch>=2.4.0",
    ],
    entry_points={
        "console_scripts": [
            "chatbot-finetuning = chatbot_finetuning.cli:cli",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A pipeline for easy fine-tuning of chat models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chatbot-finetuning",
    license="MIT",
)