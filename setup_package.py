from setuptools import setup, find_packages

setup(
    name="deepdoc-eval",
    version="1.0.0",
    author="DeepDoc Team",
    description="Standalone Evaluation Harness for Document Intelligence (RAGAS + LLM-Judge)",
    packages=find_packages(),
    install_requires=[
        "ragas>=0.1.7",
        "mlflow>=2.12.1",
        "openai>=1.23.0",
        "datasets>=2.19.0",
        "scipy>=1.13.0",
        "tenacity>=8.2.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
