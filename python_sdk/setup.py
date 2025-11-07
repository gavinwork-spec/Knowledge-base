"""
Setup script for the Knowledge Base API Python SDK.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="knowledge-base-sdk",
    version="1.0.0",
    author="Knowledge Base Team",
    author_email="support@knowledgebase.com",
    description="Python SDK for the Knowledge Base API Suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knowledge-base/sdk-python",
    project_urls={
        "Bug Tracker": "https://github.com/knowledge-base/sdk-python/issues",
        "Documentation": "https://docs.knowledgebase.com/sdk/python/",
        "Source Code": "https://github.com/knowledge-base/sdk-python",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.5.0",
            "mkdocstrings[python]>=0.20.0",
        ],
        "performance": [
            "orjson>=3.8.0",
            "aiohttp>=3.8.0",
        ],
    },
    keywords=[
        "knowledge-base",
        "search",
        "api",
        "sdk",
        "semantic-search",
        "machine-learning",
        "ai",
        "personalization",
        "websocket",
        "async",
    ],
    include_package_data=True,
    package_data={
        "knowledge_base_sdk": ["py.typed"],
    },
    entry_points={
        "console_scripts": [
            "kb-sdk=knowledge_base_sdk.cli:main",
        ],
    },
)