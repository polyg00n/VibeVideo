"""
Setup script for the Video Glitch Player package.
"""
from setuptools import setup, find_packages

setup(
    name="vibevideo",
    version="0.1.0",
    description="A video glitch effect processor with plugin support",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "vibevideo=src.__main__:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 