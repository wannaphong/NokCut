# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="nokcut",
    version="0.4",
    description="Thai Word Segmentation using TCC + Bidirectional RNNs",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="NokCut",
    author_email="wannaphong@kkumail.com",
    url="https://github.com/wannaphongcom/NokCut/",
    packages=find_packages(),
    python_requires=">=3.5",
    package_data={},
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords="nokcut",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: Thai",
        "Topic :: Text Processing :: Linguistic",
    ],
)
