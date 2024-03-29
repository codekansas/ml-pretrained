#!/usr/bin/env python
"""Setup script for the ml-pretrained project."""

import re

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("pretrained/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in pretrained/__init__.py"
version: str = version_re.group(1)


setup(
    name="ml-pretrained",
    version=version,
    description="ML project template repository",
    author="Benjamin Bolte",
    url="https://github.com/codekansas/ml-pretrained",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
    install_requires=["ml-starter", "safetensors"],
    tests_require=["ml-starter[dev]"],
    extras_require={"dev": ["ml-starter[dev]"], "docs": ["ml-starter[docs]"]},
    package_data={"pretrained": ["py.typed"]},
)
