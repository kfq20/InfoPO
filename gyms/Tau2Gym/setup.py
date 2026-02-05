"""
Setup configuration for Tau2Gym package.
"""

import os
from setuptools import setup, find_packages

setup(
    name="tau2gym",
    version="1.0.0",
    description="Tau2-Bench integration for UserRL - Gymnasium environment for conversational agent RL training",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="UserRL Contributors",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        # tau2-bench will be installed separately
    ],
    extras_require={
        "tau2": [
            # Users should install tau2-bench from source:
            # cd ../tau2-bench && pip install -e .
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
