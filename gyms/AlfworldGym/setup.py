from setuptools import setup, find_packages

setup(
    name="alfworldgym",
    version="1.0.0",
    description="A Gymnasium environment for alfworld household task completion",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.26.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "alfworld",
    ],
    include_package_data=True,
)
