from setuptools import setup, find_packages

setup(
    name="robot_arm",
    version="0.1.0",
    packages=find_packages(where="exts"),
    package_dir={"": "exts"},
    python_requires=">=3.10",
)
