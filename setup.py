from setuptools import setup, find_packages

setup(
    name="mingrui_utils_python",          # Package name
    version="0.1",             # Version
    packages=find_packages(include=["mr_utils"]),  # Automatically discover packages
    install_requires=[         # Dependencies (optional)
        "numpy>=1.21.0",
        "requests",
    ],
    python_requires=">=3.7",   # Python version requirement
)