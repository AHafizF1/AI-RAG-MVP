from setuptools import setup, find_packages

setup(
    name="manufacturing_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open("requirements-rag.in").read().splitlines(),
    extras_require={
        "dev": open("requirements-dev.in").read().splitlines(),
    },
    python_requires=">=3.8",
)
