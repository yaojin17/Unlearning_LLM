from setuptools import setup, find_packages

setup(
    name="llm_unlearn",
    version="0.1.0",
    packages=[
        "llm_unlearn",
        "llm_unlearn.methods",
        "llm_unlearn.utils",
    ],
    # packages=find_packages(),
)
