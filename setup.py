import setuptools
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dagbo",
    version="0.0.1",
    author="Ross Tooley",
    author_email="rjt80@cantab.ac.uk",
    description="Bayesian optimisation with semi-parametric DAG models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rosstooley/dagbo",
    project_urls={
        "Bug Tracker": "https://github.com/rosstooley/dagbo/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #package_dir={"": "dagbo"},
    #packages=find_packages(),
    #packages=find_packages(where="dagbo"),
    #packages=[package for package in find_packages() if package.startswith('dagbo')],
    python_requires=">=3.9",
)
