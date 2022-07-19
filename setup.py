"""Install script for setuptools."""

from typing import Any, List
from setuptools import find_packages, setup
from pathlib import Path


def read(*paths: Any, **kwargs: Any) -> str:
    """Read the contents of a text file safely."""

    with open(
        Path(__file__).parent.joinpath(*paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path: str) -> List[str]:
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


long_description = """CATX implements Contextual Bandits with Continuous Actions in JAX.
It allows implementing custom neural network architecture per tree layer.
For more information see [github repository](https://github.com/instadeepai/catx)."""

__version__ = read("catx", "VERSION")

setup(
    name="catx",
    version=__version__,
    description="Contextual Bandits with Continuous Actions in JAX",
    url="https://github.com/instadeepai/catx",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="InstaDeep",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=read_requirements("requirements.txt"),
    entry_points={"console_scripts": ["project_name = project_name.__main__:main"]},
    extras_require={
        "test": read_requirements("./requirements-test.txt"),
        "tool": read_requirements("./requirements-tool.txt"),
    },
)
