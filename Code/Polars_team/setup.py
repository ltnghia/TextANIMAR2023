import io
import os
from setuptools import find_packages, setup

def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    """

    content = ""
    with io.open(
            os.path.join(os.path.dirname(__file__), *paths),
            encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content

setup(
    name="shrec2023",
    version="0.0.1",
    description="Siamese model for 3D point cloud model retrieval from sketch",
    url="https://github.com/ToTuanAn/SHREC2023",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="hcmus",
    packages=find_packages(exclude=["tests", ".github"])
)
