import pathlib
from setuptools import setup, find_packages

setup_path = pathlib.Path(__file__).parent
readme = (setup_path / "README.md").read_text(encoding="utf-8")

with (setup_path / "requirements.txt").open() as f:
    requirements = f.read().splitlines()

setup(
    name="Folderesque",
    version="0.1.1",
    description="Python Script to process and upscale images in specified folders using RRDB models.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Sevilze/Folderesque",
    entry_points={
        "console_scripts": [
            "Folderesque = Folderesque.__main__:main",
        ],
    },
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    package_data={"Folderesque": ["config.py"]},
)
