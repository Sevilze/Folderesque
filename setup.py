import os
import shutil
import pathlib
from setuptools import setup, find_packages
from setuptools.command.install import install

setup_path = pathlib.Path(__file__).parent
readme = (setup_path / "README.md").read_text(encoding="utf-8")

with (setup_path / "requirements.txt").open() as f:
    requirements = f.read().splitlines()


class InstallConfig(install):
    def run(self):
        install.run(self)
        config_path = os.path.join(os.path.dirname(__file__), "Folderesque", "config.py")
        install_dir = os.path.join(os.path.dirname(__file__), "config.py")
        shutil.copy(config_path, install_dir)


setup(
    name="Folderesque",
    version="0.1.3",
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
    cmdclass={
        "install": InstallConfig,
    },
)
