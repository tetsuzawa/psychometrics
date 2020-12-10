from setuptools import setup

setup(
    name="psychometrics",
    version="0.1.1",
    description="psychometrics is a library for psychological measurement experiment",
    packages=["psychometrics"],
    install_requires=open("requirements.txt").read().splitlines(),
    author="Tetsu Takizawa",
    author_email="tetsu.varmos@gmail.com",
    url="https://github.com/tetsuzawa/psychometrics"
)
