from setuptools import setup, find_packages


def requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()


setup(
    name="dgu",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements(),
)
