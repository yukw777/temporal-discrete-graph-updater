from setuptools import setup, find_packages


def requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()


# TODO: change name
setup(
    name="project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements(),
)
