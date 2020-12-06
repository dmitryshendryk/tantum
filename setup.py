from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(
        name="tantum",
        version="0.0.5",
        description="tantum - train pytorch models",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Dmitry Shendryk",
        author_email="dmitryshendryk@gmail.com",
        url="https://github.com/dmitryshendryk/tantum",
        license="Apache License",
        packages=find_packages(),
        include_package_data=True,
        install_requires=["torch>=1.6.0"],
        python_requires=">3.6.2",
    )
