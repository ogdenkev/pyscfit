import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyscfit",
    version="0.0.1",
    
    author="Kevin Ogden",
    author_email="ogdenkev@gmail.com",
    
    description="Fit single channel gating mechanism rates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    url="https://github.com/ogdenkev/pyscfit",
    
    packages=setuptools.find_packages("src"),
    
    # tell distutils packages are under src
    package_dir={'':'src'},
    
    install_requires = [
        "numpy>=1.17",
        "scipy>=1.3",
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.7",
)
