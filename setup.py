import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="invink",
    version="0.1.0",
    author="Vishnu Vinod",
    author_email="vishnuvinod2001@gmail.com",
    description="Implementation of InvisibleInk for differentially private synthetic text generation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cerai-iitm/invisibleink",
    project_urls={
        "Bug Tracker": "https://github.com/cerai-iitm/invisibleink/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        'numpy>=2.2.4',
        'pandas>=2.2.3',
        'scipy>=1.15.2',
        'tqdm>=4.67.1',
        'torch>=2.6.0',
        'accelerate>=1.5.2',
        'transformers>=4.57.0',
        'datasets>=3.4.1'
        ]
)