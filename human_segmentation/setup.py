from setuptools import setup, find_packages

setup(
    name="human_segmentation",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "open3d>=0.13.0",
        "matplotlib>=3.4.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for segmenting 3D human models into bounding boxes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
)