from setuptools import setup, find_packages

setup(
    name="toonforge",
    version="0.3.0",
    description="Real-time multi-model cartoon stylization with hybrid face + scene pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="YOUR_NAME",
    author_email="your@email.com",
    url="https://github.com/YOUR_USERNAME/toonforge-realtime-cartoon-stylization",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
    keywords=[
        "cartoon", "stylization", "real-time", "face", "anime",
        "vtoonify", "animegan", "style-transfer", "webcam",
        "deep-learning", "pytorch", "computer-vision",
    ],
)
