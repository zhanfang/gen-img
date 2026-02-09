from setuptools import setup, find_packages

setup(
    name="stable-diffusion-core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "diffusers",
        "numpy",
        "Pillow",
        "tqdm"
    ],
    description="Stable Diffusion Core Implementation for Learning",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/stable-diffusion-core"
)
