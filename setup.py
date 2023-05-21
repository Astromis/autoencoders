import os
from setuptools import setup, find_packages

from pathlib import Path


def _install_from_git(url, package_name):
    command = f"if python -c 'import {package_name}' ; then echo Skip ; else git clone {url} && cd {Path(url).stem} && pip install . && cd - && rm -rf {Path(url).stem} ; fi "
    os.system(command)


setup(
    name='autoencoders',
    version=os.environ.get('version', '0.1.0'),
    description="indexing pipeline",
    url="https://github.com/Astromis/autoencoders",
    author='buyanov.igor.o@yandex.ru',
    licence='MIT',
    packages=find_packages(),
    install_requires=[
        'torch',
        'scikit-learn',
        'numpy',
        'tqdm',
        'faiss-cpu',
        'pyyaml',
    ],
    include_package_data = True,
    package_data={'autoencoders': ['ae_configs/*.yml']}
)


_install_from_git('https://github.com/leymir/hyperbolic-image-embeddings.git',
                  'hyptorch')
