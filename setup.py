from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='neuroglancer-helper',
    version='0.1',
    description='Convenience tools for Neuroglancer states',
    long_description=long_description,
    url='https://github.com/ceesem/neuroglancer_helper',
    author='Casey Schneider-Mizell',
    author_email='caseysm@gmail.com'
    )