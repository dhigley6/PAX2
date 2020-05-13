"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]


setup(
    name='pax2',
    version='0.0.1',
    description='deconvolution of PAX data and simulations',
    long_description_content_type='text/markdown',
    author='Daniel Higley',
    author_email='dhigley@slac.stanford.edu',
    packages=find_packages(),
    python_requires='>=3.6, <4',
    install_requires=requirements
)