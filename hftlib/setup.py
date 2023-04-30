from setuptools import setup, find_packages

setup(
    name='hftlib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
)
