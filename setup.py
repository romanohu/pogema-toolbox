import codecs
import os
import re
import pathlib

from setuptools import setup, find_packages

cur_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cur_dir, 'README.md'), 'rb') as f:
    lines = [x.decode('utf-8') for x in f.readlines()]
    lines = ''.join([re.sub('^<.*>\n$', '', x) for x in lines])
    long_description = lines


def read(*parts):
    with codecs.open(os.path.join(cur_dir, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        version_file,
        re.M,
    )
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")

    install_requires=[
        "gymnasium==0.28.1",
        "numpy>=1.19.2",
        "pydantic>=2.0.0",
    ],


def load_requirements(fname="requirements.txt"):
    lines = pathlib.Path(fname).read_text().splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]


setup(
    name='pogema-toolbox',
    author='Alexey Skrynnik',
    license='Apache License 2.0',
    version=find_version("pogema_toolbox", "__init__.py"),
    description='Evaluation toolbox for Pogema environment',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Tviskaron/pogema-toolbox',
    install_requires=load_requirements(),
    extras_require={

    },
    package_dir={'': './'},
    packages=find_packages(where='./', include='pogema_toolbox*'),
    include_package_data=True,
    python_requires='>=3.8',
    package_data={
        'pogema_toolbox': ['licenses/*', 'maps/*.yaml'],
    },
)
