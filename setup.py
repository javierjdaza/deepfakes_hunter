from setuptools import setup, find_packages
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name='deepfakes_hunter',
    version=get_version("deepfakes_hunter/__init__.py"),
    description='Face-swap deepfake detection for KYC pipelines',
    long_description="".join(open("README.md", "r").readlines()),
    long_description_content_type='text/markdown',
    url='https://github.com/javierjdaza/deepfakes_hunter',
    author='Javier Javier Daza Olivella',
    author_email='jjdazao@eafit.edu.co',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        "onnxruntime>=1.16.0",
        "numpy",
        "pillow",
        "opencv-python-headless",
        "albumentations>=1.3.0",
        "torch>=1.7.1",
        "gdown>=4.7.1",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
