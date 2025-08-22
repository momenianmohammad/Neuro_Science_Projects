"""
Setup script for Brain Connectome Analysis System
"""

from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.0'
DESCRIPTION = 'A comprehensive Python toolkit for brain connectome analysis in computational neuroscience'

# Setting up
setup(
    name="brain-connectome-analyzer",
    version=VERSION,
    author="Your Name",
    author_email="your.email@university.edu",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    url="https://github.com/yourusername/brain-connectome-analysis",
    license="MIT",
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'networkx>=2.6',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
    ],
    extras_require={
        'neuroimaging': ['nibabel>=3.2.0'],
        'advanced': ['python-igraph>=0.9.0', 'leidenalg>=0.8.0'],
        'dev': ['pytest>=6.0.0', 'pytest-cov>=2.12.0', 'black>=21.0.0', 'flake8>=3.9.0']
    },
    keywords=[
        'neuroscience', 'brain networks', 'connectome', 'graph theory', 
        'neuroimaging', 'computational neuroscience', 'network analysis'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'brain-connectome-demo=brain_connectome_analyzer:main',
        ],
    },
)
