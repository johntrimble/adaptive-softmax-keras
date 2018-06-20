from setuptools import setup
from setuptools import find_packages
from os import path

# use the README.md for the long description
project_dir = path.abspath(path.dirname(__file__))
with open(path.join(project_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# we use python 3's namespace modules, which causes find_packages() some
# problems
_packages = ['trimble.keras.%s' % p for p in find_packages(where='./trimble/keras')]

setup(name='adaptive-softmax-keras',
      version='0.0.1',
      description='Adaptive Softmax implementation for Keras using TensorFlow backend.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='John Trimble',
      author_email='trimblej@gmail.com',
      url='https://github.com/johntrimble/adaptive-softmax-keras',
      license='MIT',
      install_requires=['Keras>=2.1.5', 'tensorflow>=1.4.1'],
      extras_require={
          'examples': ['numpy>=1.13.3', 'matplotlib>=2.1.1'],
          'tests': ['pytest', 'numpy>=1.13.3'],
      },
      classifiers=(
          'Programming Language :: Python :: 3.5',
          'License :: OSI Approved :: MIT License',
          'Development Status :: 1 - Planning'
      ),
      packages=_packages)
