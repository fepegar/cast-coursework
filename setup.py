from setuptools import setup, find_packages

setup(name='cast',
      version='0.1.0',
      author='Fernando Perez-Garcia',
      author_email='fernando.perezgarcia.17@ucl.ac.uk',
      packages=find_packages(exclude=['*tests']),
      install_requires=[
          'scikit-image',
          'scikit-learn',
          'numpy',
          'matplotlib',  # on macOS matplotlib should be installed with conda
          ],
     )
