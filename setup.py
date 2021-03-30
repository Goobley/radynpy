from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='radynpy',
      version='0.5.0',
      description='Analysis tools for Radyn in Python',
      long_description=readme(),
      url='http://github.com/Goobley/radynpy',
      author='Chris Osborne',
      author_email='c.osborne.1@research.gla.ac.uk',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'scikit-image', 'matplotlib', 'scipy', 'colour', 'palettable', 'cdflib'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License'
      ],
      include_package_data=True,
      zip_safe=True)
