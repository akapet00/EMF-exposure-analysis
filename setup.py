from setuptools import setup, find_packages


with open('README.md') as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(name='dosipy',
      url='https://github.com/akapet00/EMF-exposure-analysis',
      version='0.0.1',
      packages=find_packages(),
      install_requires=requirements,
      provides=['dosipy'],
      python_requires='>=3.7',
      use_2to3=False,
      zip_safe=False,
      description='High-frequency EM dosimetry simulation software.',
      long_description=long_description,
      author='Ante Kapetanovic',
      author_email='akapet00@gmail.hr',
      license='MIT',
      classifiers=[
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
      keywords='radiation dosimetry bioelectromagnetism',
      )
