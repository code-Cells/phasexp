from setuptools import setup, find_packages

setup(
    name='phasexp',
    version='0.0.1',
    # url='https://github.com/mypackage.git',
    # author='Author Name',
    # author_email='author@gmail.com',
    description='Solid-State Coformational Phase Explorer.',
    packages=find_packages(),    
    install_requires=['numpy >= 2.3.3', 'networkx >= 3.6.1'],
)
