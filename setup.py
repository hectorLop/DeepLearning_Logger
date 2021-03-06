from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='deeplearning_logger',
    url='https://github.com/hectorLop/DeepLearning_Logger',
    author='Hector Lopez Almazan',
    author_email='lopez.almazan.hector@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0', 
        'tensorflow>=2.1.0',
        'pandas>=1.2.0',
        'typeguard',
        'torch>=1.6.0'],
    python_requires='>=3.6',
    extras_require={
        'testing': [
            "pytest"
        ],
      },
    version='0.1',
    license='MIT',
)
