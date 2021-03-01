from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='DeepLeaning_Logger',
    url='https://github.com/hectorLop/DeepLearning_Logger',
    author='Hector Lopez Almazan',
    author_email='lopez.almazan.hector@gmail.com',
    packages=['dl_logger'],
    install_requires=['numpy', 'tensorflow', 'typeguard'],
    python_requires='>=3.5',
    extras_require={
        'testing': [
            "pytest"
        ],
      },
    version='0.1',
    license='MIT',
    description='An example of a python package from pre-existing code',
)