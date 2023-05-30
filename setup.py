from setuptools import find_packages, setup
setup(
    name='sber_commit_classification',
    packages=find_packages(include=['sber_commit_classification']),
    version='0.1.0',
    description='Sber commit classification library',
    author='Me',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests'
)