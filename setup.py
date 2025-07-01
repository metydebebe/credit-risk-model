from setuptools import setup, find_packages

setup(
    name='credit_risk_model',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'feature-engine',
        'scikit-learn',
        'pandas',
        # add other dependencies here
    ],
)
