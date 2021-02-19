from setuptools import setup

setup(
    name="poliparties",
    description="Analyzing political survey data",
    url="http://github.com/malmgrek/poliparties",
    author="Stratos Staboulis",
    license="MIT",
    packages=["poliparties"],
    install_requires=[
        "numpy",
        "scikit-learn",
    ],
    extras_require={
        "test": ["pytest"],
    },
)
