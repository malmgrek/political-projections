from setuptools import setup

setup(
    name="dimred",
    description="Dimensionality reduction demo",
    url="http://github.com/malmgrek/dimred-demo",
    author="Stratos Staboulis",
    license="MIT",
    packages=["dimred"],
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
    ],
    extras_require={
        "test": ["pytest"],
        "dash": ["dash"]
    },
)
