#nsml: registry.navercorp.com/nsml/airush2020:pytorch1.5
from distutils.core import setup

setup(
    name="Spam classification - AI Rush baseline",
    version="1",
    install_requires=['imbalanced-learn']
)
