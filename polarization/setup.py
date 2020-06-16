from setuptools import setup

install_requires = [
    'appdirs==1.4.3',
    'appnope==0.1.0',
    'click==6.7',
    'colorama==0.3.9',
    'coverage==4.5.1',
    'cycler==0.10.0',
    'decorator==4.0.11',
    'h5py==2.7.1',
    'joblib==0.11',
    'kiwisolver==1.0.1',
    'matplotlib==2.2.0',
    'networkx==1.11',
    'numpy>=1.13.1',
    'packaging==16.8',
    'pandas==0.22.0',
    'parso==0.1.1',
    'pexpect==4.2.1',
    'pickleshare==0.7.4',
    'progressbar2==3.36.0',
    'prompt-toolkit==1.0.14',
    'ptyprocess==0.5.1',
    'Pygments==2.2.0',
    'pyparsing==2.2.0',
    'python-dateutil==2.6.1',
    'python-utils==2.3.0',
    'pytz==2018.3',
    'scikit-learn==0.19.1',
    'scipy==1.0.0',
    'seaborn==0.8.1',
    'simplegeneric==0.8.1',
    'six==1.10.0',
    'sklearn==0.0',
    'traitlets==4.3.2',
    'wcwidth==0.1.7'
]

setup(
    name='polarization',
    version='0.1',
    author='Matthew A. Turner',
    author_email='maturner01@gmail.com',
    entry_points='''
        [console_scripts]
        polexp=scripts.runner:cli
    ''',
    install_requires=install_requires,
    setup_requires=['nose>=1.3.7']
)
