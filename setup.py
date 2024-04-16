from setuptools import find_packages, setup

NAME = 'TESSELLATE'
DESCRIPTION = 'All Sky Transient Search through TESS data'
URL = 'https://github.com/rhoxu/TESSELLATE'
EMAIL = 'roxburghhugh@gmail.com'
AUTHOR ='Hugh Roxburgh'
VERSION = '1.0.0'
REQUIRED = ['astrocut',
            'photutils>=1.4',
            'tessreduce @ git+https://github.com/rhoxu/TESSreduce.git',
            ]



setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author_email=EMAIL,
    author=AUTHOR,
    license='MIT',
    packages=['tessellate'],
    install_requires=REQUIRED
)

