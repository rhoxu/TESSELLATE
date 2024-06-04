from setuptools import find_packages, setup

NAME = 'TESSELLATE'
DESCRIPTION = 'All Sky Transient Search through TESS data'
URL = 'https://github.com/rhoxu/TESSELLATE'
EMAIL = 'roxburghhugh@gmail.com'
AUTHOR ='Hugh Roxburgh'
VERSION = '1.0.0'
REQUIRED = ['astrocut',
            'photutils>=1.4',
            'tessreduce @ git+https://github.com/rhoxu/TESSreduce.git', #git+https://github.com/CheerfulUser/TESSreduce.git@dev',#@b3054d51f2f0993a2ce386d5f2ea7635e5aa2288'
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




