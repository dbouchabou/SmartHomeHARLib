# coding: utf-8
#!/usr/bin/env python3

from setuptools import setup,find_packages

with open('README.rst') as f:

    long_description = f.read()

VERSION = "0.1"

setup(

    name='SmartHomeHARLib',

    version=VERSION,

    license='GPL License',

    author='Damien Bouchabou',

    author_email='damien.bouchabou@gmail.com',

    #url='',

    description=('SmartHomeHARLib is a Human Activity Recognition Lib from Smart Home Datasets'

                 ''),

    long_description=long_description,

    classifiers=[

        'Programming Language :: Python',

        'Topic :: Software Development :: Libraries :: Python Modules',

    ],

    keywords=['human activity recognition', 'home-automation'],

    platforms='any',

    packages=find_packages(),

    include_package_data=True,

    install_requires=[

        "pandas",

        "numpy"

    ]

)
