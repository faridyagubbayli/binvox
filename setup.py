from distutils.core import setup
from binvox import __version__

setup(
    name='binvox',
    packages=['binvox'],
    version=__version__,
    license='MIT',
    description='Library for loading & saving the Binvox files',
    author='Farid Yagubbayli',
    author_email='faridyagubbayli@gmail.com',
    url='https://github.com/faridyagubbayli/binvox',
    keywords=['Binvox'],
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
)
