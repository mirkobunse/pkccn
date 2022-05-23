from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='pkccn',
    version='0.0.1',
    description='Algorithms for learning under partially-known class-conditional label noise',
    long_description=readme(),
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        'Development Status :: 3 - Alpha',
    ],
    keywords='machine-learning',
    url='http://github.com/mirkobunse/pkccn',
    author='Mirko Bunse',
    author_email='mirko.bunse@cs.tu-dortmund.de',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    extras_require = {
        'experiments' : ['imbalanced-learn', 'pandas', 'sklearn'],
        'tests' : ['imbalanced-learn', 'nose', 'sklearn']
    }
)
