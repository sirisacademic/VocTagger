import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VocTagger",
    version="0.1.0",
    author="Siris Academic",
  
    description="VocTagger",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/sirisacademic/siris_tools",
    packages=['VocTagger'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={'': ['data/english_spelling.txt',
                       'data/vocabulary_IDs.txt',
                       'data/pattern_char_sub.txt']},
    install_requires=[
        'pandas>=1.0.0',
        'nltk',
        'numpy',
        'spacy>3.0.0'
        'sklearn',
        'scipy',
        'requests',
    ]
)
