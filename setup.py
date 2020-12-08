import setuptools

with open('README.md', 'r') as fp:
    readme = fp.read()

setuptools.setup(
    name='ezmock',
    version='0.1',
    author='Tony Zhang',
    author_email='txz@stanford.edu',
    description="Zel'dovich-approximation-based mock galaxy catalog creation [1409.1124]",
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=['jinja2, nbodykit'],
)

