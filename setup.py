from setuptools import setup

setup(name='ttax',
      version='0.0.1',
      description='Tensor Train decomposition on Jax',
      url='https://github.com/fasghq/ttax',
      author='Alexander Novikov',
      author_email='sasha.v.novikov@gmail.com',
      license='MIT',
      packages=['ttax'],
      install_requires=[
          'numpy',
          'jax',
      ],
      zip_safe=False)
