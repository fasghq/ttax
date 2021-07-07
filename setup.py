from setuptools import setup

setup(name='ttax',
      version='0.0.2',
      description='Tensor Train decomposition on Jax',
      url='https://github.com/fasghq/ttax',
      author='Alexander Novikov and Dmitry Belousov',
      author_email='sasha.v.novikov@gmail.com',
      license='MIT',
      packages=['ttax'],
      install_requires=[
          'numpy',
          'dm-tree',
          'jax',
          'flax'
      ],
      zip_safe=False)
