from setuptools import setup, find_packages

setup(
    name='stformer',
    version='1.0',
    author='Shenghao Cao',
    author_email='cao.shenghao@sjtu.edu.cn',
    description='A framework for gene representation on spatial transcriptomics data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/csh3/stFormer',
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.5',
        'pandas==2.2.0',
        'scikit-learn==1.4.0',
        'torch==2.2.2',
        'anndata==0.10.4',
        'scanpy==1.9.6',
        'flash_attn==2.5.7',
        'tqdm==4.65.0',
    ],
    python_requires='==3.11.5',
)