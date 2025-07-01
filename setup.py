from setuptools import setup, find_packages

setup(
    name='side-scan-sonar-pipeline',
    version='0.1.0',
    packages=find_packages(),  # auto-discovers 'side_scan_sonar_pipeline'
    install_requires=[
        'numpy',
        'pandas',
        'opencv-python',
        'matplotlib',
        'tqdm',
        'scipy',
        'PyYAML'
    ],
    entry_points={
        'console_scripts': [
            'run-sss-pipeline = main:main'
        ]
    },
    author='Gabriele Scarpelli',
    description='A pipeline to preprocess and visualize side-scan sonar data.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
