from setuptools import setup, find_packages

setup(
    name="simple_cnn_classifier",
    version="0.1",
    packages=find_packages(),  
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "PyYAML"
    ],
    entry_points={
        'console_scripts': [
            'train_cnn = train:main',  
        ]
    }
)
