import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="libauc",
  version="2.0.1",
  author="Zhuoning Yuan, Tianbao Yang",
  description="LibAUC: A Deep Learning Library for X-Risk Optimization",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/Optimization-AI/LibAUC",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
  python_requires=">=3.8",
  install_requires = [
                      'torch',
                      'torchvision',
                      'numpy',
                      'tqdm',
                      'scipy',
                      'pandas',
                      'Pillow',
                      'scikit-learn',
                      'opencv-python',
                      'torch_geometric',
                      'ogb',
                      'webdataset',
                      'omegaconf',
                      'transformers',
                      'medmnist',
                      'pyyaml'],
  extras_require = {
      # libauc.auto: hyperparameter search (pip install libauc[auto])
      'auto': ['smac>=2.3', 'ConfigSpace'],
      # libauc.auto.ui: desktop GUI (pip install libauc[auto,gui])
      'gui': ['flet==0.24.1', 'psutil'],
      # Study-level MIL backbones (RADEO / BioVFM / OmniRad / CoAtNet) and
      # DICOM preprocessing (pip install libauc[medical])
      'medical': ['timm', 'pydicom'],
  },
)
