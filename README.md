# pdMISVM.jl

This repository contains the reproducibility information for the ICDM paper titled: "A Linear Primal-Dual Multi-Instance SVM for Big Data Classifications". If you have any issues running this code please open a GitHub issue [here](https://github.com/minds-mines/pdMISVM.jl/issues).

If you find this code useful please consider citing the following:

```bibtex
TODO
```

## Code

This code base is using the [Julia Language](https://julialang.org/) to make a reproducible scientific project named
> pdMISVM.jl 

To (locally) reproduce this project, do the following:

0. Download this code base. 
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/code")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.

2. Run the tests associated with the linear and kernel pdMISVM models:
   ```
   julia> include("test/pdmisvm_test.jl")
   julia> include("test/kernel_pdmisvm_test.jl")
   ```
These tests ensure that the updates derived in Algorithm 1 are correct. E.g. since variable update is derived with respect to a primal variable, and the minimization is quadratic with respect to that variable, the Lagrangian should be a minimum after that variable has been updated. **Please note: The frist time this code is run it may take some extra time.**

3. For an example on running the pdMISVM.jl model on the MUSK-2 dataset run:
   ```
   julia> include("musk_example.jl")
   ```

4. The code for the linear (Section 2.3-2.4) and kernel (Section 3.5) models are located in `pdMISVMClassifier.jl` and `KernelpdMISVMClassifier.jl`, respectively.

## Datasets

Where to download each of the datasets used in our paper. Note that each dataset should be included with `data/musk2.data` in the `data` folder.

 - **MUSK-2**: Downloaded from the UCI website: https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)
 - **Elephant, Fox, and Tiger**: Downloaded from: http://www.cs.columbia.edu/~andrews/mil/datasets.html
 - **MNIST-bags**: This dataset is derived from: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
 - **SIVAL**: Bags and labels are downloaded from: https://www.cs.wustl.edu/~sg/accio/SIVAL.html
   - If the link above doesn't work for some reason the dataset is mirrored at: https://drive.google.com/file/d/1CE7NBfOgE6l3oA-TVdWCQV3FEpZqLlt3/view?usp=sharing
   - If using the SIVAL dataset please cite: Rahmani, Rouhollah, et al. "Localized content based image retrieval." Proceedings of the 7th ACM SIGMM international workshop on Multimedia information retrieval. 2005.

### SIVAL-deep Processing Pipeline

1. Download the raw SIVAL data from above and put into the `data` folder.

2. Get the pretrained edgebox model from [here](https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz) and into the `data` folder using the following commands:

```bash
wget https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz -O data/model.yml.gz
cd data
gunzip model.yml.gz
```

3. Ensure that [PyCall](https://github.com/JuliaPy/PyCall.jl) is appropriately setup (perhaps with a conda environment named "mypyenv") and that the following dependencies are met:
 - pytorch
 - opencv

4. Run the following command to begin parsing the SIVAL-deep dataset into the `data` folder:

```bash
conda activate mypyenv
julia --project sival_deep_pipeline.jl
```

## Hyperparameter Settings

The hyperparameter settings for each method-dataset pair for the results reported in Table 1 and Table 2 are as follows:

Models implemented from https://github.com/garydoranjr/misvm:

| Dataset | Model | kernel |   C   |
| ------- | ----- | -----  | ----- |
| MUSK-2 | SIL | linear | 1e4 |
| Elephant | SIL | linear | 1.0 |
| Fox | SIL | linear | 1.0 |
| Tiger | SIL | linear | 1.0 |
| MNIST-bags | SIL | linear | 0.1 |
| SIVAL | SIL | linear | 1e4 |
| MUSK-2 | miSVM | linear | 1e4 |
| Elephant | miSVM | linear | 10.0 |
| Fox | miSVM | linear | 100.0 |
| Tiger | miSVM | linear | 10.0 |
| MNIST-bags | miSVM | linear | 100.0 |
| SIVAL | miSVM | linear | 1e4 |
| MUSK-2 | MISVM | linear | 100.0 |
| Elephant | MISVM | linear | 10.0 |
| Fox | MISVM | linear | 100.0 |
| Tiger | MISVM | linear | 1e5 |
| MNIST-bags | MISVM | linear | 1e4 |
| SIVAL | MISVM | linear | 1e4 |
| MUSK-2 | NSK | linear | 0.1 |
| Elephant | NSK | linear | 0.1 |
| Fox | NSK | linear | 0.01 |
| Tiger | NSK | linear | 10.0 |
| MNIST-bags | NSK | linear | 100.0 |
| SIVAL | NSK | linear | 100.0 |
| MUSK-2 | sMIL | linear | 1.0 |
| Elephant | sMIL | linear | 1e-5 |
| Fox | sMIL | linear | 0.01 |
| Tiger | sMIL | linear | 0.1 |
| MNIST-bags | sMIL | linear | 0.01 |
| SIVAL | sMIL | linear | 0.01 |
| MUSK-2 | sbMIL | linear | 1e5 |
| Elephant | sbMIL | linear | 100.0 |
| Fox | sbMIL | linear | 1e5 |
| Tiger | sbMIL | linear | 1e5 |
| MNIST-bags | sbMIL | linear | 1000 |
| SIVAL | sbMIL | linear | 1e4 |

Models implemented from https://github.com/yanyongluan/MINNs:

| Dataset | Model |   pooling   |   lr   |   decay   | momentum | max_epoch |
| ------- | ----- | ----- | ----- | ----- | ----- | ----- |
| MUSK-2 | miNet | max | 1e-4 | 0.03 | 0.9 | 50 |
| Elephant | miNet | max | 1e-4 | 0.05 | 0.9 | 50 |
| Fox | miNet | max | 5e-4 | 0.05 | 0.9 | 50 |
| Tiger | miNet | max | 5e-4 | 0.03 | 0.9 | 50 |
| MNIST-bags | miNet | max | 1e-4 | 0.001 | 0.9 | 50 |
| SIVAL | miNet | max | 1e-4 | 0.003 | 0.9 | 50 |
| MUSK-2 | MINet | max | 1e-4 | 0.03 | 0.9 | 50 |
| Elephant | MINet | ave | 1e-4 | 0.005 | 0.9 | 50 |
| Fox | MINet | ave | 1e-4 | 0.05 | 0.9 | 50 |
| Tiger | MINet | lse | 1e-4 | 0.05 | 0.9 | 50 |
| MNIST-bags | MINet | max | 1e-4 | 0.001 | 0.9 | 50 |
| SIVAL | MINet | max | 1e-4 | 0.001 | 0.9 | 50 |

Our implemented models:

| Dataset | Model |   C   |   μ   |   ρ   |
| ------- | ----- | ----- | ----- | ----- |
| MUSK-2 | Ours | 1e3 | 1e-5 | 1.2 |
| Elephant | Ours | 1e-3 | 1e-3 | 1.2 |
| Fox | Ours | 100.0 | 1e-3 | 1.2 |
| Tiger | Ours | 1e-3 | 1e-3 | 1.2 |
| MNIST-bags | Ours | 1.0 | 1e-3 | 1.2 |
| SIVAL | Ours | 1e2 | 1e-10 | 1.2 |
| MUSK-2 | Ours (inexact) | 1e10 | 1e-10 | 1.2 |
| Elephant | Ours (inexact) | 0.01 | 1e-6 | 1.2 |
| Fox | Ours (inexact) | 1000.0 | 1e-6 | 1.2 |
| Tiger | Ours (inexact) | 10.0 | 1e-6 | 1.2 |
| MNIST-bags | Ours (inexact) | 1e8 | 1e-5 | 1.2 |
| SIVAL | Ours (inexact) | 1e2 | 1e-10 | 1.2 |
| SIVAL-deep | Ours (inexact) | 1e10 | 1e-10 | 1.2 |
