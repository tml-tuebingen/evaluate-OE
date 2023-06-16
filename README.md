# Anonymous code submission for the paper - "Insights into Ordinal Embedding Algorithms: A Systematic Evaluation."
GPU-supported Python Library of ordinal embedding methods implemented using Pytorch.

## Methods
The following methods are implemented in this repository.
- Soft Ordinal Embedding (SOE, [6]) 
- t-Stochastic Triplet Embedding (t-STE, [7])
- Stochastic Triplet Embedding (STE, [7]) 
- Fast Ordinal Triplet Embedding (FORTE, [4]) 
- Generalized Non-metric Multi-Dimensional Scaling (GNMDS, [1]) 
- Crowd Kernel Learning (CKL, [5]) 
- Crowd Kernel Learning over X (CKL_X, [5]) 
- Landmark Ordinal Embedding (LOE, [2]) 
- Large-Scale Landmark Ordinal Embedding (LLOE, [3]) 
- Ordinal Embedding Neural Network (OENN)

Each method is associated with a Unique Method Identifier (UMI). In the library we always use the corresponding identifier to refer to a method in the code as well as in file and folder names. For example, "soe" is the unique identifier for the method SOE. To see the unique identifier for all methods use the following command.

```python docs_artifacts/print_method_ids.py```

## Setup and Installation
Create the conda environment to install required libraries for running the aforementioned algorithms using the environment.yml file.

```conda env create -f environment.yml``` 

We recommend using a machine with GPU access for faster computation times. 

## Datasets
Several datasets are integrated into the library to allow the user to experiment with and evaluate the methods. The list of available datasets can be obtained by using the following command. 

```python docs_artifacts/print_datasets.py```

## Test the methods on the '2D' Dataset Aggregation
To evaluate the methods, we provide sample configuration files for each algorithm. The configuration files contain all the parameters required for the respective method, dataset, and experiment. As an example, we provide configuration files for every method to run once on the dataset "aggregation":

```python scripts/train_UMI.py -config sample_configs/UMI/aggregation_baseline.json```

After the execution of the above scripts, you can find a log file and a 2D plot, which is saved for each method in ```logs/UMI/```. The log file ```logs/UMI/*.log``` includes most of the important parameters of the experiment specified in the config file.

## Evaluation Experiments of the main paper 
For each kind of experiment provided in the paper, we provide a sample config file for one dataset. The commands to run each of the experiment are provided below. The sample config files can easily be changed to run the experiments for other datasets as well. Note, that some of these experiments need a long time to run. We recommend to run the 'Increasing Triplets' experiment for quick results.  
 
### Increasing Sample size.

```python scripts/evaluation_experiments.py -config sample_configs/UMI/increasing_sample_size_uniform.json```

### Increasing Triplets

```python scripts/evaluation_experiments.py -config sample_configs/UMI/increasing_triplets_compound.json```

### Increasing Noise (Bernoulli, Gaussian)

```python scripts/evaluation_experiments.py -config sample_configs/UMI/aggregation_bernoulli.json```
```python scripts/evaluation_experiments.py -config sample_configs/UMI/aggregation_gaussian.json```

### Increasing Embedding dimension

```python scripts/evaluation_experiments.py -config sample_configs/UMI/increasing_d_kmnist.json```

### Increasing Original dimension

```python scripts/evaluation_experiments.py -config sample_configs/UMI/increasing_org_d_mnistpc.json```

### Convergence

```python scripts/evaluation_experiments.py -config sample_configs/UMI/convergence_char_2.json```

### CPU vs GPU

```python scripts/evaluation_experiments.py -config sample_configs/UMI/cpu_benchmark_gmm.json```

```python scripts/evaluation_experiments.py -config sample_configs/UMI/gpu_benchmark_gmm.json```

### General experiment on one dataset.

```python scripts/evaluation_experiments.py -config sample_configs/UMI/gen_usps_2.json```


## References

[1] S. Agarwal, J. Wills, L. Cayton, G. Lanckriet, D. Kriegman, and S. Belongie.  Generalized non-metric multi-dimensional scaling.  In AISTATS, 2007.

[2] J. Anderton and J. Aslam.  Scaling up ordinal embedding:  A landmark approach.  In ICML, 2019.

[3] N. Ghosh, Y. Chen, and Y. Yue.  Landmark ordinal embedding.  In NeurIPS, 2019.

[4] L. Jain, K. G. Jamieson, and R. Nowak.  Finite sample prediction and recovery bounds for ordinal embedding.  In NeurIPS, 2016.

[5] O. Tamuz, C. Liu, S. Belongie, O. Shamir, and A. Kalai. Adaptively learning the crowd kernel. In ICML, 2011.

[6] Y. Terada and U. von Luxburg.  Local ordinal embedding.  In ICML, 2014.

[7] L.  van  der  Maaten  and  K.  Weinberger.   Stochastic triplet embedding.  In Machine Learning for Signal Processing (MLSP), 2012.
