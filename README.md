# JL+LSH Experimental Setup

This repository contains the experimental infrastructure for evaluating Johnson-Lindenstrauss (JL) projection combined with Locality-Sensitive Hashing (LSH) for Approximate Nearest Neighbor (ANN) search.

## Project Structure

```
project/
├── data/                           # Datasets not included in repo
│   ├── sift/
│   │   ├── sift_base.fvecs
│   │   ├── sift_query.fvecs
│   │   └── sift_groundtruth.ivecs
│   ├── gist/
│       ├── gist_base.fvecs
│       ├── gist_query.fvecs
│       └── gist_groundtruth.ivecs
│
├── evaluator.py                    # ANNEvaluator class for consistent metrics
├── datasets.py                     # Dataset loading utilities
├── 01_baseline_knn.ipynb           # Baseline experiments (this file)
├── 03_lsh_ann_djl.ipynb            # Search for hyperparameters to use in experiments
├── DJL+LSH_vs_H_gist.ipynb         # GIST Dataset experiments
├── DJL+LSH_vs_H_gist.ipynb         # SIFT Dataset experiments
│
│
├── pre_testing_notebooks/          # Testing DJL and LSH to figure out how we wanted to set up experiments/algorithm
│   ├── 02_jl_ann_optimized.ipynb
│   ├── 02_jl_ann.ipynb
│   ├── 03_lsh_ann.ipynb
│   └── 03_lsh_faiss_optimized.ipynb
│
├── results/                        # Experiment results (auto-created)
│   ├── baseline_knn_sift.json
│   ├── baseline_knn_gist.json
│   └── baseline_knn_summary.json
│
└── README.md                       # This file
```

## Installation

### Requirements

```bash
pip install numpy scikit-learn matplotlib seaborn psutil jupyter
```

### Optional (for future notebooks)
```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install hnswlib
```

## Datasets

Download the benchmark datasets from:

- **SIFT-1M**: http://corpus-texmex.irisa.fr/
- **GIST-1M**: http://corpus-texmex.irisa.fr/
- **Deep1B**: http://sites.skoltech.ru/compvision/noimi/

Place the `.fvecs` and `.ivecs` files in the corresponding `data/` subdirectories.

**Note**: Deep1B is very large (1 billion vectors). Use the 10M subset (`deep10M_base.fvecs`) for initial experiments.

## Usage

### 1. Baseline KNN Experiments

The baseline notebook establishes exact k-NN performance:

```bash
jupyter notebook 01_baseline_knn.ipynb
```

**What it does:**
- Loads SIFT, GIST, and Deep1B datasets
- Computes ground truth using brute-force k-NN
- Measures query time, build time, memory usage, and recall
- Saves results to `results/` directory

### 2. Using the Evaluator Class

The `ANNEvaluator` class provides a consistent interface for evaluating any ANN algorithm:

```python
from evaluator import ANNEvaluator
from datasets import DatasetLoader

# Load dataset
loader = DatasetLoader(data_dir="data")
X_train, X_test = loader.load_sift(n_train=100000, n_test=1000)

# Initialize evaluator
evaluator = ANNEvaluator(X_train, X_test, k=10)

# Compute ground truth
evaluator.compute_ground_truth()

# Define your algorithm
def my_index_builder(X_train, **params):
    # Build your index here
    return index

def my_query_func(index, X_test, k):
    # Query your index here
    return indices, distances

# Evaluate
results = evaluator.evaluate(
    index_builder=my_index_builder,
    query_func=my_query_func,
    method_name="MyMethod",
    param1=value1,
    param2=value2
)

# Save results
evaluator.save_results(results, "results/my_method.json")
```

### 3. Metrics Computed

The evaluator automatically computes:

- **Recall@K**: Fraction of true k-nearest neighbors retrieved
- **Query time**: Average milliseconds per query
- **Build time**: Index construction time in seconds
- **Memory**: Memory footprint in MB
- **Recall statistics**: Mean, std, min, max across queries

## Experiments & Results

We validated our theoretical Lipschitz analysis by simulating a **JL + LSH composition pipeline** on standard ANN benchmarks. Our goal was to determine if our derived **Hardness Score ($H$)** correlates with the probability of retrieval failure (i.e., failing to find the true nearest neighbor).

### Experimental Setup
* **Datasets:** SIFT-1M (128d, local descriptors) and GIST-1M (960d, global descriptors).
* **Pipeline:**
    1.  **Dimension Reduction:** Gaussian Random Projection ($d \rightarrow d'$).
    2.  **Indexing:** E2LSH (Exact Euclidean LSH) with $p$-stable distributions.
* **Parameters:** We tested varying target dimensions ($d' \in \{32, 48, 96\}$) and LSH bucket widths ($r \in \{150, 250\}$).

### Key Findings

#### 1. Hardness is a Strong Predictor of Failure
Across all experimental settings, we observed a strong negative correlation between the Hardness Score and collision probability. Queries with high $H$ scores consistently failed to retrieve their true nearest neighbors, validating $H$ as a reliable confidence metric.

#### 2. Distinct "Failure Zones" by Geometry
We identified specific geometric thresholds where the JL distortion overwhelms the gap between the nearest neighbor and false positives. These thresholds vary by dataset geometry:
* **SIFT-1M (Logarithmic Behavior):** Failures cluster in the high hardness range ($0.8 < H < 1.0$). The relationship between hardness and distance is logarithmic, reflecting the exponentially increasing volume of high-dimensional local feature spaces.
* **GIST-1M (Linear Behavior):** Failures cluster in the lower range ($0.2 < H < 0.6$). The relationship is linear, due to GIST descriptors lying on a normalized hypersphere where Euclidean distance proxies angular separation.

#### 3. Sensitivity to Quantization
Tighter LSH bucket widths ($r=150$) resulted in a steeper drop-off in success probability relative to hardness compared to looser widths ($r=250$). However, the *ranking* of hard queries remained invariant—the same queries were identified as "hard" regardless of hyperparameters.

### Visualizations
*(See `results/` for full charts)*

| Confidence Curve | Predictor Separation |
| :---: | :---: |
| ![Confidence Curve](path/to/your/confidence_chart.png) | ![Density Plot](path/to/your/density_chart.png) |
| *Probability of success drops as Hardness ($H$) increases.* | *Failures (Red) are clearly separated from Successes (Green) by $H$.* |

### Reproducing Results
To replicate the experiments and generate the charts:

1.  **Run the Verification Script:**
    ```bash
    python numerical_verification_E2LSH.py
    ```
2.  **Run the Dataset Analysis:**
    Open `DJL+LSH_vs_H_sift.ipynb` or `DJL+LSH_vs_H_gist.ipynb` to execute the pipeline and visualize the geometric failure zones.

## Key Results to Track

For each method, track:

1. **Space-Time Trade-off**: Index size vs query time
2. **Accuracy-Time Trade-off**: Recall vs query time
3. **Dimensionality Scaling**: How performance changes with d
4. **Optimal k**: Best target dimension for JL projection


## Tips for Next Notebooks

### 01_baseline_knn.ipynb
- Establish ground truth using brute-force Exact k-NN (Linear Scan)
- Benchmark query time, build time, and memory usage for SIFT-1M and GIST-1M
- Validate that Recall@k is 1.0 to ensure dataset integrity
- Set baseline performance metrics for later comparison with ANN methods

### 03_lsh_ann_djl.ipynb
- Combine JL + LSH
- Analyze collision probability preservation (Lipschitz analysis)
- Generate final Pareto curves
- Derive practical guidelines

### DJL+LSH_vs_H_sift.ipynb
- Execute the full JL + LSH pipeline on the SIFT-1M dataset (Local Descriptors)
- Calculate the **Lipschitz Hardness Score ($H$)** for every query
- Visualize the "Failure Zone" ($0.8 < H < 1.0$) where retrieval fails
- Demonstrate the logarithmic relationship between hardness and query-neighbor distance
- Validate that hardness rankings are invariant to bucket width ($r$) and target dimension ($d'$)

### DJL+LSH_vs_H_gist.ipynb
- Execute the full JL + LSH pipeline on the GIST-1M dataset (Global Descriptors)
- Validate Hardness Score predictions on a dataset with different geometry (Hypersphere)
- Visualize the "Failure Zone" ($0.2 < H < 0.6$) specific to normalized data
- Demonstrate the linear relationship between hardness and angular distance
- Confirm the robustness of the Hardness metric across diverse data manifolds

## Troubleshooting

### Dataset Loading Issues
- Ensure `.fvecs` and `.ivecs` files are in correct directories
- Check file permissions
- For Deep1B, use 10M subset first

### Memory Issues
- Reduce `n_train` and `n_test` in configuration
- Process queries in smaller batches
- Use swap space for very large datasets

### Slow Performance
- Use `n_jobs=-1` to enable parallel processing
- Reduce test set size for faster iteration
- Start with SIFT (smallest dataset)

## Citation

If you use this code, please cite:

```bibtex
@article{jl-lsh-composition,
  title={Johnson-Lindenstrauss + LSH Composition via Lipschitz Analysis},
  author={Goyal, Dev and Castillo, Alessandro and Khajanchi, Anoushka},
  year={2025}
}
```

## References

- Johnson & Lindenstrauss (1984): Original JL Lemma
- Achlioptas (2001): Sparse random projections
- Indyk & Motwani (1998): LSH framework
- Datar et al. (2004): E2LSH family
- Andoni & Indyk (2008): LSH survey

## Contact

Alessandro Castillo: agc2166@columbia.edu
Anoushka Khajanchi: ak5446@columbia.edu
Dev Goyal: dg3513@columbia.edu

---
