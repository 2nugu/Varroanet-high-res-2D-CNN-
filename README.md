# Interpretable Deep Learning for Varroa Mite Detection

**Multi-Resolution Grad-CAM++ Analysis with Channel Attention and Preprocessing Sensitivity for Interpretable *Varroa destructor* Detection**

> Submitted to *Agronomy* (MDPI) — under major revision

## Overview

This repository provides the experimental framework for training and evaluating **25 CNN model configurations** across four feature-map resolutions (**$7\times7$**, **$14\times14$**, **$28\times28$**, **$56\times56$**) with a focus on explainability. The full study encompasses **1,548 training runs** across **300 unique model-dataset-resolution combinations**, providing a comprehensive analysis of how spatial resolution and preprocessing affect both classification accuracy and XAI localization quality.

 **Unified Resolution Scaling for Comparative Robustness** : To ensure a rigorous and fair comparative analysis, the stride modification strategy was applied consistently across all evaluated architectures (ShuffleNet-V2, MobileNetV3-Small, and EfficientNet-B0). By selectively replacing stride-2 convolutions with stride-1 in the final stages, every model was evaluated across identical feature map resolutions. This framework allows for a direct assessment of how spatial resolution affects the localization of small-scale *Varroa* mites, independent of the underlying backbone design.

 **Optimization for Small-Object Detection** : *Varroa destructor* mites are significantly small-scale targets relative to their honeybee hosts. This repository focuses on an architecture optimized to capture these fine-grained features even within low-resolution feature maps, ensuring high discriminative power where standard CNNs often suffer from information loss during successive downsampling.

**Key Contributions:**

* **VarroaNet** : A ShuffleNet-V2 backbone enhanced with custom Squeeze-and-Excitation (SE) channel attention.
* **Multi-resolution Feature Maps** : Strategic stride removal in downsampling blocks to amplify spatial detail for XAI.
* **Comprehensive XAI Evaluation** : 5 quantitative localization metrics (IoU@50, IoU@30, Pointing Game, Energy Inside, Distance Error) evaluated on bounding-box-annotated test images.

**Key Results:**
* **Phase 1 (684 runs):** VarroaNet r=8 ranked 1st among 19 architectures (97.28% ± 0.59%, 3-fold CV). Morphology-preserving resize was the dominant beneficial preprocessing factor (*d* = +1.00), while histogram normalization was significantly harmful (*d* = −0.95).
* **Phase 2 (864 runs):** 28×28 is the jointly optimal resolution for classification (+0.25 pp) and Grad-CAM++ localization (best on all five metrics). VarroaNet r=8 at 28×28 achieves the top combined rank (accuracy 97.26%, IoU@30 = 0.349, Pointing Game = 0.691). The recommended configuration with D+MR preprocessing achieved PG = 0.927.
* **SE reversal:** Full SE (r=4) is superior for localization only at 14×14, while r=8 is superior at 7×7, 28×28, and 56×56, demonstrating a resolution-dependent channel attention crossover.

---

## Architecture

 **VarroaNet Design Rationale** : Unlike conventional lightweight models, VarroaNet is engineered to mitigate the "vanishing feature" problem for small objects. By integrating a custom Squeeze-and-Excitation (SE) channel attention mechanism with a resolution-preserving stride strategy, the model selectively excites the structural signatures of mites while maintaining computational efficiency.

| **Architecture** | **7×7** | **14×14** | **28×28** | **56×56** | **Attention** |
| ---------------------- | -------------- | ---------------- | ---------------- | ---------------- | ------------------- |
| ShuffleNet-V2-x1.0     | O              | O                | O                | O                | None                |
| ShuffleNet-V2-x0.5     | O              | O                | O                | O                | None                |
| MobileNetV3-Small      | O              | O                | O                | O                | SE (built-in)       |
| EfficientNet-B0        | O              | O                | O                | O                | SE (built-in)       |
| **VarroaNet (r=4)**    | **O**    | **O**      | **O**      | **O**      | **Custom SE** |
| **VarroaNet (r=8)**    | **O**    | **O**      | **O**      | **O**      | **Custom SE** |
| **VarroaNet (r=16)**   | O              | O                | O                | O                | Custom SE           |

### Resolution Enhancement Methodology

High-resolution feature maps are achieved by disabling the downsampling mechanism in the deeper layers. For a given input size **$H \times W$**, the output resolution **$H_{out} \times W_{out}$** was controlled by modifying the stride **$s$** of the **$n$**-th downsampling block:

$$
H_{out} = \frac{H_{in}}{2^{(k-m)}}, \quad \text{where } m \in \{0, 1, 2, 3\} \text{ is the number of modified blocks.}
$$

**Implementation Detail:**

* **$7\times7$ (Base)** : No modification.
* **$14\times14$** : Last downsample block stride **$2 \rightarrow 1$**.
* **$28\times28$** : Last 2 downsample blocks stride **$2 \rightarrow 1$**.
* **$56\times56$** : Last 3 downsample blocks stride **$2 \rightarrow 1$**.

---

## Grad-CAM++ Implementation

The framework utilizes the true Grad-CAM++ formulation to handle the localization of multiple and small instances by using second-order gradients. The pixel-wise weights **$w_k^c$** for class **$c$** and feature map **$A^k$** are calculated as:

$$
w_k^c = \sum_{i,j} \alpha_{i,j}^{kc} \cdot \text{ReLU}\left(\frac{\partial Y^c}{\partial A_{i,j}^k}\right)
$$

where the coefficient **$\alpha_{i,j}^{kc}$** accounts for the higher-order derivatives of the class score **$Y^c$** with respect to the feature maps **$A$**.

**Target layers per architecture:**

| Architecture              | Target Layer     |
| ------------------------- | ---------------- |
| ShuffleNet-V2 / VarroaNet | `conv5`        |
| MobileNetV3-Small         | `features[-1]` |
| EfficientNet-B0           | `features[-1]` |

---

## Repository Structure

```
.
├── unified_experiment.py            # Main: training + Grad-CAM++ + XAI metrics (25 model configs)
├── preprocessing_xai_analysis.py    # Post-analysis: preprocessing factor impact (Cohen's d, Mann-Whitney U)
├── threshold_sensitivity_viz.py     # Post-analysis: XAI threshold sensitivity (Kendall's tau)
├── normalization.py                 # Data preprocessing: normalization
├── resize.py                        # Data preprocessing: aspect-ratio preserving / stretch resize
├── requirements.txt                 # Python dependencies
├── LICENSE
└── README.md
```

## Usage

### Full experiment (single command)

```bash
python unified_experiment.py --gpu 0
```

This runs three phases sequentially:

| Phase             | Description                              | Key Outputs                                            |
| ----------------- | ---------------------------------------- | ------------------------------------------------------ |
| **Phase 1** | 3-fold stratified cross-validation       | `kfold_summary.csv`, epoch logs                      |
| **Phase 2** | Single-split train + Grad-CAM++ heatmaps | Best/last weights,`.npy` heatmaps, inference latency |
| **Phase 3** | XAI metrics computation                  | `xai_localization.csv` (IoU, PG, Energy, Dist)     |

### Options

```bash
# Resume an interrupted run (skips completed model-dataset pairs)
python unified_experiment.py --gpu 0 --resume

# Run specific resolution group
python unified_experiment.py --gpu 1 --resolution 56x56

# Run specific models
python unified_experiment.py --gpu 0 --models varroanet varroanet_56

# Run specific phase only
python unified_experiment.py --gpu 1 --phase kfold
python unified_experiment.py --phase xai

# List all available models
python unified_experiment.py --list-models
```

### Post-experiment analysis

```bash
# Preprocessing factor impact on XAI (Cohen's d, Kruskal-Wallis, Mann-Whitney U)
python preprocessing_xai_analysis.py

# Threshold sensitivity analysis (Kendall's tau rank stability)
python threshold_sensitivity_viz.py
```

---

## Training Configuration

| Parameter          | Value                                      |
| ------------------ | ------------------------------------------ |
| Optimizer          | AdamW (lr=0.01, weight_decay=0.01)         |
| Scheduler          | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping     | patience=12, min_delta=0.0005              |
| Max epochs         | 500                                        |
| K-fold             | StratifiedKFold (k=3, seed=42)             |
| Single-split ratio | 70% train / 20% val / 10% test             |
| Input size         | 224 x 224                                  |
| Batch size         | 32 (7x7, 14x14, 28x28) / 8 (56x56)        |

## Output Structure

All outputs are generated under `outputs/`:

```
outputs/
├── logs/
│   ├── {model_name}/                    # Per-model logs
│   │   ├── epoch_log.csv                #   Every epoch: loss, acc, lr, patience, ...
│   │   ├── kfold_summary.csv            #   Per-fold: train/val/test metrics + confusion matrix
│   │   └── singlesplit_summary.csv      #   Single-split metrics + inference latency
│   ├── merged_epoch_log.csv             # All models consolidated
│   ├── merged_kfold_summary.csv
│   ├── merged_kfold_agg.csv             # Mean +/- std per model x dataset
│   ├── merged_singlesplit_summary.csv
│   ├── xai_localization.csv             # Localization XAI metrics (IoU@50, IoU@30, PG, Energy, Dist)
│   └── model_metadata.csv              # Parameter counts, attention type
├── weights/d{1-12}/                     # Best & last model weights (.pth)
├── weights_kfold/d{1-12}/               # Per-fold best weights
├── gradcam/d{1-12}/{model}/             # Grad-CAM++ heatmaps (.npy) + metadata (.json)
└── analysis/                            # Post-experiment figures & CSV tables
```

## XAI Metrics

### Quantitative Localization Metrics (requires bounding-box annotations)

| Metric        | Description                                                         |
| ------------- | ------------------------------------------------------------------- |
| IoU@50        | Intersection over Union with ground-truth bounding box (threshold 0.50) |
| IoU@30        | Intersection over Union with ground-truth bounding box (threshold 0.30) |
| Pointing Game | Whether CAM peak falls inside the bounding box                      |
| Energy Inside | Proportion of total CAM energy within the bounding box              |
| Distance Error| Normalized distance from CAM peak to bounding box center            |

---

## Key Experimental Results (25 Phase 2 Configurations)

| Architecture        | Res.  | Acc (%) | IoU@30 | PG    | Energy | Dist↓ |
|---------------------|-------|---------|--------|-------|--------|-------|
| VarroaNet (r=8)     | 28×28 | 97.26   | 0.349  | 0.691 | 0.385  | 0.045 |
| ShuffleNet-V2-x0.5  | 28×28 | 97.24   | 0.368  | 0.513 | 0.386  | 0.064 |
| VarroaNet (r=4)     | 14×14 | 97.31   | 0.277  | 0.523 | 0.292  | 0.058 |
| VarroaNet (r=4)     | 28×28 | 97.22   | 0.301  | 0.483 | 0.327  | 0.063 |
| ShuffleNet-V2-x1.0  | 14×14 | 97.34   | 0.160  | 0.359 | 0.164  | 0.081 |

> The top-ranked configuration (VarroaNet r=8, 28×28, D+MR preprocessing) achieved **PG = 0.927** on the D+MR dataset, meaning the Grad-CAM++ peak fell within the mite bounding box in 92.7% of test images.

---

## Requirements

Python 3.8+ with CUDA-compatible GPU recommended.

```bash
pip install -r requirements.txt
```

See [requirements.txt](requirements.txt) for the full list of dependencies.

## Dataset

This repository contains **code only**. The Varroa mite image datasets used in the paper are not publicly released. To use this framework with your own data, organize images as a binary classification dataset:

```
dataset/{dataset_name}/
├── bee/
│   ├── img_001.png
│   └── ...
└── mite/
    ├── img_001.png
    └── ...
```

The experiment uses 12 dataset variants in a 2 × 2 × 3 factorial design:

| Factor        | Levels                                                                  |
| ------------- | ----------------------------------------------------------------------- |
| Source        | Original, Deblurred (MPRNet)                                            |
| Normalization | None, Histogram-equalized                                               |
| Resizing      | None (zero-pad only), MR (aspect-ratio preserved + pad), NR (stretched) |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lee2026varroanet,
  title     = {Interpretable Deep Learning for Varroa Mite Detection:
               Multi-Resolution Grad-CAM++ Analysis with Channel Attention
               and Preprocessing Sensitivity},
  author    = {Lee, Hong-Gu and Shin, Jeong-Yong and Han, Woon-Tak and
               Kim, Su-Bae and Kim, Min-Jee and Kim, Giyoung and Mo, Changyeun},
  journal   = {Agronomy},
  volume    = {},
  pages     = {},
  year      = {2026},
  doi       = {},
  publisher = {MDPI},
  note      = {Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
