# DBSegNet

<p align="center">
  <strong>Dual-branch complementary feature learning for retrogressive thaw slump segmentation</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> | 
  <a href="#data-preparation">Data</a> | 
  <a href="#training">Training</a> | 
  <a href="#inference">Inference</a>
</p>

DBSegNet is a semantic segmentation network developed for extracting
retrogressive thaw slumps (RTSs) from high-resolution remote-sensing imagery.
It combines CNN and Transformer representations through cross-branch feature
exchange, dense complementary gating, Stage-3 bidirectional cross-attention,
and a bidirectional multi-scale decoder.

## Result Showcase

<p align="center">
  <img src="img/定性图.png" width="900" alt="DBSegNet qualitative results">
</p>

## Highlights

- Dual CNN-Transformer encoder for local detail and long-range context.
- Cross-branch residual exchange at multiple feature stages.
- Learnable dense complementary feature gating (LGF).
- Bidirectional cross-attention at Stage 3.
- Top-down and bottom-up multi-scale feature decoding.
- Reproducible training and inference configuration.

## Repository Structure

```text
DBSegNet/
|-- train.py                 # Training configuration and entry point
|-- config.py                # Typed configuration and path overrides
|-- predict.py               # Single-image inference
|-- nets/
|   |-- DBSegNet.py          # Final DBSegNet architecture
|   |-- encoder.py           # Dual-branch encoder and feature interaction
|   `-- DBSegNet_legacy.py   # Legacy implementation retained for reference
|-- training/                # Training loop, optimizer and checkpoint helpers
|-- utils/                   # Data loading, metrics, callbacks and losses
|-- splits/                  # Optional split files (no imagery)
`-- verify_reproduction.py   # Protocol and checkpoint consistency audit
```

Datasets, checkpoints, logs, and prediction outputs are intentionally kept
outside this repository.

## Installation

Python 3.10 and a CUDA-capable GPU are recommended. The reference environment
uses PyTorch 2.1.0.

```bash
conda create -n dbsegnet python=3.10 -y
conda activate dbsegnet
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA driver when the default pip
wheel is unsuitable. For a headless server, `opencv-python-headless` may replace
`opencv-python`.

## Data Preparation

`dataset_path` must point to the directory **containing** `VOC2007`, rather
than to `VOC2007` itself.

```text
DATASET_PATH/
`-- VOC2007/
    |-- JPEGImages/
    |   `-- 000001.tif
    |-- SegmentationClass/
    |   `-- 000001.tif
    `-- ImageSets/
        `-- Segmentation/
            |-- train.txt
            `-- val.txt
```

Images and masks must share the same filename stem. Masks use `0` for
background and `1` (or any positive value for binary conversion) for RTS.

### Dataset Download

The dataset and optional checkpoints will be released through Baidu Netdisk:

```text
Baidu Netdisk: [LINK TO BE ADDED]
Extraction code: [CODE TO BE ADDED]
```

## Training

Edit the path settings and experiment parameters near the top of `train.py`:

```python
DATA = DataConfig(
    dataset_path="/path/to/dataset_root",
    split_dir="/path/to/split_directory",
    input_shape=(512, 512),
)

TRAIN = TrainConfig(
    epochs=100,
    batch_size=4,
    save_dir="/path/to/output_logs",
)
```

Then start training:

```bash
python -u train.py
```

The reference protocol uses seed 11, normal initialization, FP16, AdamW,
CE + Dice loss, cosine learning-rate decay, `planet_mild` augmentation, and no
augmentation during the final 20 epochs. Validation mIoU is calculated every
five epochs.

## Inference

Run binary segmentation on a single image:

```bash
python predict.py \
  --checkpoint /path/to/best_miou_weights.pth \
  --input /path/to/image.tif \
  --output /path/to/prediction.png
```

The output is an 8-bit binary mask. The current lightweight inference entry
point resizes one image to the configured input size; large georeferenced
rasters should be processed with an external tiled-inference workflow.

## Reproducibility

Run the audit utility before reproducing an archived experiment:

```bash
python verify_reproduction.py
```

The utility checks model/checkpoint compatibility, the forward fingerprint,
critical training components, and train/validation split hashes. Exact metrics
also depend on using the same imagery, masks, split files, CUDA stack, and
random seed.

## Citation

If this repository is useful for your research, please cite the associated
paper. The BibTeX entry will be added after publication.

```bibtex
@article{DBSegNet,
  title   = {To be updated},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2026}
}
```

## License

Please refer to the repository license before redistributing the code or data.
