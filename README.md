# MM-ShortVideo-Rec

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Abstract
A recommendation framework designed for short video platforms is introduced. The system utilizes a Transformer-based architecture to capture dynamic preferences of users and a neural matrix factorization component to model static interactions between users and items. The approach processes historical interaction sequences to improve recommendation accuracy.

## Methodology
The architecture, denoted as Sequential Neural Matrix Factorization or SeqNeuMF, integrates a sequential encoder with Neural Matrix Factorization. The sequence of historical user interactions is processed by a self-attention mechanism to generate a dynamic representation of the user. This dynamic representation is concatenated with a static embedding of the user. The combined features are passed through multi-layer perceptrons to predict the interaction probability with target items.


Detailed architectural diagrams of the implemented models are provided below:

### Standard NeuMF
![NeuMF Architecture](assets/neumf.png)

### Multimodal NeuMF (M-NeuMF)
![M-NeuMF Architecture](assets/m-neumf.png)

### Multimodal Sequential NeuMF (M-SeqNeuMF)
![M-SeqNeuMF Architecture](assets/m-seqmf.png)

## Environment and Setup
The environment is managed by the ``uv'' package manager. The dependencies are specified in the ``pyproject.toml'' and ``uv.lock'' files to ensure reproducibility. The environment setup is performed by the following commands.

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Data Preparation
The MicroLens-5k dataset is utilized for empirical validation. The raw dataset is downloaded and extracted into the ``data/microlens-5k'' directory. The download process is automated by a utility script.

```bash
python utils/download_data.py
```

The input data must contain user interaction sequences. Data preprocessing is handled internally by the data loader module before being passed to the model.

## Usage

The pipeline is executed via the command line interface. The execution is separated into training and evaluation phases.

### Training

The framework supports multiple model configurations. Use the flags below to toggle between variants:

#### 1. Standard NeuMF (Collaborative Filtering Only)
```bash
python src/train.py --no_visual
```

#### 2. Multimodal NeuMF (M-NeuMF)
```bash
python src/train.py
```

#### 3. Multimodal Sequential NeuMF (M-SeqNeuMF)
```bash
python src/train.py --use_seq_user
```

*Note: Use `--use_cuda` if a GPU is available.*

### Evaluation (Inference)

To generate recommendations for a specific user, provide the checkpoint path and the user ID.

#### 1. NeuMF
```bash
python src/inference.py --no_visual --user_id "user_id_here" --checkpoint "checkpoints/neumf_model.model"
```

#### 2. M-NeuMF
```bash
python src/inference.py --user_id "user_id_here" --checkpoint "checkpoints/m_neumf_model.model"
```

#### 3. M-SeqNeuMF
```bash
python src/inference.py --use_seq_user --user_id "user_id_here" --checkpoint "checkpoints/m_seqmf_model.model"
```

## Project Structure

The directory tree of the source code is presented below.

```text
MM-ShortVideo-Rec/
├── data/
│   └── microlens-5k/
├── src/
│   ├── data.py
│   ├── engine.py
│   ├── inference.py
│   ├── metrics.py
│   ├── mlp.py
│   ├── neumf.py
│   ├── seqneumf.py
│   └── train.py
├── utils/
│   └── download_data.py
├── pyproject.toml
└── README.md
```

## Results

Empirical evaluations are conducted to compare the performance of the proposed sequential and multimodal architectures against standard baselines. Metrics such as Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) are utilized to assess the top-K recommendation quality.

## Acknowledgements

The MicroLens dataset is acknowledged for providing the short video interaction data used in the experiments.

## License

This project is licensed under the MIT License.
