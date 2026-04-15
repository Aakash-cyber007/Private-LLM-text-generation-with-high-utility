<!-- # InvisibleInk-Experiments -->

<div align="center" style="font-family: charter;">
<h1>InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy</h1>

[![arxiv](https://img.shields.io/badge/InvisibleInk-2507.02974-red?label=InvisibleInk&link=https%3A%2F%2Farxiv.org%2Fabs%2F2507.02974)](https://arxiv.org/abs/2507.02974)

[Vishnu Vinod](https://vishnuvinod8.github.io/), [Krishna Pillutla](https://krishnap25.github.io/), [Abhradeep Thakurta](https://athakurta.squarespace.com/)  

</div>

This repository contains the code and the scripts to reproduce the experiments in the paper *InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy* published at **NeurIPS 2025**.

Our paper introduces InvisibleInk, a highly scalable compute-efficient framework for differentially private synthetic text generation. InvisibleInk privatizes LLM inference by casting the decoding (next-token prediction) step in autoregressive language models as an instance of the canonical exponential mechanism for differential privacy, with two innovations:

1. **Difference Clipping**: A novel clipping function that isolates and clips only the sensitive information in model logits prior to decoding using *10x smaller clipping norms* compared to prior works.

2. **Top-k+ sampling**: An extension of truncated sampling approaches adopted by the wider NLP community to sample tokens from an expanded top-k set of the public logits *without additional privacy cost*.

The repository contains scripts for private text generation and evaluation to reproduce all major results, presented in the paper. The rest of the README describes how to access the datasets used in this paper, install required libraries and generate private synthetic text.

### InvisibleInk is now available as a standalone Python package!!
Check out the project repository [here](https://github.com/cerai-iitm/invisibleink) or install it via `pip install invink`! 

---

## Datasets

We use the following datasets for our experiments:

1. **MIMIC IV Clinical Notes Dataset (MIMIC):** The MIMIC Notes dataset contains anonymized patient discharge notes. The dataset can be accessed [here](https://physionet.org/content/mimic-iv-note/2.2/), after obtaining the requisite credentials, signing the Data Use Agreement, and completing the reqired training programs. We note that we do not release actual samples from the data or synthetic generations, to comply with the dataset license.

2. **Yelp Open Dataset (YELP):** The Yelp dataset contains customer reviews of a large number of businesses. The reviewers are labelled by business category and by review score. The full dataset can be accessed by following the instructions [here](https://github.com/AI-secure/aug-pe/tree/main/data).

3. **Text Anonymization Benchmark (TAB):** This dataset contains details about legal cases handled by the European Court for Human Rights. The dataset is available online and can be accessed [here](https://github.com/NorskRegnesentral/text-anonymization-benchmark).

---

## Installation & Setup

The codebase is written entirely in Python. Important dependencies are listed below:

- Python >= 3.8
- Pytorch >= 2.6.0
- Huggingface Transformers >= 4.57.0
- sentence-transformers >= 3.4.1
- accelerate >=1.5.2

We recommend using a [conda environment](https://www.anaconda.com/docs/getting-started/miniconda/main) with Python 3.13. Further instructions are given below.

---

## Clone the repository and setup the environment

Clone this repository and open the folder. All following bash commands described here are to be run within this folder.
```bash
git clone https://github.com/cerai-iitm/InvisibleInk-Experiments.git
cd InvisibleInk-Experiments
```

To setup the environment, run:
```bash
conda env create --file environment.yml
# activate the environment
conda activate invink-expts
```

Further, to install all required python packages using pip run the following:
```bash
pip install -r requirements.txt
```

---

## Download and prepare datasets

The required dataset folders (TAB and YELP) can be downloaded by running the following script:
```bash
gdown https://drive.google.com/drive/folders/1vetnesv9xx0uMYQFcrsEwlG7J9j-zeCT -O ./ --folder
```
The above code downloads the YELP and TAB datasets from [this drive folder](https://drive.google.com/drive/folders/1vetnesv9xx0uMYQFcrsEwlG7J9j-zeCT?usp=drive_link) and also creates an empty folder for the MIMIC dataset. Users can obtain access to the MIMIC dataset and download it by following the steps described [here](https://physionet.org/content/mimic-iv-note/2.2/). After downloading the dataset and unzipping, the `discharge.csv` file should be placed within the `./data/mimic/` folder before running the code that follows.

We now preprocess the dataset by running the following script:
```bash
bash scripts/dataset.sh
```

This creates the required datasets which are used for synthetic data generation in our experiments. If the users do not have access to the MIMIC dataset they may run `scripts/dataset.sh` with the `--nomimic` flag.

---

## Code Execution

We note that it is necessary to set the huggingface access token in the runtime environment prior to code execution. Users may follow the steps given [here]() to obtain their huggingface access token (a read only token should suffice). To set the login token in the environment run the following code block:

```bash
export HF_TOKEN="add_token_here"
```

Users may also make this change permanent by writing the above line to the `.bashrc` file but we recommend ample caution while making any permanent changes to the `.bashrc` file.

To execute all the scripts for private synthetic text generation, run the following scripts:

```bash
bash scripts/amin.sh
bash scripts/adapmixed.sh
bash scripts/invink.sh
```

This creates the synthetic data for the MIMIC, Yelp and TAB datasets. If the users do not have access to the MIMIC dataset, they may instead run the same scripts with the `--nomimic` flag to skip MIMIC generations, as described below:

```bash
bash scripts/amin.sh --nomimic
bash scripts/adapmixed.sh --nomimic
bash scripts/invink.sh --nomimic
```

---

## Result Storage

The results are all stored in the `./results` folder. Every particular hyperparameter setting is stored in the folder `./results/{dataset}/{method}/{hyperparameter setting}`. Within each such result folder, for a given random seed for generation `seed`, the following files are created:

- `data_{seed}.csv`: contains all text, token sequences and other generation statistics
- `datagen_{seed}.log`: contains generation and evaluation logs for all metrics
- `results_{seed}.pickle`: contains all results stored in a pickled dictionary

---

**Yelp Downstream Classification**

To evaluate the synthetic data on the downstream task of finetuning a business-class and review-score classifier, the following script can be run:
```bash
bash scripts/classify_yelp.sh
```
This creates a `robert_yelp50class_{model}.log` file which logs all reevant evaluation metrics. 

---

## Memory and Compute requirements

We note that all scripts and experiments were tested on NVIDIA L40S GPUs with 48GB memory. We note that users attempting to reproduce these experiments on smaller GPUs, may run into out-of-memory errors. To fix this error, users can decrease the value of the `minibatch` argument in each generation script. For example to run the code on a GPU with 24GB RAM, it is recommended to therefore use `minibatch=8` (decrease approximately linearly).

We note that each setting of generation for InvisibleInk takes between 2 to 4 hours to complete. Larger models and baselines take between 8 to 10 hours per setting. The total compute time across all settings (and random seeds) exceeds 4000 GPU hours on an NVIDIA L40S GPUs. 

---

## Citations

If you find this repository useful, or you use it in your research, please cite the following papers:

```
@inproceedings{vinod2025invisibleink,
  author       = {Vishnu Vinod and Krishna Pillutla and Abhradeep Thakurta},
  title        = {{InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy}},
  booktitle    = {NeurIPS},
  year         = {2025},
}
```

---

## Ackowledgements

This work was supported by the Post-Baccalaureate Fellowship at the Centre for Responsible AI (CeRAI), IIT Madras, the startup compute grant of the Wadhwani School of Data Science & AI (WSAI), IIT Madras, and faculty research awards.