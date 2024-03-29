# 🤖 Unlearning_LLM
This repo contains code and data for the paper "[Machine Unlearning of Pre-trained Large Language Models](https://arxiv.org/abs/2402.15159)"

[Paper](https://arxiv.org/pdf/2402.15159.pdf) | [Dataset](https://huggingface.co/datasets/llmunlearn/unlearn_dataset)

The complete code will be updated soon.

## 🌟 Abstract

<details><summary>Abstract</summary>

This study investigates the concept of the `right to be forgotten' within the context of large language models (LLMs). We explore machine unlearning as a pivotal solution, with a focus on pre-trained models--a notably under-researched area. Our research delineates a comprehensive framework for machine unlearning in pre-trained LLMs, encompassing a critical analysis of seven diverse unlearning methods. Through rigorous evaluation using curated datasets from arXiv, books, and GitHub, we establish a robust benchmark for unlearning performance, demonstrating that these methods are over $10^5$ times more computationally efficient than retraining. Our results show that integrating gradient ascent with gradient descent on in-distribution data improves hyperparameter robustness. We also provide detailed guidelines for efficient hyperparameter tuning in the unlearning process. Our findings advance the discourse on ethical AI practices, offering substantive insights into the mechanics of machine unlearning for pre-trained LLMs and underscoring the potential for responsible AI development.

</details>

## 📊 Dataset
We collect and provide the **unlearn_dataset**, which serves as a benchmark for evaluating unlearning methodologies in pre-trained large language models across diverse domains, including arXiv, GitHub. Access our **unlearn_dataset** directly on [Hugging Face](https://huggingface.co/datasets/llmunlearn/unlearn_dataset).

### 🔍 Loading the datasets

To load the dataset:

```python
from datasets import load_dataset
dataset = load_dataset("llmunlearn/unlearn_dataset", name="arxiv", split="forget")
```
* Available configuration names and corresponding splits:
  - `arxiv`: `forget, approximate, retain`
  - `github`: `forget, approximate, retain`
  - `general`: `evaluation, retain`

## ✈️ How to run
### Environment Setup
```
git clone https://github.com/yaojin17/Unlearning_LLM.git
cd Unlearning_LLM
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
pip install -e .
pip install transformers==4.35.0
pip install wandb
```
### Download Yi-6B model
```
mkdir models
cd models
git lfs install
git clone https://huggingface.co/01-ai/Yi-6B
```
### Prepare tokenized datasets
```
cd utils
python save_tokenized_dataset.py --tokenizer_name_or_path ../../models/Yi-6B
python ascent_plus_descent_tokenizer.py --tokenizer_name_or_path ../../models/Yi-6B
```

## ⭐ Citation Information

If you find this code or dataset useful, please consider citing our paper:

```bib
@article{yao2024machine,
  title={Machine Unlearning of Pre-trained Large Language Models},
  author={Yao, Jin and Chien, Eli and Du, Minxin and Niu, Xinyao and Wang, Tianhao and Cheng, Zezhou and Yue, Xiang},
  journal={arXiv preprint arXiv:2402.15159},
  year={2024}
}
```

### Contact
Feel free to reach out if you have any questions. [Jin Yao](mailto:rry4fg@virginia.edu), [Eli Chien](mailto:ichien6@gatech.edu), [Xiang Yue](mailto:yue.149@osu.edu)