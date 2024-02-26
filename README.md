# Unlearning_LLM
This repo contains code and data for the paper "Machine Unlearning of Pre-trained Large Language Models"

The complete code and data will be updated soon.

## Abstract

<details><summary>Abstract</summary>

This study investigates the concept of the `right to be forgotten' within the context of large language models (LLMs). We explore machine unlearning as a pivotal solution, with a focus on pre-trained models--a notably under-researched area. Our research delineates a comprehensive framework for machine unlearning in pre-trained LLMs, encompassing a critical analysis of seven diverse unlearning methods. Through rigorous evaluation using curated datasets from arXiv, books, and GitHub, we establish a robust benchmark for unlearning performance, demonstrating that these methods are over $10^5$ times more computationally efficient than retraining. Our results show that integrating gradient ascent with gradient descent on in-distribution data improves hyperparameter robustness. We also provide detailed guidelines for efficient hyperparameter tuning in the unlearning process. Our findings advance the discourse on ethical AI practices, offering substantive insights into the mechanics of machine unlearning for pre-trained LLMs and underscoring the potential for responsible AI development.

</details>

## Environment Setup
```
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
pip install -e .
pip install transformers==4.35.0
pip install wandb
```


## Citation Information

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