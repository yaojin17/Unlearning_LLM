# Unlearning_LLM
This repo contains code for the paper "Machine Unlearning of Pre-trained Large Language Models"



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

## Tokenize datasets
```bash
cd llm_unlearn/utils
python save_tokenized_dataset.py --tokenizer_name_or_path <your-model-path>
python ascent_plus_descent_tokenizer.py --tokenizer_name_or_path <your-model-path>
```
## Run unlearning experiments
```bash
cd llm_unlearn
torchrun --nproc_per_node=8 --master_port=20001  run_unlearn.py   \
    --target_model_name_or_path <your-model-path>  \
    --per_device_train_batch_size 1     \
    --per_device_eval_batch_size 4   \
    --gradient_accumulation_steps 85 \
    --do_unlearn  \
    --output_dir ./output \
    --overwrite_output_dir     \
    --num_train_epochs 1    \
    --logging_steps 1     \
    --learning_rate 3e-5     \
    --warmup_ratio 0.03 \
    --overwrite_cache \
    --save_steps 8 \
    --save_total_limit 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --bf16 True \
    --tf32 True \
    --weight_decay 0. \
    --output_sufix wd0 \
    --lr_scheduler_type "cosine" \
    --unlearn_method random_label \
    --completely_random True \
    --domain github
```
- Please replace `<your-model-path>` to your model path. 
- The options of domains include `github, arxiv`. 
- The unlearning methods and corresponding arguments include:
  - `--unlearn_method gradient_ascent`
  - `--unlearn_method random_label --completely_random True`
  - `--unlearn_method random_label --top_k 1 --rm_groundtruth True`
  - `--unlearn_method ascent_plus_descent`
  - `--unlearn_method ascent_plus_kl_divergence`
