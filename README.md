# LiveEdit

Source code for paper [Uncovering Locality Failures in Multimodal Editing through Dynamic Evaluation] 
We apply the Dynamic Evaluation with in the dataset/vllm.py.


# Setup
1. Please download the EVQA and VLKEB datasets from the URL provided in [1] and [2], and place the related folders in the `data/easy-edit-mm` directory.
2. Please download the VLKEB dataset from the URL provided in [2] and place the related folders in the `data/VLKEB` directory.
3. Please modify the `ROOT_PATH` in `utils/GLOBAL.py` to the absolute path of the current directory, and update `model_path_map` to the absolute paths of each backbone's weights.

# Evaluation data
 You could get the evaluation data form `/dataset/vllm.py`, and evalution with it. 
 The `vlkeb_embeddings_collect.pkl` or `vqa_embeddings_llava.pkl` is got by `encode_ike_facts_multimodal()` in `\easyeditor\models\ike\util.py`.
 You could also replace with your own retriever.

# Train editors
Please follow the script below to train an editor that requires training:

For MEND, we training use the easyeditor, please refer to [1].

For others, we training use [2]:
```
python train_vllm_editor.py -en FT_VL -mn blip2 -dna EVQA -bs 8 -dvc "cuda:0" -edvc 0 -lkpt None -tnp EVQA -eps 50 -sci 500
```

# Evaluate editors
Please follow the script below to evaluate the editors. If an editor does not require training, please set `-ckpt` to `None`."

```
python test_vllm_edit.py -en mend -mn blip2 -sen 1000 -dvc "cuda:0" -dn "EVQA" -ckpt the_ckpt_for_mend
```


# Reference
[1] Can We Edit Multimodal Large Language Models? EMNLP 2023

[2] VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark. NeurIPS 2024, Datasets and Benchmarks Track

[3] Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts

This repo is based on [1] and [3].