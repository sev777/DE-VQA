# from ..dataset.processor.blip_processors import BlipImageEvalProcessor
# from .editor import BaseEditor
# import os.path
# from typing import Optional, Union, List, Tuple, Dict
# from time import time
# from torch.utils.data import Dataset
# from tqdm import tqdm
# import json
# import torch
# import logging
# import numpy as np
# from PIL import Image
#
# import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import LlamaTokenizer, LlamaForCausalLM
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# from transformers import GPT2TokenizerFast, GPT2Tokenizer
# from ..util.globals import *
# from .singleton_editor import SingletonEditor
# from .batch_editor import BatchEditor
# from ..evaluate import (compute_icl_multimodal_edit_quality,
#                         compute_multimodal_edit_results,
#                         compute_multimodal_edit_results_demo)
# from ..util import nethook
# from ..util.hparams import HyperParams
# from ..util.alg_dict import *
#
# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
#
# LOG = logging.getLogger(__name__)
#
#
# def make_logs():
#
#     f_h, s_h = get_handler("logs/", log_name='run.log')
#     LOG.addHandler(f_h)
#     LOG.addHandler(s_h)
#
#
# class MultimodalEditor:
#     """Multimodal editor for all methods"""
#
#     @classmethod
#     def from_hparams(cls, hparams: HyperParams):
#
#         return cls(hparams)
#
#     def __init__(self,
#                 hparams: HyperParams,
#                  ):
#
#         assert hparams is not None or print('Error: hparams is None.')
#
#         self.model_name = hparams.model_name
#         self.apply_algo = ALG_MULTIMODAL_DICT[hparams.alg_name]
#         self.alg_name = hparams.alg_name
#
#         make_logs()
#
#         LOG.info("Instantiating model")
#
#         if type(self.model_name) is str:
#             if hparams.model_name == "blip2":
#                 from ..trainer.blip2_models import Blip2OPT
#
#                 model = Blip2OPT(
#                     vit_model="eva_clip_g",
#                     img_size=364,
#                     use_grad_checkpoint=True,
#                     vit_precision="fp32",
#                     freeze_vit=True,
#                     opt_model=hparams.name,
#                     state_dict_file=hparams.state_dict_file,
#                     qformer_name_or_path=hparams.qformer_name_or_path,
#                     qformer_checkpoint=hparams.qformer_checkpoint
#                 )
#             elif hparams.model_name == "minigpt4":
#                 from ..trainer.blip2_models import MiniGPT4
#
#                 model = MiniGPT4(
#                     vit_model="eva_clip_g",
#                     qformer_checkpoint=hparams.qformer_checkpoint,
#                     img_size=364,
#                     use_grad_checkpoint=True,
#                     vit_precision="fp32",
#                     freeze_vit=True,
#                     llama_model=hparams.name,
#                     state_dict_file=hparams.state_dict_file,
#                     qformer_name_or_path=hparams.qformer_name_or_path,
#                     pretrained_ckpt=hparams.pretrained_ckpt,
#                 )
#             elif hparams.model_name == "llava":
#                 from ..trainer.llava.model.builder import load_pretrained_model
#                 model = load_pretrained_model(model_path=hparams.name, device=hparams.device)
#             elif "qwen-vl" in hparams.model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(hparams.name, trust_remote_code=True, pad_token='<|endoftext|>')
#                 model = AutoModelForCausalLM.from_pretrained(
#                     hparams.name, trust_remote_code=True
#                 )
#
#             elif "owl-2" in hparams.model_name.lower():
#                 from ..trainer.mPLUG_Owl2.mplug_owl2.model.builder import load_pretrained_model
#                 from ..trainer.mPLUG_Owl2.mplug_owl2.model.modeling_mplug_owl2 import replace_llama_modality_adaptive
#                 replace_llama_modality_adaptive()
#                 tokenizer , model, _, _ = load_pretrained_model(hparams.name, None, 'mplug_owl2', load_8bit=False, load_4bit=False, device=f"cuda:{hparams.device}")
#                 for param in model.parameters():
#                     param.requires_grad = True
#
#             self.model = model
#             # Get tokenizer and vis_processor
#             if hparams.model_name in  ["blip2", "minigpt4"]:
#                 vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
#                 self.vis_tok = vis_processor
#             elif hparams.model_name == "llava":
#                 vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
#                 self.vis_tok = lambda image: vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
#             elif hparams.model_class ==  "qwen-vl":
#                 vis_processor = BlipImageEvalProcessor(image_size=448, mean=None, std=None)
#                 self.vis_tok = vis_processor
#             elif hparams.model_name.lower() == "owl-2":
#                 from transformers.models.clip.image_processing_clip import CLIPImageProcessor
#                 vis_processor = CLIPImageProcessor.from_pretrained(hparams.name, trust_remote_code=True)
#                 self.vis_tok = vis_processor
#             else:
#                 raise NotImplementedError(f"Model {hparams.model_name} not supported")
#
#             if (hparams is not None and hasattr(hparams, 'tokenizer_name')):
#                 tok_name = (
#                     hparams.tokenizer_name
#                     if hparams.tokenizer_name is not None
#                     else hparams.name
#                 )
#                 tokenizer = getattr(transformers, hparams.tokenizer_class).from_pretrained(
#                     tok_name
#                 )
#                 if tokenizer.pad_token == None or tokenizer.pad_token == '':
#                     tokenizer.pad_token = tokenizer.eos_token
#                 self.tok = tokenizer
#             else:
#                 raise NotImplementedError('missing tokenizer_name in hparams')
#         else:
#             self.model, self.tok = self.model_name
#         # device_map = {
#         #     0: [_ for _ in range(0, 16)],
#         #     1: [_ for _ in range(16, 32)],
#         #     2: [_ for _ in range(32, 48)]
#         # }
#         # self.model.parallelize(device_map=device_map)
#         self.model.to(f'cuda:{hparams.device}')
#
#         self.hparams = hparams
#         self.vis_root = hparams.coco_image
#         self.rephrase_root = hparams.rephrase_image
#
#     def edit(self,
#             prompts: Union[str, List[str]],
#             targets: Union[str, List[str]],
#             image: Union[str, List[str]],
#             rephrase_prompts: Optional[Union[str, List[str]]] = None,
#             rephrase_image: Optional[Union[str, List[str]]] = None,
#             locality_inputs: Optional[dict] = None,
#             keep_original_weight=False,
#             verbose=True,
#             **kwargs
#             ):
#         """
#         `prompts`: list or str
#             the prompts to edit
#         `targets`: str
#             the expected outputs
#         `image`: dict
#             for multimodal
#         """
#         if isinstance(prompts, List):
#             assert len(prompts) == len(targets) == len(image)
#         else:
#             prompts, targets, image = [prompts,], [targets,], [image,]
#
#         if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
#             self.hparams.batch_size = 1
#
#         requests = self._prepare_requests(prompts, targets, image, rephrase_prompts, rephrase_image, locality_inputs,
#                                           **kwargs)
#
#         if hasattr(self.hparams, 'batch_size') :
#                assert self.hparams.batch_size == 1 or \
#                       print(f'Single Edit, pls set the batch_size to 1....')
#
#         # if not os.path.exists(RESULTS_DIR):
#         #     os.mkdir(RESULTS_DIR)
#         # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
#         # if not os.path.exists(base_case_path):
#         #     os.mkdir(base_case_path)
#         # print(f"Results will be stored at {base_case_path}")
#         all_metrics = []
#         for i, request in enumerate(requests):
#             start = time()
#
#             if self.alg_name == 'IKE':
#                 assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
#                 edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
#                     self.model,
#                     self.tok,
#                     request,
#                     self.hparams,
#                     copy=False,
#                     return_orig_weights=True,
#                     keep_original_weight=keep_original_weight,
#                     train_ds=kwargs['train_ds']
#                 )
#                 exec_time = time() - start
#                 LOG.info(f"Execution {i} editing took {exec_time}")
#                 start = time()
#                 metrics = {
#                     'case_id': i,
#                     # "requested_rewrite": request,
#                     "time": exec_time,
#                     "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
#                                                      request, self.hparams.device),
#                     "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
#                                                      request, self.hparams.device, pre_edit=True)
#                 }
#                 metrics['pre'].pop('locality_acc')
#                 metrics['pre'].pop('locality_image_acc')
#                 metrics['pre'].pop('portability_acc')
#
#                 LOG.info(f"Evaluation took {time() - start}")
#
#                 if verbose:
#                     LOG.info(
#                         f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
#                     )
#
#                 all_metrics.append(metrics)
#             else:
#                 edited_model, weights_copy = self.apply_algo(
#                     self.model,
#                     self.tok,
#                     [request],
#                     self.hparams,
#                     copy=False,
#                     return_orig_weights=True,
#                     keep_original_weight=keep_original_weight,
#                     train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
#                 )
#                 exec_time = time() - start
#                 LOG.info(f"Execution {i} editing took {exec_time}")
#
#                 start = time()
#                 metrics = {
#                     'case_id': i,
#                     # "requested_rewrite": request,
#                     "time": exec_time,
#                     "post": compute_multimodal_edit_results(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
#                 }
#                 if self.alg_name == 'KN':
#                     with torch.no_grad():
#                         weights_copy() # unpatch_fn
#                 else:
#                     with torch.no_grad():
#                         for k, v in weights_copy.items():
#                             nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
#                 metrics["pre"] = compute_multimodal_edit_results(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device)
#                 if 'locality_output' in metrics['post'].keys():
#                     assert len(metrics['post']['locality_output']) == \
#                             len(metrics['pre']['locality_output'])
#                     metrics['post']['locality_acc'] = \
#                         np.mean(np.equal(metrics['post']['locality_output'],
#                                             metrics['pre']['locality_output']))
#                     metrics['post'].pop('locality_output')
#                     metrics['pre'].pop('locality_output')
#
#                 if 'multimodal_locality_output' in metrics['post'].keys():
#                     assert len(metrics['post']['multimodal_locality_output']) == \
#                             len(metrics['pre']['multimodal_locality_output'])
#                     metrics['post']['multimodal_locality_acc'] = \
#                         np.mean(np.equal(metrics['post']['multimodal_locality_output'],
#                                             metrics['pre']['multimodal_locality_output']))
#                     metrics['post'].pop('multimodal_locality_output')
#                     metrics['pre'].pop('multimodal_locality_output')
#
#                 LOG.info(f"Evaluation took {time() - start}")
#
#                 if verbose:
#                     LOG.info(
#                         f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
#                     )
#
#                 all_metrics.append(metrics)
#
#             # case_result_path = base_case_path / f"case_{i}.json"
#
#             # Dump metrics in .json
#             # with open(case_result_path, "w") as f:
#             #     json.dump(metrics, f, indent=1)
#
#         return all_metrics, edited_model, weights_copy
#
#     # edit_demo will return the logits after/before editing
#     def edit_demo(self,
#             prompts: Union[str, List[str]],
#             targets: Union[str, List[str]],
#             image: Union[str, List[str]],
#             rephrase_prompts: Optional[Union[str, List[str]]] = None,
#             rephrase_image: Optional[Union[str, List[str]]] = None,
#             locality_inputs: Optional[dict] = None,
#             keep_original_weight=False,
#             verbose=True,
#             **kwargs
#             ):
#         """
#         `prompts`: list or str
#             the prompts to edit
#         `targets`: str
#             the expected outputs
#         `image`: dict
#             for multimodal
#         """
#         if isinstance(prompts, List):
#             assert len(prompts) == len(targets) == len(image)
#         else:
#             prompts, targets, image = [prompts,], [targets,], [image,]
#
#         if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
#             self.hparams.batch_size = 1
#
#         requests = self._prepare_requests(prompts, targets, image, rephrase_prompts, rephrase_image, locality_inputs,
#                                           **kwargs)
#
#         if hasattr(self.hparams, 'batch_size') :
#                assert self.hparams.batch_size == 1 or \
#                       print(f'Single Edit, pls set the batch_size to 1....')
#
#         # if not os.path.exists(RESULTS_DIR):
#         #     os.mkdir(RESULTS_DIR)
#         # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
#         # if not os.path.exists(base_case_path):
#         #     os.mkdir(base_case_path)
#         # print(f"Results will be stored at {base_case_path}")
#         all_metrics = []
#         for i, request in enumerate(requests):
#             start = time()
#
#             if self.alg_name == 'IKE':
#                 assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
#                 edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
#                     self.model,
#                     self.tok,
#                     request,
#                     self.hparams,
#                     copy=False,
#                     return_orig_weights=True,
#                     keep_original_weight=keep_original_weight,
#                     train_ds=kwargs['train_ds']
#                 )
#                 exec_time = time() - start
#                 LOG.info(f"Execution {i} editing took {exec_time}")
#                 start = time()
#                 metrics = {
#                     'case_id': i,
#                     # "requested_rewrite": request,
#                     "time": exec_time,
#                     "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
#                                                      request, self.hparams.device),
#                     "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
#                                                      request, self.hparams.device, pre_edit=True)
#                 }
#                 metrics['pre'].pop('locality_acc')
#                 metrics['pre'].pop('locality_image_acc')
#
#                 LOG.info(f"Evaluation took {time() - start}")
#
#                 if verbose:
#                     LOG.info(
#                         f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
#                     )
#
#                 all_metrics.append(metrics)
#             else:
#                 edited_model, weights_copy = self.apply_algo(
#                     self.model,
#                     self.tok,
#                     [request],
#                     self.hparams,
#                     copy=False,
#                     return_orig_weights=True,
#                     keep_original_weight=keep_original_weight,
#                     train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
#                 )
#                 exec_time = time() - start
#                 LOG.info(f"Execution {i} editing took {exec_time}")
#
#                 start = time()
#                 post, post_logits = compute_multimodal_edit_results_demo(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device)
#                 metrics = {
#                     'case_id': i,
#                     # "requested_rewrite": request,
#                     "time": exec_time,
#                     "post": post
#                 }
#                 if self.alg_name == 'KN':
#                     with torch.no_grad():
#                         weights_copy() # unpatch_fn
#                 else:
#                     with torch.no_grad():
#                         for k, v in weights_copy.items():
#                             nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
#                 pre, pre_logits = compute_multimodal_edit_results_demo(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device)
#                 metrics["pre"] = pre
#                 if 'locality_output' in metrics['post'].keys():
#                     assert len(metrics['post']['locality_output']) == \
#                             len(metrics['pre']['locality_output'])
#                     metrics['post']['locality_acc'] = \
#                         np.mean(np.equal(metrics['post']['locality_output'],
#                                             metrics['pre']['locality_output']))
#                     metrics['post'].pop('locality_output')
#                     metrics['pre'].pop('locality_output')
#
#                 if 'multimodal_locality_output' in metrics['post'].keys():
#                     assert len(metrics['post']['multimodal_locality_output']) == \
#                             len(metrics['pre']['multimodal_locality_output'])
#                     metrics['post']['multimodal_locality_acc'] = \
#                         np.mean(np.equal(metrics['post']['multimodal_locality_output'],
#                                             metrics['pre']['multimodal_locality_output']))
#                     metrics['post'].pop('multimodal_locality_output')
#                     metrics['pre'].pop('multimodal_locality_output')
#
#                 LOG.info(f"Evaluation took {time() - start}")
#
#                 if verbose:
#                     LOG.info(
#                         f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
#                     )
#
#                 all_metrics.append(metrics)
#
#             # case_result_path = base_case_path / f"case_{i}.json"
#
#             # Dump metrics in .json
#             # with open(case_result_path, "w") as f:
#             #     json.dump(metrics, f, indent=1)
#
#         return all_metrics, edited_model, weights_copy, post_logits, pre_logits
#
#     def edit_dataset(self,
#                      ds: Dataset,
#                      keep_original_weight=False,
#                      verbose=True,
#                      cur_time='',
#                      **kwargs
#                      ):
#         # Make Sure dataset supported
#         assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in MULTIMODAL_DS_DICT.values()]) > 0 \
#         or print(f'DataSet {ds} not supported yet.')
#
#         # assert hasattr(self.hparams, 'batch_size') or \
#         #         print(f'Method {self.alg_name} found, pls set the batch_size correctly')
#
#         num_edits = 1
#         # num_edits = self.hparams.batch_size
#
#         all_metrics = []
#         save_txt =  os.path.join(self.hparams.results_dir, 'IKE/{cur_time}_{self.model_name}_port_hop{ds.hop}.txt')
#         save_json = os.path.join(self.hparams.results_dir, 'IKE/{cur_time}_{self.model_name}_port_hop{ds.hop}.json')
#         port_result = []
#
#         for i, request in tqdm(enumerate(ds), desc='Editing dataset', total=len(ds), ncols=120):
#             start = time()
#
#             if self.alg_name == 'IKE':
#                 assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
#                 edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
#                     self.model,
#                     self.tok,
#                     request,
#                     self.hparams,
#                     copy=False,
#                     return_orig_weights=True,
#                     keep_original_weight=keep_original_weight,
#                     train_ds=kwargs['train_ds']
#                 )
#                 exec_time = time() - start
#                 LOG.info(f"Execution {i} editing took {exec_time}")
#                 start = time()
#                 metrics = {
#                     'case_id': i,
#                     # "requested_rewrite": request,
#                     "time": exec_time,
#                     "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
#                                                      request, self.hparams.device),
#                     "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
#                                                      request, self.hparams.device, pre_edit=True)
#                 }
#                 # metrics['pre'].pop('locality_acc')
#                 # metrics['pre'].pop('locality_image_acc')
#                 # metrics['pre'].pop('portability_acc')
#                 # port = {
#                 #     'edit_input': ' '.join([request["prompt"], request["target"]]),
#                 #     # 'port_input': ' '.join([request['portability_prompt'][0], request['portability_ground_truth'][0]]),
#                 #     'port_acc': metrics['post']['portability_acc'].item(),
#                 #     'port_pred_ids': metrics['post']['pred_ids'],
#                 #     'port_targ_ids': metrics['post']['targ_ids']
#                 # }
#                 # port_result.append(port)
#                 # write port kv to txt
#                 # os.makedirs(os.path.dirname(save_txt), exist_ok=True)
#                 # with open(save_txt, 'a') as f:
#                 #     f.write(f"{port['edit_input']}\n{port['port_input']}\n{port['port_acc']}\npred: {port['port_pred_ids']}\ntarget: {port['port_targ_ids']}\n\n")
#
#                 LOG.info(f"Evaluation took {time() - start}")
#
#                 if verbose:
#                     LOG.info(
#                         f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
#                     )
#
#                 all_metrics.append(metrics)
#                 # break
#             else:
#                 edited_model, weights_copy = self.apply_algo(
#                     self.model,
#                     self.tok,
#                     [request],
#                     self.hparams,
#                     copy=False,
#                     return_orig_weights=True,
#                     keep_original_weight=keep_original_weight,
#                     train_ds=None
#                 )
#                 exec_time = time() - start
#                 LOG.info(f"Execution {i} editing took {exec_time}")
#
#                 start = time()
#                 post, post_logits = compute_multimodal_edit_results_demo(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device)
#                 metrics = {
#                     'case_id': i,
#                     # "requested_rewrite": request,
#                     "time": exec_time,
#                     "post": post
#                 }
#                 if self.alg_name == 'KN':
#                     with torch.no_grad():
#                         weights_copy() # unpatch_fn
#                 else:
#                     with torch.no_grad():
#                         for k, v in weights_copy.items():
#                             nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
#                 pre, pre_logits = compute_multimodal_edit_results_demo(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device)
#                 metrics["pre"] = pre
#                 if 'locality_output' in metrics['post'].keys():
#                     assert len(metrics['post']['locality_output']) == \
#                             len(metrics['pre']['locality_output'])
#                     metrics['post']['locality_acc'] = \
#                         np.mean(np.equal(metrics['post']['locality_output'],
#                                             metrics['pre']['locality_output']))
#                     metrics['post'].pop('locality_output')
#                     metrics['pre'].pop('locality_output')
#
#                 if 'multimodal_locality_output' in metrics['post'].keys():
#                     assert len(metrics['post']['multimodal_locality_output']) == \
#                             len(metrics['pre']['multimodal_locality_output'])
#                     metrics['post']['multimodal_locality_acc'] = \
#                         np.mean(np.equal(metrics['post']['multimodal_locality_output'],
#                                             metrics['pre']['multimodal_locality_output']))
#                     metrics['post'].pop('multimodal_locality_output')
#                     metrics['pre'].pop('multimodal_locality_output')
#
#                 LOG.info(f"Evaluation took {time() - start}")
#
#                 if verbose:
#                     LOG.info(
#                         f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
#                     )
#
#                 all_metrics.append(metrics)
#
#         # with open(save_json, 'w') as f:
#         #     json.dump(port_result, f, indent=2)
#
#         return all_metrics, edited_model, weights_copy
#
#     def _chunks(self, arr, n):
#         """Yield successive n-sized chunks from arr."""
#         for i in range(0, len(arr), n):
#             yield arr[i: i + n]
#
#     def _init_ds(self, ds: Dataset):
#         """Init ds to inputs format."""
#         data = {
#             'prompts': [],
#             'targets': [],
#             'image': [],
#             'rephrase_prompts': [],
#             'rephrase_image': [],
#             'locality_inputs': {'text': {'prompt': [], 'ground_truth': []}, 'vision': {'image': [], 'prompt': [], 'ground_truth': []}}
#         }
#
#         for record in ds:
#             data['prompts'].append(record['src'])
#             data['targets'].append(record['alt'])
#             data['image'].append(record['image'])
#             data['rephrase_prompts'].append(record['rephrase'])
#             data['rephrase_image'].append(record['image_rephrase'])
#             data['locality_inputs']['text']['prompt'].append(record['loc'])
#             data['locality_inputs']['text']['ground_truth'].append(record['loc_ans'])
#             data['locality_inputs']['vision']['image'].append(record['m_loc'])
#             data['locality_inputs']['vision']['prompt'].append(record['m_loc_q'])
#             data['locality_inputs']['vision']['ground_truth'].append(record['m_loc_a'])
#
#         return data
#
#     def _prepare_requests(self,
#                           prompts: Union[str, List[str]],
#                           targets: Union[str, List[str]],
#                           image: Union[str, List[str]],
#                           rephrase_prompts: Optional[Union[str, List[str]]] = None,
#                           rephrase_image: Optional[Union[str, List[str]]] = None,
#                           locality_inputs: Optional[dict] = None,
#                           **kwargs
#                           ):
#         if isinstance(image, str):
#             image = [image, ]
#         image_path = [os.path.join(self.vis_root, image_) for image_ in image]
#         image = [Image.open(ip).convert("RGB") for ip in image_path]
#         image = [self.vis_tok(i).to(self.hparams.device) for i in image]
#
#         requests = [{
#             'prompt': prompt,
#             'target': target,
#             'image': image_,
#         }
#         for prompt, target, image_ in zip(prompts, targets, image)
#         ]
#
#         if "text" in locality_inputs.keys():
#             locality_prompts = locality_inputs['text']['prompt']
#             locality_ground_truth = locality_inputs['text']['ground_truth']
#             if isinstance(locality_prompts, str):
#                 locality_prompts = [locality_prompts, ]
#             if isinstance(locality_ground_truth, str):
#                 locality_ground_truth = [locality_ground_truth, ]
#             assert len(locality_inputs['text']['prompt']) == len(locality_inputs['text']['ground_truth']) \
#                 == len(requests) or print('One Edit instance needs one locality input.....')
#         if "vision" in locality_inputs.keys():
#             multimodal_locality_prompts = locality_inputs['vision']['prompt']
#             multimodal_locality_ground_truth = locality_inputs['vision']['ground_truth']
#             multimodal_locality_image = locality_inputs['vision']['image']
#             if isinstance(multimodal_locality_prompts, str):
#                 multimodal_locality_prompts = [multimodal_locality_prompts, ]
#             if isinstance(multimodal_locality_ground_truth, str):
#                 multimodal_locality_ground_truth = [multimodal_locality_ground_truth, ]
#             if isinstance(multimodal_locality_image, str):
#                 multimodal_locality_image = [multimodal_locality_image, ]
#             assert len(locality_inputs['vision']['prompt']) == len(locality_inputs['vision']['ground_truth']) \
#                 == len(locality_inputs['vision']['image']) == len(requests) or print('One Edit instance needs one locality input.....')
#
#         if rephrase_prompts is not None:
#             if isinstance(rephrase_prompts, str):
#                 rephrase_prompts = [rephrase_prompts,]
#
#             for i, request in enumerate(requests):
#                 request.update(
#                     {
#                         'rephrase_prompt': rephrase_prompts[i],
#                     }
#                 )
#         if rephrase_image is not None:
#             if isinstance(rephrase_image, str):
#                 rephrase_image = [rephrase_image, ]
#             rephrase_image_path = [os.path.join(self.rephrase_root, rephrase_image_) for rephrase_image_ in rephrase_image]
#             rephrase_image = [Image.open(ip).convert("RGB") for ip in rephrase_image_path]
#             rephrase_image = [self.vis_tok(i).to(self.hparams.device) for i in rephrase_image]
#
#             for i, request in enumerate(requests):
#                 request.update(
#                     {
#                         'image_rephrase': rephrase_image[i],
#                     }
#                 )
#
#         if "text" in locality_inputs.keys():
#
#             for i, request in enumerate(requests):
#                 request.update(
#                     {
#                         'locality_prompt': locality_prompts[i],
#                         'locality_ground_truth': locality_ground_truth[i]
#                     }
#                 )
#
#         if "vision" in locality_inputs.keys():
#
#             locality_image_path = [os.path.join(self.vis_root, multimodal_locality_image_) for multimodal_locality_image_ in multimodal_locality_image]
#             locality_image = [Image.open(ip).convert("RGB") for ip in locality_image_path]
#             locality_image = [self.vis_tok(i).to(self.hparams.device) for i in locality_image]
#
#             for i, request in enumerate(requests):
#                 request.update(
#                     {
#                         'multimodal_locality_image': locality_image[i],
#                         'multimodal_locality_prompt': multimodal_locality_prompts[i],
#                         'multimodal_locality_ground_truth': multimodal_locality_ground_truth[i],
#                     }
#                 )
#
#         return requests
#
#
# # if __name__ == "__main__":
# #
# #     editor = BaseEditor(alg_name='KN', model_name='/nature/peng/serac/hugging_cache/t5-3b-finetuned-counterfact-10000', hparams_fname='t5-3b.json')
# #
# #     editor.edit(
# #         prompts='What university did Watts Humphrey attend?',
# #         ground_truth='Illinois Institute of Technology',
# #         target_new='University of Michigan'
# #     )
# #
# #     metrics, edited_model, _ = editor.edit(prompts='What university did Watts Humphrey attend?', ground_truth='Illinois Institute of Technology', target_new='University of Michigan')
#
#
from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from .editor import BaseEditor
import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
from PIL import Image

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .batch_editor import BatchEditor
from ..evaluate import (compute_icl_multimodal_edit_quality,
                        compute_multimodal_edit_results,
                        compute_multimodal_edit_results_demo)
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)


def make_logs():

    f_h, s_h = get_handler("logs/", log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def get_model_key(model_name):
    # by default, put all module on cuda:0
    model_name = model_name.lower()
    device_map = {}
    if 'gpt-j-6b' in model_name:
        key = ['transformer.wte', 'transformer.drop'] + ['transformer.h.'+str(i) for i in range(28)] + ['transformer.ln_f', 'lm_head']
    elif 'gpt2-xl' in model_name:
        key = ['transformer.wte', 'transformer.wpe'] + ['transformer.h.'+str(i) for i in range(48)] + ['transformer.ln_f', 'lm_head']
    elif 'opt-2.7b' in model_name:
        key = ['model.decoder.embed_tokens', 'model.decoder.embed_positions', 'lm_head'] + ['model.decoder.layers.'+str(i) for i in range(32)] + ['model.decoder.self_attn_layer_norm', 'model.decoder.fc1', 'model.decoder.fc2', 'model.decoder.final_layer_norm']
    elif 'opt-125m' in model_name:
        key = ['model.decoder.embed_tokens', 'model.decoder.embed_positions', 'lm_head'] + ['model.decoder.layers.'+str(i) for i in range(12)] + ['model.decoder.self_attn_layer_norm', 'model.decoder.fc1', 'model.decoder.fc2', 'model.decoder.final_layer_norm']
    elif 'chatglm2_6b' in model_name:
        key = ['transformer.embedding', 'transformer.rotary_pos_emb'] + ['transformer.encoder.layers.'+str(i) for i in range(28)] + ['transformer.encoder.final_layernorm', 'transformer.output_layer']
    elif 'llama-2-7b' in model_name:
        key = ['model.embed_tokens'] + ['model.layers.'+str(i) for i in range(32)] + ['model.norm', 'lm_head']
    elif 'vicuna-7b' in model_name:
        key = ['model.embed_tokens'] + ['model.layers.'+str(i) for i in range(32)] + ['model.norm', 'lm_head']
    elif 'vicuna-13b' in model_name:
        key = ['model.embed_tokens'] + ['model.layers.'+str(i) for i in range(40)] + ['model.norm', 'lm_head']
    else:
        return 'auto'
    for k in key:
        device_map[k] = 0
    return device_map

def get_device_map(device_map, hparams):
    # change the exist device_map, and set module by module name and split in hparams
    if device_map == 'auto':
        return device_map
    index = 0
    for (k, _) in device_map.items():
        if len(hparams.gpu_split) == 0:
            device_map[k] = hparams.device
        else:
            for j in range(len(hparams.gpu_split)):
                if hparams.gpu_split[j] in k and k[-2] == hparams.gpu_split[j][-2]:
                    index += 1
            device_map[k] = hparams.gpu_used_id[index]
    print(device_map)
    return device_map

class MultimodalEditor:
    """Multimodal editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                 hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_MULTIMODAL_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")
        #
        # if type(self.model_name) is str:
        #     if hparams.model_name == "blip2":
        #         from ..trainer.blip2_models import Blip2OPT
        #
        #         model = Blip2OPT(
        #             vit_model="eva_clip_g",
        #             img_size=364,
        #             use_grad_checkpoint=True,
        #             vit_precision="fp32",
        #             freeze_vit=True,
        #             opt_model=hparams.name,
        #             state_dict_file=hparams.state_dict_file,
        #             qformer_name_or_path=hparams.qformer_name_or_path,
        #             qformer_checkpoint=hparams.qformer_checkpoint
        #         )
        if type(self.model_name) is str:
            # from ..util.tools import get_model_key, get_device_map
            device_map = get_model_key(hparams.name)
            device_map = get_device_map(device_map, hparams)
            if hparams.model_name == "blip2":
                from ..trainer.blip2_models import Blip2OPT

                model = Blip2OPT(
                    vit_model="eva_clip_g",
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    opt_model=hparams.name,
                    state_dict_file=hparams.state_dict_file,
                    qformer_name_or_path=hparams.qformer_name_or_path,
                    qformer_checkpoint=hparams.qformer_checkpoint
                )
                # if hparams.model_parallel:
                #     model.ln_vision.to(hparams.device)
                #     model.visual_encoder.to(hparams.device)
                #     model.Qformer.to(hparams.device)
                #     model.opt_proj.to(hparams.device)
            elif hparams.model_name == "minigpt4":
                from ..trainer.blip2_models import MiniGPT4

                model = MiniGPT4(
                    vit_model="eva_clip_g",
                    qformer_checkpoint=hparams.qformer_checkpoint,
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    llama_model=hparams.name,
                    state_dict_file=hparams.state_dict_file,
                    qformer_name_or_path=hparams.qformer_name_or_path,
                    pretrained_ckpt=hparams.pretrained_ckpt,
                    # model_parallel = hparams.model_parallel,
                    # device_map = device_map
                )
                if hparams.model_parallel:
                    model.ln_vision.to(hparams.device)
                    model.visual_encoder.to(hparams.device)
                    model.Qformer.to(hparams.device)
                    model.llama_proj.to(hparams.device)
            self.model = model
            # Get tokenizer and vis_processor
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)

            self.vis_tok = vis_processor
            if (hparams is not None and hasattr(hparams, 'tokenizer_name')):
                tok_name = (
                    hparams.tokenizer_name
                    if hparams.tokenizer_name is not None
                    else hparams.name
                )
                tokenizer = getattr(transformers, hparams.tokenizer_class).from_pretrained(
                    tok_name
                )
                if tokenizer.pad_token == None or tokenizer.pad_token == '':
                    tokenizer.pad_token = tokenizer.eos_token
                self.tok = tokenizer
        else:
            self.model, self.tok = self.model_name

        self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams
        self.vis_root = hparams.coco_image
        self.rephrase_root = hparams.rephrase_image

    def edit(self,
             prompts: Union[str, List[str]],
             targets: Union[str, List[str]],
             image: Union[str, List[str]],
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             rephrase_image: Optional[Union[str, List[str]]] = None,
             locality_inputs: Optional[dict] = None,
             keep_original_weight=False,
             verbose=True,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `targets`: str
            the expected outputs
        `image`: dict
            for multimodal
        """
        assert self.alg_name == 'IKE' or print('Only IKE supported for MultimodalEditor')
        if isinstance(prompts, List):
            assert len(prompts) == len(targets) == len(image)
        else:
            prompts, targets, image = [prompts,], [targets,], [image,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        requests = self._prepare_requests(prompts, targets, image, rephrase_prompts, rephrase_image, locality_inputs,
                                          **kwargs)

        if hasattr(self.hparams, 'batch_size') :
            assert self.hparams.batch_size == 1 or \
                   print(f'Single Edit, pls set the batch_size to 1....')

        all_metrics = []
        for i, request in enumerate(requests):
            start = time()

            assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
            edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                self.model,
                self.tok,
                request,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
                train_ds=kwargs['train_ds']
            )
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")
            start = time()
            metrics = {
                'case_id': i,
                # "requested_rewrite": request,
                "time": exec_time,
                "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                            request, self.hparams.device),
                "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                           request, self.hparams.device, pre_edit=True)
            }
            if 'locality_output' in metrics['post'].keys():
                assert len(metrics['post']['locality_output']) == \
                       len(metrics['pre']['locality_output'])
                base_logits = metrics['pre']['locality_output'].to(torch.float32)
                post_logits = metrics['post']['locality_output'].to(torch.float32)
                if post_logits.shape[1] > base_logits.shape[1]:
                    post_logits = post_logits[:, -base_logits.shape[1]:, :]
                else:
                    base_logits = base_logits[:, -post_logits.shape[1]:, :]

                base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
                post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=1, dim=-1).indices
                metrics['post']['locality_acc'] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                metrics['post'].pop('locality_output')
                metrics['pre'].pop('locality_output')

            if 'multimodal_locality_output' in metrics['post'].keys():
                assert len(metrics['post']['multimodal_locality_output']) == \
                       len(metrics['pre']['multimodal_locality_output'])
                base_image_logits = metrics['pre']['multimodal_locality_output'].to(torch.float32)
                post_image_logits = metrics['post']['multimodal_locality_output'].to(torch.float32)
                if post_image_logits.shape[1] > base_image_logits.shape[1]:
                    post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
                else:
                    base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

                base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
                metrics['post']['multimodal_locality_acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                metrics['post'].pop('multimodal_locality_output')
                metrics['pre'].pop('multimodal_locality_output')

            LOG.info(f"Evaluation took {time() - start}")

            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                )

            all_metrics.append(metrics)

        return all_metrics, edited_model, weights_copy

    def edit_dataset(self,
                     ds: Dataset,
                     keep_original_weight=False,
                     verbose=True,
                     **kwargs
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in MULTIMODAL_DS_DICT.values()]) > 0 \
               or print(f'DataSet {ds} not supported yet.')

        assert self.alg_name == 'IKE' or print('Only IKE supported for MultimodalEditor')
        num_edits = 1
        # num_edits = self.hparams.batch_size

        all_metrics = []

        for i, request in enumerate(tqdm(ds, desc='Editing dataset', total=len(ds))):

            start = time()

            assert 'train_ds' in kwargs.keys() or print('IKE need train_ds (For getting In-Context prompt)')
            edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                self.model,
                self.tok,
                request,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
                train_ds=kwargs['train_ds']
            )
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")
            start = time()
            metrics = {
                'case_id': i,
                "time": exec_time,
                "post": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                            request, self.hparams.device),
                "pre": compute_icl_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                           request, self.hparams.device, pre_edit=True)
            }
            if 'locality_output' in metrics['post'].keys():
                assert len(metrics['post']['locality_output']) == \
                       len(metrics['pre']['locality_output'])
                base_logits = metrics['pre']['locality_output'].to(torch.float32)
                post_logits = metrics['post']['locality_output'].to(torch.float32)
                if post_logits.shape[1] > base_logits.shape[1]:
                    post_logits = post_logits[:, -base_logits.shape[1]:, :]
                else:
                    base_logits = base_logits[:, -post_logits.shape[1]:, :]

                base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
                post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=1, dim=-1).indices
                metrics['post']['locality_acc'] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                metrics['post'].pop('locality_output')
                metrics['pre'].pop('locality_output')

            if 'multimodal_locality_output' in metrics['post'].keys():
                assert len(metrics['post']['multimodal_locality_output']) == \
                       len(metrics['pre']['multimodal_locality_output'])
                base_image_logits = metrics['pre']['multimodal_locality_output'].to(torch.float32)
                post_image_logits = metrics['post']['multimodal_locality_output'].to(torch.float32)
                if post_image_logits.shape[1] > base_image_logits.shape[1]:
                    post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
                else:
                    base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

                base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
                metrics['post']['multimodal_locality_acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                metrics['post'].pop('multimodal_locality_output')
                metrics['pre'].pop('multimodal_locality_output')

            LOG.info(f"Evaluation took {time() - start}")

            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                )

                all_metrics.append(metrics)

        return all_metrics, edited_model, weights_copy

    def edit_dataset_hice(self,
                          ds: Dataset,
                          keep_original_weight=False,
                          verbose=True,
                          **kwargs
                          ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in MULTIMODAL_DS_DICT.values()]) > 0 \
               or print(f'DataSet {ds} not supported yet.')
        assert 'train_ds' in kwargs.keys() or print('HICE need train_ds (For getting In-Context prompt)')
        device = torch.device(f'cuda:{self.hparams.device}')

        path = 'ranpac_{}_{}.pth'.format(self.hparams.task_name.lower(), self.hparams.sentence_model_name.split('/')[-1])
        params = torch.load(path)
        Wo, W_rand = params['Wo'].to(device), params['W_rand'].to(device)

        memory_path = 'memory_{}_{}.pth'.format(self.hparams.task_name.lower(), self.hparams.sentence_model_name.split('/')[-1])
        mloc_memory = torch.load(memory_path)
        memory_fea = torch.cat(mloc_memory['embeddings'], dim=0) # store to accelerate
        mloc_memory = memory_fea / torch.norm(memory_fea, p=2, dim=-1).unsqueeze(-1)

        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer(self.hparams.sentence_model_name).to(device)

        stored_data = torch.load(f'hice_{self.hparams.task_name.lower()}_{self.hparams.model_name.lower()}_embeddings.pth')
        stored_data['embeddings'] = torch.tensor(stored_data['embeddings']).to(device)
        LOG.info('number of saved sentences are {}'.format(len(stored_data['sentences'])))


        all_metrics = []
        keys = ['img_topk', 'txt_topk', 'img_last_topk', 'txt_last_topk']
        keys += ['ori_rt_img_topk', 'ori_rt_txt_topk', 'ori_rt_img_last_topk', 'ori_rt_txt_last_topk']
        KEYs = ['acc', 'pred', 'targ', 'fc_acc']
        res = {}
        for K in KEYs:
            res[K] = {}
            for key in keys:
                res[K][key] = []

        from ..evaluate import compute_hice_multimodal_edit_quality
        for i, request in enumerate(tqdm(ds, desc='Editing dataset', total=len(ds))):
            start = time()

            edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                self.model,
                self.tok,
                request,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
                train_ds=kwargs['train_ds'],
                sentence_model=sentence_model,
                stored_data=stored_data,
            )
            if len(''.join(icl_examples)) > 2500:
                while (len(''.join(icl_examples)) > 2500):
                    del icl_examples[-2]

            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")
            start = time()

            metrics = {
                'case_id': i,
                "time": exec_time,
                "post": compute_hice_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                             request, self.hparams.device, Wo=Wo, W_rand=W_rand, sentence_model=sentence_model,
                                                             stored_data=stored_data, mloc_memory=mloc_memory),
                "pre": compute_hice_multimodal_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                            request, self.hparams.device, pre_edit=True)
            }

            if 'locality_output' in metrics['post'].keys():
                assert len(metrics['post']['locality_output']) == \
                       len(metrics['pre']['locality_output'])
                base_logits = metrics['pre']['locality_output'].to(torch.float32)
                post_logits = metrics['post']['locality_output'].to(torch.float32)
                if post_logits.shape[1] > base_logits.shape[1]:
                    post_logits = post_logits[:, -base_logits.shape[1]:, :]
                else:
                    base_logits = base_logits[:, -post_logits.shape[1]:, :]

                base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
                post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=1, dim=-1).indices
                metrics['post']['locality_acc'] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                metrics['post'].pop('locality_output')
                metrics['pre'].pop('locality_output')

            if 'multimodal_locality_output' in metrics['post'].keys():
                assert len(metrics['post']['multimodal_locality_output']) == \
                       len(metrics['pre']['multimodal_locality_output'])
                base_image_logits = metrics['pre']['multimodal_locality_output'].to(torch.float32)
                post_image_logits = metrics['post']['multimodal_locality_output'].to(torch.float32)
                if post_image_logits.shape[1] > base_image_logits.shape[1]:
                    post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
                else:
                    base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

                base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
                metrics['post']['multimodal_locality_acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                metrics['post'].pop('multimodal_locality_output')
                metrics['pre'].pop('multimodal_locality_output')

            if 'random_multimodal_locality_output' in metrics['post'].keys():
                assert len(metrics['post']['random_multimodal_locality_output']) == \
                       len(metrics['pre']['random_multimodal_locality_output'])
                base_image_logits = metrics['pre']['random_multimodal_locality_output'].to(torch.float32)
                post_image_logits = metrics['post']['random_multimodal_locality_output'].to(torch.float32)
                if post_image_logits.shape[1] > base_image_logits.shape[1]:
                    post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
                else:
                    base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

                base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
                metrics['post']['random_multimodal_locality_output'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                metrics['post'].pop('random_multimodal_locality_output')
                metrics['pre'].pop('random_multimodal_locality_output')

            # import pdb
            # pdb.set_trace()
            editing_qs, editing_ans = request['prompt'], request['target']
            if 'img_topk' in request.keys() or 'ori_rt_img_topk' in request.keys():# and i > 4250:# img_topk and txt_topk
                tok = self.model.llama_tokenizer if 'vicuna' in self.hparams.name.lower() else self.model.opt_tokenizer
                from ..evaluate import icl_multimodal_lm_eval, our_predict_domain, select_icl_samples
                for key in keys:
                    t_res = {}
                    for K in KEYs:
                        t_res[K] = []
                    for ind in request[key]:
                        record = ds.all_edit_inner[ind] if 'ori_rt' not in key else ds.ori_right[ind]
                        target = record["target"][0]
                        prompt = record["prompt"][0]
                        image = record["image"] if record["image"].is_cuda else record["image"].to(self.hparams.device)

                        query_embedding = None
                        _, fc_preds = our_predict_domain(prompt, '', sentence_model, Wo, W_rand, device, mloc_memory=mloc_memory, threshold=self.hparams.threshold)
                        if fc_preds == 0:
                            edit_acc, preds_id, targ = icl_multimodal_lm_eval(self.model, self.model_name, self.hparams, self.tok, [''],
                                                                              target, prompt, image, return_targ=True)
                            t_res['fc_acc'].append(0)
                        else:
                            icl_examples = select_icl_samples(prompt, '', stored_data, sentence_model, device, self.hparams, query_embedding=query_embedding)
                            edit_acc, preds_id, targ = icl_multimodal_lm_eval(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                                              # target, new_fact, image)
                                                                              target, f'New Fact: {editing_qs} {editing_ans}\nPrompt: {prompt}', image, return_targ=True)
                            t_res['fc_acc'].append(1)

                        t_res['acc'].append(edit_acc)
                        t_res['pred'].append([preds_id[preds_id!=0]])
                        # t_res['targ'].append(target)

                        if 'ori' in key: # KPI
                            t_res['targ'].append([record[f'ori_pred_{self.hparams.model_name.lower()}']])
                        else: # KGI
                            t_res['targ'].append([targ[targ!=0]])

                        record["image"] = record["image"].cpu()
                        torch.cuda.empty_cache()
                    for K in KEYs:
                        res[K][key].append(t_res[K])

            LOG.info(f"Evaluation took {time() - start}")
            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target']}  \n {metrics}"
                )

                all_metrics.append(metrics)

                if i % 20 == 0 and i > 0:
                    acc_per_sample_path = 'HICE_{}_{}_checkpoint.pth'.format(self.model_name, self.hparams.task)
                    acc_per_sample_path_final = 'HICE_{}_{}.pth'.format(self.model_name, self.hparams.task)
                    acc_path = 'HICE_{}_{}_metrics_checkpoint.pth'.format(self.model_name, self.hparams.task)
                    acc_path_final = 'HICE_{}_{}_metrics.pth'.format(self.model_name, self.hparams.task)
                    torch.save(all_metrics, acc_path)
                    if 'img_topk' in request.keys() or 'ori_rt_img_topk' in request.keys():
                        dd = {}
                        for K in KEYs:
                            dd[K] = {}
                            for key in keys:
                                dd[K][key] = res[K][key]
                        torch.save(dd, acc_per_sample_path)

        torch.save(all_metrics, acc_path_final)
        os.remove(acc_path)
        if 'img_topk' in request.keys() or 'ori_rt_img_topk' in request.keys():
            dd = {}
            for K in KEYs:
                dd[K] = {}
                for key in keys:
                    dd[K][key] = res[K][key]
            torch.save(dd, acc_per_sample_path_final)
            os.remove(acc_per_sample_path)
        return all_metrics, edited_model, weights_copy


    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]

    def _init_ds(self, ds: Dataset):
        """Init ds to inputs format."""
        data = {
            'prompts': [],
            'targets': [],
            'image': [],
            'rephrase_prompts': [],
            'rephrase_image': [],
            'locality_inputs': {'text': {'prompt': [], 'ground_truth': []}, 'vision': {'image': [], 'prompt': [], 'ground_truth': []}}
        }

        for record in ds:
            data['prompts'].append(record['src'])
            data['targets'].append(record['alt'])
            data['image'].append(record['image'])
            data['rephrase_prompts'].append(record['rephrase'])
            data['rephrase_image'].append(record['image_rephrase'])
            data['locality_inputs']['text']['prompt'].append(record['loc'])
            data['locality_inputs']['text']['ground_truth'].append(record['loc_ans'])
            data['locality_inputs']['vision']['image'].append(record['m_loc'])
            data['locality_inputs']['vision']['prompt'].append(record['m_loc_q'])
            data['locality_inputs']['vision']['ground_truth'].append(record['m_loc_a'])

        return data

    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          targets: Union[str, List[str]],
                          image: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          rephrase_image: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[dict] = None,
                          **kwargs
                          ):
        if isinstance(image, str):
            image = [image, ]
        image_path = [os.path.join(self.vis_root, image_) for image_ in image]
        image = [Image.open(ip).convert("RGB") for ip in image_path]
        image = [self.vis_tok(i).to(self.hparams.device) for i in image]

        requests = [{
            'prompt': prompt,
            'target': target,
            'image': image_,
        }
            for prompt, target, image_ in zip(prompts, targets, image)
        ]

        if "text" in locality_inputs.keys():
            locality_prompts = locality_inputs['text']['prompt']
            locality_ground_truth = locality_inputs['text']['ground_truth']
            if isinstance(locality_prompts, str):
                locality_prompts = [locality_prompts, ]
            if isinstance(locality_ground_truth, str):
                locality_ground_truth = [locality_ground_truth, ]
            assert len(locality_inputs['text']['prompt']) == len(locality_inputs['text']['ground_truth']) \
                   == len(requests) or print('One Edit instance needs one locality input.....')
        if "vision" in locality_inputs.keys():
            multimodal_locality_prompts = locality_inputs['vision']['prompt']
            multimodal_locality_ground_truth = locality_inputs['vision']['ground_truth']
            multimodal_locality_image = locality_inputs['vision']['image']
            if isinstance(multimodal_locality_prompts, str):
                multimodal_locality_prompts = [multimodal_locality_prompts, ]
            if isinstance(multimodal_locality_ground_truth, str):
                multimodal_locality_ground_truth = [multimodal_locality_ground_truth, ]
            if isinstance(multimodal_locality_image, str):
                multimodal_locality_image = [multimodal_locality_image, ]
            assert len(locality_inputs['vision']['prompt']) == len(locality_inputs['vision']['ground_truth']) \
                   == len(locality_inputs['vision']['image']) == len(requests) or print('One Edit instance needs one locality input.....')

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
        if rephrase_image is not None:
            if isinstance(rephrase_image, str):
                rephrase_image = [rephrase_image, ]
            rephrase_image_path = [os.path.join(self.rephrase_root, rephrase_image_) for rephrase_image_ in rephrase_image]
            rephrase_image = [Image.open(ip).convert("RGB") for ip in rephrase_image_path]
            rephrase_image = [self.vis_tok(i).to(self.hparams.device) for i in rephrase_image]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'image_rephrase': rephrase_image[i],
                    }
                )

        if "text" in locality_inputs.keys():

            for i, request in enumerate(requests):
                request.update(
                    {
                        'locality_prompt': locality_prompts[i],
                        'locality_ground_truth': locality_ground_truth[i]
                    }
                )

        if "vision" in locality_inputs.keys():

            locality_image_path = [os.path.join(self.vis_root, multimodal_locality_image_) for multimodal_locality_image_ in multimodal_locality_image]
            locality_image = [Image.open(ip).convert("RGB") for ip in locality_image_path]
            locality_image = [self.vis_tok(i).to(self.hparams.device) for i in locality_image]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_locality_image': locality_image[i],
                        'multimodal_locality_prompt': multimodal_locality_prompts[i],
                        'multimodal_locality_ground_truth': multimodal_locality_ground_truth[i],
                    }
                )

        return requests