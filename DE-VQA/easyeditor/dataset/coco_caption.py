"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import pickle
from collections import OrderedDict

from sentence_transformers import SentenceTransformer, util
from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm
from copy import deepcopy

from ..trainer.mPLUG_Owl2.mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from ..trainer.mPLUG_Owl2.mplug_owl2.mm_utils import tokenizer_image_token, process_images

class CaptionDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, name='',*args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("./multmodal/VLKEB-main/clip-vit-large-patch14-336")
        elif config.model_class ==  "qwen-vl":
            vis_processor = BlipImageEvalProcessor(image_size=448, mean=None, std=None)
        elif "owl-2" in config.model_name.lower():
            from transformers.models.clip.image_processing_clip import CLIPImageProcessor
            vis_processor = CLIPImageProcessor.from_pretrained(config.name, trust_remote_code=True)
        else:
            raise NotImplementedError("unknown model class")

        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            if config.tokenizer_class == "QWenTokenizer":
                tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=True, pad_token='<|endoftext|>')
            elif config.model_name == "owl-2":
                tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=False, trust_remote_code=True)
            else:
                tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                    tok_name, trust_remote_code=True
                )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = './data/VLKEB-data/mmkb_images'#config.coco_image
        rephrase_root = './data/VLKEB-data/mmkb_images'#config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: "

        data = []

        if size is not None:
            self.annotation = self.annotation[:size]
        print(len(self.annotation))
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]

        if os.path.exists(name):
            data = torch.load(name)
            print('Loaded datasets from {}'.format(name))
        else:
            print('preocesson data: ')
            self.sentence_model = SentenceTransformer(
                '/root/siton-data-9aa46a4f0e354f65bd8679947a35e67e/LMs/LMs/huggingface/all-MiniLM-L6-v2').to(config.device)

            with open(
                    f'./multmodal/VLKEB-main/results/embedding/vlkeb_embeddings_collect.pkl',
                    "rb") as fIn:
                stored_data = pickle.load(fIn)
                self.stored_sentences = stored_data['sentences']
                self.save_image_path = stored_data['images']
                self.prompts = stored_data['prompts']
                stored_embeddings = stored_data['embeddings']
            stored_embeddings = torch.tensor(stored_embeddings).to(0)

            self.stored_embeddings = util.normalize_embeddings(stored_embeddings)


            for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):

                if record['alt'] == "":
                    continue
                if hop and 'port_new' not in record.keys():
                    continue

                image_path = os.path.join(self.vis_root, record["image"])
                rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
                locality_image_path = os.path.join(self.vis_root, record['m_loc'])
                sim_edit = self.finds_sim(record['src'],record['alt'])

                item = {
                    'prompt': record['src'],
                    'pred': record['pred'],
                    'target': record['alt'],
                    'rephrase_prompt': record['rephrase'],
                    # 'image': image,
                    # 'image_rephrase': rephrase_image,
                    'image': image_path,
                    'image_rephrase': rephrase_image_path,
                    'sim_edit': sim_edit[0][0],
                    'sim_edit_label': sim_edit[0][1],
                    'sim_edit_image': sim_edit[1],
                    'cond': "{} >> {} || {}".format(
                        record['pred'],
                        record['alt'],
                        record['src']
                    )
                }

                item['locality_prompt'] = record['loc']
                item['locality_ground_truth'] = record['loc_ans']

                # item['multimodal_locality_image'] = locality_image
                item['multimodal_locality_image'] = locality_image_path

                item['multimodal_locality_prompt'] = record['m_loc_q']
                item['multimodal_locality_ground_truth'] = record['m_loc_a']

                if hop and 'port_new' in record.keys():
                    item['portability_prompt'] = []
                    item['portability_ground_truth'] = []
                    find_hop = False
                    for ports in record['port_new']:
                        if ports['port_type'] == port_type:
                            find_hop = True
                            port_q = ports['Q&A']['Question']
                            port_a = ports['Q&A']['Answer']
                            item['portability_prompt'].append(port_q)
                            item['portability_ground_truth'].append(port_a)
                            break

                    if not find_hop:
                        continue
                data.append(item)
            torch.save(data,name)
        # if size is not None:
        #     data = data[:size]        
        self._data = data
        print('load:',len(self._data))
        self.no_image = no_image



    def finds_sim(self,src,trg,tops=5):
        query_embedding = util.normalize_embeddings(torch.tensor(self.sentence_model.encode(
            src, show_progress_bar=False)).unsqueeze(0).to(0))
        hits = util.semantic_search(query_embedding, self.stored_embeddings, score_function=util.dot_score, top_k=tops)

        hit = hits[0]

        close_examples = []
        for k in range(len(hit)):
            if self.prompts[hit[k]["corpus_id"]][1]!=trg:
                close_examples.append(self.prompts[hit[k]["corpus_id"]])
                imgs=self.save_image_path[hit[k]["corpus_id"]]
                break

        if  close_examples == []:
            close_examples.append(self.prompts[hit[-1]["corpus_id"]])
            imgs = self.save_image_path[hit[-1]["corpus_id"]]

        if self.config.model_class == "LLaVA":
            image = self.vis_processor(Image.open(imgs).convert("RGB"), return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        else:
            image= self.vis_processor(Image.open(imgs).convert("RGB"))
        return close_examples[0],image


    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])        
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")
        
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)
        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        elif self.config.model_class == "qwen-vl":
            image = os.path.join(self.vis_root, image_path)
            rephrase_image = os.path.join(self.rephrase_root, rephrase_image_path)
            locality_image = os.path.join(self.vis_root, locality_image_path)
        elif self.config.model_name == "owl-2":
            
                    
            _image = Image.open(image_path).convert('RGB')
            max_edge = max(_image.size) 
            image = process_images([_image.resize((max_edge, max_edge))], self.vis_processor)

            _image = Image.open(rephrase_image_path).convert('RGB')
            max_edge = max(_image.size) 
            rephrase_image = process_images([_image.resize((max_edge, max_edge))], self.vis_processor)

            _image = Image.open(locality_image_path).convert('RGB')
            max_edge = max(_image.size) 
            locality_image = process_images([_image.resize((max_edge, max_edge))], self.vis_processor)
        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image

        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch] if "owl-2" not in self.config.model_name else [b['image'] for b in batch][0]
        image_rephrase = [b['image_rephrase'] for b in batch] if "owl-2" not in self.config.model_name else [b['image_rephrase'] for b in batch][0]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch] if "owl-2" not in self.config.model_name else [b['multimodal_locality_image'] for b in batch][0]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        sim_src=[b['sim_edit'] for b in batch]
        sim_image=[b['sim_edit_image'] for b in batch]
        tokenizer = AutoTokenizer.from_pretrained(self.config.name, use_fast=False) if self.config.model_name == "owl-2" else None
        
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
        edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['inputs'] = self.tok(f'Picture 1: <img>{image[0]}</img>\n{src[0]} {trg[0]}', return_tensors='pt')["input_ids"]
        edit_inner['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + src[0] + " " + trg[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # edit_outer
        edit_outer = {}
        edit_outer['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
        edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['inputs'] = self.tok(f'Picture 1: <img>{image[0]}</img>\n{rephrase[0]} {trg[0]}', return_tensors='pt')["input_ids"]
        edit_outer['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + rephrase[0] + " " + trg[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = torch.stack(image_rephrase, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image_rephrase
        edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['inputs'] = self.tok(f'Picture 1: <img>{image_rephrase[0]}</img>\n{src[0]} {trg[0]}', return_tensors='pt')["input_ids"]
        edit_outer_image['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + src[0] + " " + trg[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        
        # loc
        loc = {}
        loc['image'] = torch.zeros(1, 3, 448, 448) if "owl-2" in self.config.model_name else None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['inputs'] = self.tok(f"{loc_q[0]} {loc_a[0]}", return_tensors='pt')["input_ids"]
        loc['input_ids'] = tokenizer_image_token(loc_q[0] + " " + loc_a[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # m_loc
        loc_image = {}
        loc_image['image'] = torch.stack(m_loc_image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else m_loc_image
        loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['inputs'] = self.tok(f'Picture 1: <img>{m_loc_image[0]}</img>\n{m_loc_q[0]} {m_loc_a[0]}', return_tensors='pt')["input_ids"]
        loc_image['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + m_loc_q[0] + " " + m_loc_a[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]


        # tv_loc  loc image with the rephrase text  Random_image Locality
        tv_loc_image = {}
        tv_loc_image['image'] = torch.stack(m_loc_image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else m_loc_image
        tv_loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(rephrase, trg)]
        tv_loc_image['inputs'] = self.tok(f'Picture 1: <img>{m_loc_image[0]}</img>\n{rephrase[0]} {trg[0]}', return_tensors='pt')["input_ids"]
        tv_loc_image['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + rephrase[0] + " " + trg[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        tv_loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in rephrase]
        tv_loc_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # tv_loc  loc image with the rephrase text  Random_image Locality
        t3i1 = {}
        t3i1['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
        t3i1['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        t3i1['inputs'] = self.tok(f'Picture 1: <img>{image[0]}</img>\n{m_loc_q[0]} {m_loc_a[0]}', return_tensors='pt')["input_ids"]
        t3i1['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + m_loc_q[0] + " " + m_loc_a[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        t3i1['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        t3i1['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]

##For train   Random_image Locality
        tv_loc_image_train = {}
        tv_loc_image_train['image'] = torch.stack(m_loc_image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else m_loc_image
        tv_loc_image_train['text_input'] = [" ".join([q, a]) for q, a in zip(src, trg)]
        tv_loc_image_train['inputs'] = self.tok(f'Picture 1: <img>{m_loc_image[0]}</img>\n{src[0]} {trg[0]}', return_tensors='pt')["input_ids"]
        tv_loc_image_train['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + src[0] + " " + trg[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        tv_loc_image_train['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in src]
        tv_loc_image_train['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]



        # close_edit  close image with close question

        # ran_sen_prompt, sim_image_paths = self.finds_sim(src)  Image-consistent locality   low is better
        close_edit = {}
        close_edit['image'] = torch.stack(sim_image, dim=0)
        close_edit['text_input'] = [self.prompt.format(q) + a for q, a in zip(sim_src, trg)]
        # close_edit['labels'] = [trg]
        close_edit['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in sim_src]
        close_edit['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]



        #src, trg
        t1i2 = {}
        t1i2['image'] = torch.stack(sim_image, dim=0)
        t1i2['text_input'] = [self.prompt.format(q) + a for q, a in zip(src, trg)]
        # close_edit['labels'] = [trg]
        t1i2['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in src]
        t1i2['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        #src, trg
        t2i1 = {}
        t2i1['image'] = torch.stack(image, dim=0)
        t2i1['text_input'] = [self.prompt.format(q) + a for q, a in zip(sim_src, trg)]
        # close_edit['labels'] = [trg]
        t2i1['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in sim_src]
        t2i1['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        t2i4 = {}
        t2i4['image'] =None
        t2i4['text_input'] = [self.prompt.format(q) + a for q, a in zip(sim_src, trg)]
        # close_edit['labels'] = [trg]
        t2i4['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in sim_src]
        t2i4['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        t1i4 = {}
        t1i4['image'] = None
        t1i4['text_input'] = [self.prompt.format(s) + f"{t}" for s, t in zip(src, trg)]
        t1i4['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            t1i4['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            t1i4['labels'] = self.tok.encode(trg, add_special_tokens=False, return_tensors="pt",)
        else:
            t1i4['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            t1i4['labels'] = self.tok.encode(trg, return_tensors="pt",)


        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        edit_ports = None
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                port['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
                port['text_input'] = [' '.join([port_q, port_a])]
                port['inputs'] = self.tok(f'Picture 1: <img>{image[0]}</img>\n{port_q} {port_a}', return_tensors='pt')["input_ids"]
                port['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + port_q + " " + port_a, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
                port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt",)["input_ids"]
                edit_ports.append(port)

        
        batch = {
            "edit_inner": edit_inner,#t1i1
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc, #t4i4
            "loc_image": loc_image,#t3i3
            'port': edit_ports,
            "cond": cond,
            #RI
            "tv_loc_image": tv_loc_image, #t1i3
            "t3i1":t3i1,
            "tv_loc_image_train": tv_loc_image_train,

            #CI
            "close_edit": close_edit, #t2i2
            "t1i2":t1i2,
            "t2i1":t2i1,
            "t1i4": t1i4,
            "t2i4": t2i4,
        }
        return dict_to(batch, self.config.device)
