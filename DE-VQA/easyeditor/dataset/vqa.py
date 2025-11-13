"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import pickle
from collections import OrderedDict
from copy import deepcopy

from sentence_transformers import SentenceTransformer, util


from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers

class VQADataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, name='./test.pkl',*args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        # vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("./multmodal/VLKEB-main/clip-vit-large-patch14-336")

        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        # vis_root = config.coco_image
        # rephrase_root = config.rephrase_image
        vis_root='./data/'
        rephrase_root='./data/'
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: "

        data = []

        if size is not None:
            self.annotation = self.annotation[:size]
        print(len(self.annotation))
        # if os.path.exists(name):
        #     data = torch.load(name)
        #     print('Loaded datasets from {}'.format(name))
        # else:
        #
        self.sentence_model = SentenceTransformer('/root/siton-data-9aa46a4f0e354f65bd8679947a35e67e/LMs/LMs/huggingface/all-MiniLM-L6-v2').to(config.device)



        if self.config.model_class == "LLaVA":
            with open(f'./multmodal/VLKEB-main/results/embedding/vqa_embeddings_llava.pkl', "rb") as fIn:
                stored_data = pickle.load(fIn)
                self.stored_sentences = stored_data['sentences']
                self.save_image_path = stored_data['images']
                self.prompts = stored_data['prompts']
                stored_embeddings = stored_data['embeddings']
            stored_embeddings = torch.tensor(stored_embeddings).to(0)
            self.stored_embeddings = util.normalize_embeddings(stored_embeddings)
        else:
            with open(
                    f'./multmodal/VLKEB-main/results/embedding/vqa_embeddings_llava.pkl',
                    "rb") as fIn:
                stored_data = pickle.load(fIn)
                self.stored_sentences = stored_data['sentences']
                self.save_image_path = stored_data['images']
                self.prompts = stored_data['prompts']
                stored_embeddings = stored_data['embeddings']
            stored_embeddings = torch.tensor(stored_embeddings).to('cuda:0')
            self.stored_embeddings = util.normalize_embeddings(stored_embeddings)



        for i, record in enumerate(self.annotation):

            if record['alt'] == "":
                continue

            image_path = os.path.join(self.vis_root, record["image"])
            rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
            locality_image_path = os.path.join(self.vis_root, record['m_loc'])

            image = Image.open(image_path).convert("RGB")
            rephrase_image = Image.open(rephrase_image_path).convert("RGB")
            locality_image = Image.open(locality_image_path).convert("RGB")

            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)

            sim_edit=  self.finds_sim(record['src'],record['alt'])
            sim_img=self.vis_processor(Image.open(sim_edit[1]).convert("RGB"))
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': image,
                'image_path': image_path,
                'image_rephrase': rephrase_image,
                'rephrase_image_path': rephrase_image_path,
                'sim_edit':sim_edit[0][0][0],
                'sim_edit_label':sim_edit[0][0][1],
                'sim_edit_image':sim_img,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }

            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']

            item['multimodal_locality_image'] = locality_image
            item['locality_image_path'] = locality_image_path
            item['multimodal_locality_prompt'] = record['m_loc_q']
            item['multimodal_locality_ground_truth'] = record['m_loc_a']
            data.append(item)

        # if size is not None:
        #     data = data[:size]
        # torch.save(data,name)

        self._data = data
        print('load:', len(self._data))

    def __getitem__(self, index):
        # print(self._data[index]['image'])

        data = deepcopy(self._data[index])

        # if self.config.model_class == "LLaVA":
        #     data['image'] = self.vis_processor(Image.open(self._data[index]['image']).convert("RGB"), return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        #     data['image_rephrase'] = self.vis_processor(Image.open(self._data[index]['image_rephrase']).convert("RGB"), return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        #     data['multimodal_locality_image'] = self.vis_processor(Image.open(self._data[index]['multimodal_locality_image']).convert("RGB"), return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        # else:
        #
        #     data['image']= self.vis_processor(Image.open(self._data[index]['image']).convert("RGB") )
        #     data['image_rephrase']=self.vis_processor( Image.open(self._data[index]['image_rephrase']).convert("RGB") )
        #     data['multimodal_locality_image']=self.vis_processor(  Image.open(self._data[index]['multimodal_locality_image']).convert("RGB") )

        return data
    
    def __len__(self):
        return len(self._data)

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

        # if self.config.model_class == "LLaVA":
        # imgs= self.vis_processor(Image.open(imgs).convert("RGB"), return_tensors='pt')['pixel_values'].to(dtype=torch.float16)



        #
        # close_examples = [ self.prompts[hit[k]["corpus_id"]] for k in range(len(hit))][-1]
        #
        # icl_image = [self.save_image_path[hit[k]["corpus_id"]] for k in range(len(hit))][-1]

        return close_examples,imgs


    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]
        sim_src=''#[b['sim_edit'] for b in batch]
        sim_image=''#[b['sim_edit_image'] for b in batch]

        # edit_inner
        edit_inner = {}
        edit_inner['image'] = image# torch.stack(image, dim=0)
        edit_inner['text_input'] = [self.prompt.format(s) + f"{t}" for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok.encode(trg, add_special_tokens=False, return_tensors="pt",)
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_inner['labels'] = self.tok.encode(trg, return_tensors="pt",)

        edit_inner['image_path'] = [b['image_path'] for b in batch]
        edit_inner['ori_qs'] = src
        # edit_inner['pid'] = [b['pid'] for b in batch]
        edit_inner['prompt'] = src
        edit_inner['target'] = trg
        edit_inner['rephrase_prompt'] = rephrase
        edit_inner['rephrase_text_input'] = [self.prompt.format(r) + t for r, t in zip(rephrase, trg)]
        edit_inner['locality_prompt'] = loc_q
        edit_inner['locality_ground_truth'] = loc_a
        # edit_inner['source'] = sources
        edit_inner['cat'] = 'edit'
        edit_inner['target_new'] = trg
        # edit_outer
        edit_outer = {}
        edit_outer['image'] = image#torch.stack(image, dim=0)
        edit_outer['text_input'] = [self.prompt.format(r) + f"{t}" for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_outer['prompts_len'] = [len(self.tok.encode(self.prompt.format(r), add_special_tokens=False)) for r in rephrase]
            edit_outer['labels'] = self.tok.encode(trg, add_special_tokens=False, return_tensors="pt",)
        else:
            edit_outer['prompts_len'] = [len(self.tok.encode(self.prompt.format(r))) for r in rephrase]
            edit_outer['labels'] = self.tok.encode(trg, return_tensors="pt",)
        edit_outer['cat'] = 'rephrase'
        edit_outer['prompt'] = rephrase
        edit_outer['target'] = trg

        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] =image_rephrase# torch.stack(image_rephrase, dim=0)
        edit_outer_image['text_input'] = [self.prompt.format(s) + f"{t}" for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_outer_image['labels'] = self.tok.encode(trg, add_special_tokens=False, return_tensors="pt",)
        else:
            edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_outer_image['labels'] = self.tok.encode(trg, return_tensors="pt",)

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

        # loc
        loc = {}
        loc['image'] = None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
            loc['labels'] = self.tok.encode(loc_a, add_special_tokens=False, return_tensors="pt",)
        else:
            loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
            loc['labels'] = self.tok.encode(loc_a, return_tensors="pt",)


        loc['locality_prompt'] = loc_q
        loc['prompt'] = loc_q
        loc['target'] = loc_a
        loc['cat'] = 'locality_prompt'
        # m_loc
        loc_image = {}
        loc_image['image'] = m_loc_image#torch.stack(m_loc_image, dim=0)
        loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
            loc_image['labels'] = self.tok.encode(m_loc_a, add_special_tokens=False, return_tensors="pt",)
        else:
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
            loc_image['labels'] = self.tok.encode(m_loc_a, return_tensors="pt",)


        loc_image['multimodal_locality_prompt'] = m_loc_q
        loc_image['prompt'] = m_loc_q
        loc_image['target'] = m_loc_a
        loc_image['cat'] = 'multimodal_locality_image'
        # tv_loc  loc image with the rephrase text   Random_image Locality
        tv_loc_image = {}
        tv_loc_image['image'] = m_loc_image#torch.stack(m_loc_image, dim=0)
        tv_loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(rephrase, trg)]
        tv_loc_image['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            tv_loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in rephrase]
            tv_loc_image['labels'] = self.tok.encode(trg, add_special_tokens=False, return_tensors="pt",)
        else:
            tv_loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in rephrase]
            tv_loc_image['labels'] = self.tok.encode(trg, return_tensors="pt",)

        ##For train
        tv_loc_image_train = {}
        tv_loc_image_train['image'] =m_loc_image# torch.stack(m_loc_image, dim=0)
        tv_loc_image_train['text_input'] = [self.prompt.format(q) + a for q, a in zip(src, trg)]
        tv_loc_image_train['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            tv_loc_image_train['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in src]
            tv_loc_image_train['labels'] = self.tok.encode(trg, add_special_tokens=False, return_tensors="pt",)
        else:
            tv_loc_image_train['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in src]
            tv_loc_image_train['labels'] = self.tok.encode(trg, return_tensors="pt",)

        #t3i1
        t3i1 = {}
        t3i1['image'] = image#torch.stack(image, dim=0)
        t3i1['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        t3i1['labels'] = m_loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            t3i1['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
            t3i1['labels'] = self.tok.encode(m_loc_a, add_special_tokens=False, return_tensors="pt",)
        else:
            t3i1['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
            t3i1['labels'] = self.tok.encode(m_loc_a, return_tensors="pt",)



        #Image-consistent locality   low is better
        # close_edit = {}
        # close_edit['image'] = torch.stack(sim_image, dim=0)
        # close_edit['text_input'] = [self.prompt.format(q) + a for q, a in zip(sim_src, trg)]
        # # close_edit['labels'] = [trg]
        # close_edit['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in sim_src]
        # close_edit['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        #
        #
        # #src, trg
        # t1i2 = {}
        # t1i2['image'] = torch.stack(sim_image, dim=0)
        # t1i2['text_input'] = [self.prompt.format(q) + a for q, a in zip(src, trg)]
        # # close_edit['labels'] = [trg]
        # t1i2['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in src]
        # t1i2['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        #
        # #src, trg
        # t2i1 = {}
        # t2i1['image'] = torch.stack(image, dim=0)
        # t2i1['text_input'] = [self.prompt.format(q) + a for q, a in zip(sim_src, trg)]
        # # close_edit['labels'] = [trg]
        # t2i1['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in sim_src]
        # t2i1['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        #
        # #Image-consistent locality   low is better
        # t2i4 = {}
        # t2i4['image'] =None
        # t2i4['text_input'] = [self.prompt.format(q) + a for q, a in zip(sim_src, trg)]
        # # close_edit['labels'] = [trg]
        # t2i4['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in sim_src]
        # t2i4['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        #

        edit_ports = None
        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            "cond": cond,
            'port': edit_ports,
            "tv_loc_image":tv_loc_image,
            # "close_edit":close_edit,
            'tv_loc_image_train':tv_loc_image_train,
            # "t3i1": t3i1,
            # "t1i2": t1i2,
            # "t2i1": t2i1,
            # "t1i4": t1i4,
            # "t2i4": t2i4,
        }

        return dict_to(batch, self.config.device)
