from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime
import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)
from tqdm import tqdm
from transformers import AutoTokenizer
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

LOG = logging.getLogger(__name__)


def kl_value(pre, post):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.contiguous().view(-1, pre.shape[-1])
    post_ = post.contiguous().view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                    (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        if pre_.shape[-1] > 1:
            kl = (
                    pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))
            ).sum(-1)
            return kl.sum()
def cosine_similarity(x, y):
    # 计算余弦相似度
    dot_product = torch.sum(x * y, dim=-1)
    norm_x = x.norm(dim=-1)
    norm_y = y.norm(dim=-1)
    return (dot_product / (norm_x * norm_y)).numpy()


def show_s(outputs,k,tokenizer,tilte='',indexs='',only_text=False,image_tokens=32):
    return 0
    if not os.path.exists(f'./multmodal/VLKEB-main/results/fin_scores/{indexs}/{tilte}.pkl'):  # 文件不存在
            pass
    else:
        return 'Exits'
    if image_tokens!=32:
        middle_layer = outputs.hidden_states[33:]


        if outputs.logits.shape[1]>image_tokens:
            query_lens = k['prompts_len'][0] + image_tokens
            custom_labels = ['T'] * image_tokens + [tokenizer.decode(i) for i in outputs.opt_tokens['input_ids'][0]]
            custom_labels = custom_labels[:query_lens]
            h0s = [i[0][0][:query_lens] for i in middle_layer]  # input
            h1s = [i[3][0][:query_lens] for i in middle_layer]  # attention
            h2s = [i[6][0][:query_lens] for i in middle_layer]  # mlp
            # h_sum=[i+j+k  for i,j,k in zip(h0s,h1s,h2s)]
            hiddens = [h[0][:query_lens] for h in outputs.hidden_states[1:33]]
            atts = [i[:, :, :query_lens, :query_lens] for i in outputs.layers_outs]
        if only_text or  outputs.logits.shape[1]<image_tokens:
            query_lens = k['prompts_len'][0]
            custom_labels = [tokenizer.decode(i) for i in outputs.opt_tokens['input_ids'][0]]
            custom_labels = custom_labels[:query_lens]
            h0s = [i[0][0][:query_lens] for i in middle_layer]  # input
            h1s = [i[3][0][:query_lens] for i in middle_layer]  # attention
            h2s = [i[6][0][:query_lens] for i in middle_layer]  # mlp
            # h_sum=[i+j+k  for i,j,k in zip(h0s,h1s,h2s)]
            hiddens = [h[0][:query_lens] for h in outputs.hidden_states[1:33]]
            atts = [i[:, :, :query_lens, :query_lens] for i in outputs.layers_outs]
    else:
        middle_layer = outputs.hidden_states[34:]


        if outputs.logits.shape[1]>image_tokens:
            query_lens = k['prompts_len'][0] + image_tokens
            custom_labels = ['T'] * image_tokens + [tokenizer.decode(i) for i in outputs.opt_tokens['input_ids'][0]]
            custom_labels = custom_labels[:query_lens]
            h0s = [i[0][0][:query_lens] for i in middle_layer]  # input
            h1s = [i[3][0][:query_lens] for i in middle_layer]  # attention
            h2s = [i[6][:query_lens] for i in middle_layer]  # mlp
            # h_sum=[i+j+k  for i,j,k in zip(h0s,h1s,h2s)]
            hiddens = [h[0][:query_lens] for h in outputs.hidden_states[1:34]]
            atts = [i[:, :, :query_lens, :query_lens] for i in outputs.layers_outs]
        if only_text or  outputs.logits.shape[1]<image_tokens:
            query_lens = k['prompts_len'][0]
            custom_labels = [tokenizer.decode(i) for i in outputs.opt_tokens['input_ids'][0]]
            custom_labels = custom_labels[:query_lens]
            h0s = [i[0][0][:query_lens] for i in middle_layer]  # input
            h1s = [i[3][0][:query_lens] for i in middle_layer]  # attention
            h2s = [i[6][:query_lens] for i in middle_layer]  # mlp
            # h_sum=[i+j+k  for i,j,k in zip(h0s,h1s,h2s)]
            hiddens = [h[0][:query_lens] for h in outputs.hidden_states[1:34]]
            atts = [i[:, :, :query_lens, :query_lens] for i in outputs.layers_outs]
        hiddens=hiddens[:-1]

    # 计算每个张量的归一化贡献比例
    contribution_a = np.array([(b - a).norm(dim=-1).cpu().tolist() for a, b in zip(h0s, hiddens)])
    contribution_b = np.array([(b - a).norm(dim=-1).cpu().tolist() for a, b in zip(h1s, hiddens)])
    contribution_c = np.array([(b - a).norm(dim=-1).cpu().tolist() for a, b in zip(h2s, hiddens)])


    cos_contribution_a, cos_contribution_b, cos_contribution_c = [], [], []
    for i in range(len(hiddens)):
        # 计算余弦相似度
        cos_contribution_a.append(cosine_similarity(h0s[i].detach().cpu(), hiddens[i].detach().cpu()))
        cos_contribution_b.append(cosine_similarity(h1s[i].detach().cpu(), hiddens[i].detach().cpu()))
        cos_contribution_c.append(cosine_similarity(h2s[i].detach().cpu(), hiddens[i].detach().cpu()))

    # 找出每个位置上的最大贡献
    # max_indices = np.argmax([cos_contribution_a, cos_contribution_b, cos_contribution_c], axis=0)

    uses_nodes=[]
    edges = []
    chooses = [[(len(custom_labels) - 1)]]
    token_score = [[0 for _ in range(len(custom_labels))] for l in range(32)]
    coff=0.03
    for i in range(31, -1, -1):
        # attention level
        indx = 0# 跳的次数
        while indx < len(chooses[31 - i]):
            j = chooses[31 - i][indx]

            norm_score_b = contribution_b[i, j] / (
                    contribution_a[i, j] + contribution_b[i, j] + contribution_c[i, j])

            if (abs(cos_contribution_b[i][j]) > 0.4 and norm_score_b > 0.3) or norm_score_b > 0.6 or abs(
                    cos_contribution_b[i][j]) > 0.6:  # att
                # tops = torch.topk(atts[i][0].mean(dim=0)[j], k=6)[1].tolist()
                top_score, tops = torch.topk(atts[i][0].mean(dim=0)[j], k=8)

                for ti, t in enumerate(tops.tolist()):
                    edges.append((f'L{i}N{t}', f'L{i}N{j}'))
                    token_score[i][t] += 1/(indx+1) * norm_score_b * top_score[ti].item()
                    if t < 32 or j < 32:
                        uses_nodes.append(f'L{i}N{t}__L{i}N{j}')
                    if t not in chooses[31 - i]:
                        chooses[31 - i].append(t)

            indx += 1

        # layer level
        temps = []
        for j in chooses[31 - i]:
            norm_score_a = contribution_a[i, j] / (
                    contribution_a[i, j] + contribution_b[i, j] + contribution_c[i, j])

            norm_score_c = contribution_c[i, j] / (
                    contribution_a[i, j] + contribution_b[i, j] + contribution_c[i, j])
            if (abs(cos_contribution_a[i][j]) > 0.4 and norm_score_a > 0.3) or norm_score_a > 0.7 or abs(
                    cos_contribution_a[i][j]) > 0.7:  # h0
                token_score[i][j] += 1/(indx+1) * norm_score_a
                temps.append(j)
            if (abs(cos_contribution_c[i][j]) > 0.4 and norm_score_c > 0.3) or norm_score_c > 0.7 or abs(
                    cos_contribution_c[i][j]) > 0.7:  # mlp
                token_score[i][j] += 1/(indx+1)  * norm_score_c
                temps.append(j)
        for t in temps:
            if i - 1 > -1:
                edges.append((f'L{i - 1}N{t}', f'L{i}N{t}'))
        chooses.append(list(set(temps)))

    if not os.path.exists(f'./multmodal/VLKEB-main/results/fin_scores/{indexs}'):
        os.makedirs(f'./multmodal/VLKEB-main/results/fin_scores/{indexs}')

    torch.save([edges,uses_nodes,tilte,token_score],f'./multmodal/VLKEB-main/results/fin_scores/{indexs}/{tilte}.pkl')

class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]
    def find_def(self,before_outer_outputs,befer_edit_dict,post_edit_outputs,post_edit_dict):
        print('pre results:',befer_edit_dict)
        print('after  results:',post_edit_dict)
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        before_attention=[i[1].mean(dim=1)[0]  for i in before_outer_outputs.layers_outs]
        after_attention=[i[1].mean(dim=1)[0] for i in post_edit_outputs.layers_outs]
        diffs=[  np.abs((a-b).cpu()*100) for a,b in zip(before_attention,after_attention)]

        for ai,a in enumerate(diffs):
            plt.figure(figsize=(12, 10))
            plt.imshow(a, cmap='bwr', interpolation='nearest')
            # 添加颜色条以指示数值大小
            plt.colorbar()
            # 添加标题和标签（可选）
            plt.title('Heatmap of Random N×N Matrix')
            plt.title(f'attention diffs {28+ai}')

            # 显示热图
            # plt.show()

            plt.savefig(f'./multmodal/VLKEB-main/shows/{self.config.loss}{cur_time} attention {28+ai} with_loss.png')
        # # 找出绝对值最大的Top 10%的元素的索引
        # threshold = np.percentile(diffs, 99)  # 计算差异的90百分位数
        # # 获取大于或等于阈值的元素的索引
        # top_10_percent_indices = np.argwhere(diffs >= threshold)
        # # 输出这些索引
        # print("Indices and values of the top 10% largest absolute differences:")
        # for index in top_10_percent_indices:
        #     idx_tuple = tuple(index)
        #     value = diffs[idx_tuple[0]][idx_tuple[1]][idx_tuple[2]]
        #     print(f"Index: {idx_tuple}, Value: {value}")


        befoer_hiddens=[i[2]  for i in before_outer_outputs.layers_outs ]
        after_hiddens=[i[2]  for i in post_edit_outputs.layers_outs ]
        diffs_hidden=[[(aa-bb).norm(dim=-1).cpu() for aa,bb in zip(a,b)]  for a,b in zip(befoer_hiddens,after_hiddens) ]
        #before_fc1,after_fc1,after_act,after_fc2,after_dop,after_res
        keys=['before_fc1','after_fc1','after_act','after_fc2','after_dop','after_res']

        all_keys=[k+l for l in ['28','29','30','31'] for k in keys]
        all_difs=[ dd[0] if len(dd)==1 else dd for d in diffs_hidden for dd in d]
        plt.figure(figsize=(12, 10))
        plt.imshow(all_difs, cmap='bwr', interpolation='nearest')

        # 添加颜色条以指示数值大小
        plt.colorbar()
        # 添加标题和标签（可选）
        plt.title('Heatmap of Random N×N Matrix')
        plt.yticks(ticks=range(24), labels=all_keys)
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.title(f'hidden state diffs, pre res: {befer_edit_dict["acc"].item()} after res: {post_edit_dict["acc"].item()}')

        # 显示热图
        # plt.show()

        plt.savefig(f'./multmodal/VLKEB-main/shows/{self.config.loss} {cur_time} mlps.png')
        print('Generate Finish')

        torch.save([diffs,all_keys,all_difs],f'./multmodal/VLKEB-main/shows/{cur_time}{self.config.loss}.pkl')






    def shows_heat_map(self,matrix):
        # 创建热图


        for mi,mat in enumerate(matrix):
            plt.imshow(mat.cpu(), cmap='bwr', interpolation='nearest')

            # 添加颜色条以指示数值大小
            plt.colorbar()
            # 添加标题和标签（可选）
            plt.title('Heatmap of Random N×N Matrix')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')
            plt.title(str(mi))

            # 显示热图
            plt.show()
    def check_logits(self,inputs):
        if not isinstance(inputs, torch.Tensor):
            logits = inputs.logits
        else:
            logits = inputs

        return logits
    def edit_step(self, batch, training: bool,indexs=''):
        self.model.train(training)
        self.original_model.train(training)
        if True:#self.config.eval_only:
            try:
                tokenizer=  self.model.model.opt_tokenizer
                image_tokens = 32
            except:
                tokenizer = self.model.model.llama_tokenizer
                image_tokens = 43
        ####################################################################################################
        with torch.no_grad():
            #t4i4
            base_outputs = self.model(batch["loc"])
            if not isinstance(base_outputs, torch.Tensor):
                base_logits = base_outputs.logits
            else:
                base_logits = base_outputs

            # t3i3
            base_image_outputs = self.model(batch["loc_image"])
            if not isinstance(base_image_outputs, torch.Tensor):
                base_image_logits = base_image_outputs.logits
            else:
                base_image_logits = base_image_outputs

            #t1i3
            tv_loc_image_outputs = self.model(batch["tv_loc_image"])
            if not isinstance(base_image_outputs, torch.Tensor):
                eval_loc_image_logits = tv_loc_image_outputs.logits
            else:
                eval_loc_image_logits = tv_loc_image_outputs


            if True:#self.config.eval_only:

                show_s(base_image_outputs, batch["loc_image"], tokenizer,
                       tilte=f't3i3_before_{self.config.model_name}', indexs=indexs,image_tokens=image_tokens)

                show_s(tv_loc_image_outputs, batch["tv_loc_image"], tokenizer,
                       tilte=f't1i3_before_{self.config.model_name}', indexs=indexs,image_tokens=image_tokens)

                ot_post_edit_outputs = self.model(batch["edit_outer"])
                show_s(ot_post_edit_outputs, batch["edit_outer"], tokenizer,
                       tilte=f'edit_outer_before_{self.config.model_name}',indexs=indexs,image_tokens=image_tokens)
                #t2i2
                t2i2_outputs = self.model(batch["close_edit"])
                show_s(t2i2_outputs, batch["close_edit"], tokenizer,
                       tilte=f't2i2_before_{self.config.model_name}', indexs=indexs,image_tokens=image_tokens)

                #t2i1
                t2i1_outputs_before = self.model(batch["t2i1"])
                show_s(t2i1_outputs_before, batch["t2i1"], tokenizer,
                       tilte=f't2i1_before_{self.config.model_name}', indexs=indexs,image_tokens=image_tokens)
                #t1i2
                t1i2_outputs_before = self.model(batch["t1i2"])
                show_s(t1i2_outputs_before, batch["t1i2"], tokenizer,
                       tilte=f't1i2_before_{self.config.model_name}', indexs=indexs,image_tokens=image_tokens)

                #t1i2
                t3i1_outputs_before = self.model(batch["t3i1"])
                show_s(t3i1_outputs_before, batch["t3i1"], tokenizer,
                       tilte=f't3i1_before_{self.config.model_name}', indexs=indexs,image_tokens=image_tokens)

                t2i4 = batch["t2i4"]
                t2i4_outputs_before = self.model(t2i4)


                t1i4 = batch["t1i4"]
                t1i4_outputs_before = self.model(t1i4)

            # else:
            if self.config.loss == '1':
                #tv_loc_image
                tv_base_image_outputs = self.model(batch["tv_loc_image_train"])
                if not isinstance(tv_base_image_outputs, torch.Tensor):
                    loc_image_logits = tv_base_image_outputs.logits
                else:
                    loc_image_logits = tv_base_image_outputs
                #only text for edits
                only_text__edit_outputs =  self.model({i:j if i!='image' else None for i,j in batch["edit_inner"].items()})
                if not isinstance(only_text__edit_outputs, torch.Tensor):
                    only_text_logits = only_text__edit_outputs.logits
                else:
                    only_text_logits = only_text__edit_outputs
            if self.config.loss == '2' or self.config.loss== '5' or self.config.loss== '7':
                tv_base_image_outputs = self.model(batch["tv_loc_image_train"])
                if not isinstance(tv_base_image_outputs, torch.Tensor):
                    loc_image_logits = tv_base_image_outputs.logits
                else:
                    loc_image_logits = tv_base_image_outputs
            if self.config.loss == '3' or self.config.loss== '5' or self.config.loss== '6' or self.config.loss=='cl_n':
                only_text__edit_outputs =  self.model({i:j if i!='image' else None for i,j in batch["edit_inner"].items()})
                if not isinstance(only_text__edit_outputs, torch.Tensor):
                    only_text_logits = only_text__edit_outputs.logits
                else:
                    only_text_logits = only_text__edit_outputs

            if self.config.loss == '3IC':
                only_text__edit_outputs =  self.model({i:j if i!='image' else None for i,j in batch["close_edit"].items()})
                if not isinstance(only_text__edit_outputs, torch.Tensor):
                    only_text_logits = only_text__edit_outputs.logits
                else:
                    only_text_logits = only_text__edit_outputs

            if self.config.loss == '4' or self.config.loss== '5' or self.config.loss== '6'or self.config.loss== '7':
                before_close_edit_outputs = self.model(batch["close_edit"])
                if not isinstance(before_close_edit_outputs, torch.Tensor):
                    close_edit_logits = before_close_edit_outputs.logits
                else:
                    close_edit_logits = before_close_edit_outputs







        ####################################################################################################


        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edit_time = time.time() - start

        l_total, l_edit, l_loc, l_base = 0, 0, 0, 0
        info_dict = {}

        # Your model inference or training code
        ####################################################################################################
        with torch.set_grad_enabled(training):
            # Editing loss
            post_edit_outputs = edited_model(batch["edit_outer"])
            post_batch_labels = batch["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs


            # rephrase image
            post_image_edit_outputs = edited_model(batch["edit_outer_image"])
            post_image_batch_labels = batch["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs



            inner_edit_outputs = edited_model(batch["edit_inner"])
            inner_batch_labels = batch["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)["nll"]
            else:
                l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])["nll"]
            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:
                l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)["nll"]
            else:
                l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])["nll"]

            # Collect some useful metrics
            with torch.no_grad():
                if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                    post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
                else:
                    post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])

                if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
                else:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])

                if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:
                    image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
                else:
                    image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])



            post_base_outputs = edited_model(batch["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
                kl_mask = post_base_outputs.attention_mask
            else:
                post_base_logits = post_base_outputs
                kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(post_base_logits.device)

            post_image_base_outputs = edited_model(batch["loc_image"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
                kl_image_mask = post_image_base_outputs.attention_mask
            else:
                post_image_base_logits = post_image_base_outputs
                kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(base_image_logits.device)

            if self.config.loss == '1' :
                ## add kl loss for tv and only text
                post_tv_outputs = edited_model(batch["tv_loc_image_train"]) #
                if not isinstance(post_tv_outputs, torch.Tensor):
                    post_tv_logits = post_tv_outputs.logits
                    tv_kl_mask = post_tv_outputs.attention_mask
                else:
                    post_tv_logits = post_tv_outputs
                    tv_kl_mask = torch.ones(post_tv_outputs.shape[0], post_tv_outputs.shape[1]).to(post_tv_outputs.device)

                post_only_text_outputs = edited_model({i:j if i!='image' else None for i,j in batch["edit_inner"].items()})
                if not isinstance(post_only_text_outputs, torch.Tensor):
                    post_only_text_logits = post_only_text_outputs.logits
                    only_textkl_image_mask = post_only_text_outputs.attention_mask
                else:
                    post_only_text_logits = post_only_text_outputs
                    only_textkl_image_mask = torch.ones(post_only_text_logits.shape[0], post_only_text_logits.shape[1]).to(
                        base_image_logits.device)
            if self.config.loss == '2'  or self.config.loss== '5' or self.config.loss== '7' or  self.config.loss == 'cl' or self.config.loss=='cl_n':#RI
                post_tv_outputs = edited_model(batch["tv_loc_image_train"])
                if not isinstance(post_tv_outputs, torch.Tensor):
                    post_tv_logits = post_tv_outputs.logits
                    tv_kl_mask = post_tv_outputs.attention_mask
                else:
                    post_tv_logits = post_tv_outputs
                    tv_kl_mask = torch.ones(post_tv_outputs.shape[0], post_tv_outputs.shape[1]).to(post_tv_outputs.device)
            if self.config.loss == '3'  or self.config.loss== '5'or self.config.loss== '6' or self.config.loss=='cl_n':
                post_only_text_outputs = edited_model({i:j if i!='image' else None for i,j in batch["edit_inner"].items()})
                if not isinstance(post_only_text_outputs, torch.Tensor):
                    post_only_text_logits = post_only_text_outputs.logits
                    only_textkl_image_mask = post_only_text_outputs.attention_mask
                else:
                    post_only_text_logits = post_only_text_outputs
                    only_textkl_image_mask = torch.ones(post_only_text_logits.shape[0], post_only_text_logits.shape[1]).to(
                        base_image_logits.device)

            if self.config.loss=='3IC':
                post_only_text_outputs = edited_model(
                    {i: j if i != 'image' else None for i, j in batch["close_edit"].items()})
                if not isinstance(post_only_text_outputs, torch.Tensor):
                    post_only_text_logits = post_only_text_outputs.logits
                    only_textkl_image_mask = post_only_text_outputs.attention_mask
                else:
                    post_only_text_logits = post_only_text_outputs
                    only_textkl_image_mask = torch.ones(post_only_text_logits.shape[0],
                                                        post_only_text_logits.shape[1]).to(base_image_logits.device)

            if self.config.loss == '4' or self.config.loss== '5' or self.config.loss== '6' or self.config.loss== '7' or  self.config.loss == 'cl' or self.config.loss=='cl_n':
                close_post_edit_outputs = edited_model(batch["close_edit"])

                if not isinstance(close_post_edit_outputs, torch.Tensor):
                    post_close_edit_logits = close_post_edit_outputs.logits
                    close_edit_mask = close_post_edit_outputs.attention_mask
                else:
                    post_close_edit_logits = close_post_edit_outputs
                    close_edit_mask = torch.ones(close_post_edit_outputs.shape[0], close_post_edit_outputs.shape[1]).to(
                        close_post_edit_outputs.device)



        if not self.config.eval_only:
            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)
            l_image_loc = kl_loc_loss(base_image_logits.detach(), post_image_base_logits, mask=kl_image_mask)

            if l_image_loc.isnan():
                print("l_image_loc is nan")
                print("input: ", batch["loc_image"]['text_input'])
                l_image_loc=0

            if l_edit.isnan():
                print("l_edit is nan")
                print("input: ", batch["edit_outer"]['text_input'])
                l_edit=0

            if l_image_edit.isnan():
                print("l_image_edit is nan")
                print("input: ", batch["edit_outer_image"]['text_input'])
                l_image_edit=0

            if l_loc.isnan():
                print("l_loc is nan")
                print("input: ", batch["loc"]['text_input'])
                l_loc=0


            if self.config.alg == "SERAC_MULTI":
                l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc + self.config.iedit * l_image_edit
            else:
                if self.config.loss=='0':
                    l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc+l_image_loc) + self.config.iedit * l_image_edit
                if self.config.loss=='1': #l_loc_tv(TC) , l_text_only_loc (TO)
                    l_loc_tv = kl_loc_loss(loc_image_logits.detach(), post_tv_logits, mask=tv_kl_mask)
                    l_text_only_loc = kl_loc_loss(only_text_logits.detach(), post_only_text_logits,
                                                  mask=only_textkl_image_mask)
                    l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc + l_image_loc) + self.config.iedit * l_image_edit+ l_loc_tv + l_text_only_loc
                if self.config.loss=='2':#l_loc_tv(TC)
                    l_loc_tv = kl_loc_loss(loc_image_logits.detach(), post_tv_logits, mask=tv_kl_mask)
                    l_total_edit = self.config.cedit * l_edit + self.config.cloc * (
                                l_loc + l_image_loc) + self.config.iedit * l_image_edit + l_loc_tv
                if self.config.loss=='3':#l_text_only_loc (TO)
                    l_text_only_loc = kl_loc_loss(only_text_logits.detach(), post_only_text_logits,
                                                  mask=only_textkl_image_mask)
                    l_total_edit = self.config.cedit * l_edit + self.config.cloc * (
                            l_loc + l_image_loc) + self.config.iedit * l_image_edit + l_text_only_loc

                if self.config.loss=='3IC':#NO image for IC
                    l_text_only_loc = kl_loc_loss(only_text_logits.detach(), post_only_text_logits,
                                                  mask=only_textkl_image_mask)
                    l_total_edit = self.config.cedit * l_edit + self.config.cloc * (
                            l_loc + l_image_loc) + self.config.iedit * l_image_edit + l_text_only_loc
                if self.config.loss=='4':#same question with the image (IC)
                    l_ic=kl_loc_loss(close_edit_logits.detach(), post_close_edit_logits,
                                                  mask=close_edit_mask)
                    l_total_edit = self.config.cedit * l_edit + self.config.cloc * (
                            l_loc + l_image_loc) + self.config.iedit * l_image_edit + l_ic
                if  self.config.loss== '5':#IC+TO+TC
                    l_loc_tv = kl_loc_loss(loc_image_logits.detach(), post_tv_logits, mask=tv_kl_mask)

                    l_text_only_loc = kl_loc_loss(only_text_logits.detach(), post_only_text_logits,
                                                  mask=only_textkl_image_mask)
                    l_ic = kl_loc_loss(close_edit_logits.detach(), post_close_edit_logits,
                                       mask=close_edit_mask)
                    l_total_edit = self.config.cedit * l_edit + self.config.cloc * (
                            l_loc + l_image_loc) + self.config.iedit * l_image_edit + l_loc_tv+l_text_only_loc+l_ic
                if self.config.loss=='6':##IC+TO
                    l_text_only_loc = kl_loc_loss(only_text_logits.detach(), post_only_text_logits,
                                                  mask=only_textkl_image_mask)
                    l_ic = kl_loc_loss(close_edit_logits.detach(), post_close_edit_logits,
                                       mask=close_edit_mask)
                    l_total_edit = self.config.cedit * l_edit + self.config.cloc * (
                            l_loc + l_image_loc) + self.config.iedit * l_image_edit +l_text_only_loc+l_ic
                if self.config.loss=='7':#IC+TC
                    l_loc_tv = kl_loc_loss(loc_image_logits.detach(), post_tv_logits, mask=tv_kl_mask)

                    l_ic = kl_loc_loss(close_edit_logits.detach(), post_close_edit_logits,
                                       mask=close_edit_mask)
                    l_total_edit = self.config.cedit * l_edit + self.config.cloc * (
                            l_loc + l_image_loc) + self.config.iedit * l_image_edit + l_loc_tv +  l_ic

                if self.config.loss=='cl':
                    num_img_tokens=43 if self.config.model_name=='minigpt4' else 32
                    #ancho feature
                    anc_text=torch.sigmoid(inner_edit_outputs.logits[:,num_img_tokens:,:].mean(dim=1))
                    anc_image=torch.sigmoid(inner_edit_outputs.logits[:,:num_img_tokens,:].mean(dim=1))

                    #postive feature
                    pos_text=torch.sigmoid(post_image_edit_outputs.logits[:,num_img_tokens:,:].mean(dim=1))
                    pos_image=torch.sigmoid(post_image_edit_outputs.logits[:,:num_img_tokens,:].mean(dim=1))

                    #negtive1 for RI
                    neg1_text=torch.sigmoid(post_tv_outputs.logits[:,num_img_tokens:,:].mean(dim=1))
                    neg_1_image=torch.sigmoid(post_tv_outputs.logits[:,:num_img_tokens,:].mean(dim=1))

                    #negtive2 for IC
                    neg2_text=torch.sigmoid(close_post_edit_outputs.logits[:,num_img_tokens:,:].mean(dim=1))
                    neg_2_image=torch.sigmoid(close_post_edit_outputs.logits[:,:num_img_tokens,:].mean(dim=1))

                    # 计算相似度（余弦相似度）
                    anchor_feat=torch.cat((anc_image,anc_text),dim=-1)
                    positive_feat=torch.cat((pos_image,pos_text),dim=-1)
                    negative1_feat=torch.cat((neg_1_image,neg1_text),dim=-1)
                    negative2_feat=torch.cat((neg_2_image,neg2_text),dim=-1)


                    sim_positive = torch.nn.functional.cosine_similarity(anchor_feat, positive_feat)  # 目标：趋近1
                    sim_negative1 =  torch.nn.functional.cosine_similarity(anchor_feat, negative1_feat)  # 目标：趋近0
                    sim_negative2 =  torch.nn.functional.cosine_similarity(anchor_feat, negative2_feat)  # 目标：趋近0

                    # Triplet Margin Loss（拉近正样本，推远负样本）
                    margin = 0.5
                    cl_loss = torch.relu(sim_negative1 - sim_positive + margin) + \
                           torch.relu(sim_negative2 - sim_positive + margin)


                    l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc+l_image_loc) + self.config.iedit * l_image_edit+cl_loss
                if self.config.loss=='cl_n':
                    num_img_tokens=43 if self.config.model_name=='minigpt4' else 32
                    #ancho feature
                    anc_text=torch.sigmoid(inner_edit_outputs.logits[:,num_img_tokens:,:].mean(dim=1))
                    anc_image=torch.sigmoid(inner_edit_outputs.logits[:,:num_img_tokens,:].mean(dim=1))

                    #postive feature
                    pos_text=torch.sigmoid(post_image_edit_outputs.logits[:,num_img_tokens:,:].mean(dim=1))
                    pos_image=torch.sigmoid(post_image_edit_outputs.logits[:,:num_img_tokens,:].mean(dim=1))

                    #negtive1 for RI
                    neg1_text=torch.sigmoid(post_tv_outputs.logits[:,num_img_tokens:,:].mean(dim=1))
                    neg_1_image=torch.sigmoid(post_tv_outputs.logits[:,:num_img_tokens,:].mean(dim=1))

                    #negtive2 for IC
                    neg2_text=torch.sigmoid(close_post_edit_outputs.logits[:,num_img_tokens:,:].mean(dim=1))
                    neg_2_image=torch.sigmoid(close_post_edit_outputs.logits[:,:num_img_tokens,:].mean(dim=1))

                    # 计算相似度（余弦相似度）
                    anchor_feat=torch.cat((anc_image,anc_text),dim=-1)
                    positive_feat=torch.cat((pos_image,pos_text),dim=-1)
                    negative1_feat=torch.cat((neg_1_image,neg1_text),dim=-1)
                    negative2_feat=torch.cat((neg_2_image,neg2_text),dim=-1)


                    sim_positive = torch.nn.functional.cosine_similarity(anchor_feat, positive_feat)  # 目标：趋近1
                    sim_negative1 =  torch.nn.functional.cosine_similarity(anchor_feat, negative1_feat)  # 目标：趋近0
                    sim_negative2 =  torch.nn.functional.cosine_similarity(anchor_feat, negative2_feat)  # 目标：趋近0

                    # Triplet Margin Loss（拉近正样本，推远负样本）
                    margin = 0.5
                    cl_loss = torch.relu(sim_negative1 - sim_positive + margin) + \
                           torch.relu(sim_negative2 - sim_positive + margin)

                    l_text_only_loc = kl_loc_loss(only_text_logits.detach(), post_only_text_logits,
                                                  mask=only_textkl_image_mask)
                    l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc+l_image_loc) + self.config.iedit * l_image_edit+cl_loss+l_text_only_loc

        else:
            l_total_edit=0


        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)



        #loc for tv data
        tv_loc_image_outputs = edited_model(batch["tv_loc_image"])

        if not isinstance(tv_loc_image_outputs, torch.Tensor):
            tv_loc_image_logits = tv_loc_image_outputs.logits
        else:
            tv_loc_image_logits = tv_loc_image_outputs
        TV_post_edit_dict = self.model.edit_loss_fn(self.config, tv_loc_image_logits, batch["tv_loc_image"]['labels'])


        ## sim edit
        with torch.no_grad():

            # Text locality t4i4
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1,
                                                        dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1,
                                                   dim=-1).indices

            # Image locality  t3i3
            post_image_base_logits_softmax_top_k = torch.topk(
                torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10,
                                                         dim=-1).indices
            # t1i3
            post_tv_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(tv_loc_image_logits, dim=-1),
                                                      k=10,
                                                      dim=-1).indices
            base_tv_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(eval_loc_image_logits, dim=-1),
                                                      k=10,
                                                      dim=-1).indices

            #t2i2
            close_post_edit_outputs = edited_model(batch["close_edit"])

            close_post_batch_labels = batch["close_edit"]["labels"]
            if not isinstance(close_post_edit_outputs, torch.Tensor):
                close_post_edit_logits = close_post_edit_outputs.logits
            else:
                close_post_edit_logits = close_post_edit_outputs

            if close_post_edit_logits.shape[1] > close_post_batch_labels.shape[1]:
                close_post_edit_dict = self.model.edit_loss_fn(self.config, close_post_edit_logits, close_post_batch_labels)
            else:
                close_post_edit_dict = self.model.edit_loss_fn(self.config, close_post_edit_logits, close_post_batch_labels[:, -close_post_edit_logits.shape[1] - 1:])
            t2i2_before_top_k = torch.topk(torch.nn.functional.softmax(self.check_logits(t2i2_outputs), dim=-1),
                                                      k=10,
                                                      dim=-1).indices
            t2i2_after_top_k = torch.topk(torch.nn.functional.softmax(close_post_edit_logits, dim=-1),
                                                      k=10,
                                                      dim=-1).indices



            #t1i4
            only_text_ed=batch["t1i4"]
            o_post_edit_outputs = edited_model(only_text_ed)
            o_post_batch_labels = batch["t1i4"]["labels"]
            if not isinstance(o_post_edit_outputs, torch.Tensor):
                o_post_edit_logits = o_post_edit_outputs.logits
            else:
                o_post_edit_logits = o_post_edit_outputs

            if o_post_edit_logits.shape[1] > o_post_batch_labels.shape[1]:
                o_post_edit_dict = self.model.edit_loss_fn(self.config, o_post_edit_logits, o_post_batch_labels)
            else:
                o_post_edit_dict = self.model.edit_loss_fn(self.config, o_post_edit_logits,
                                                         o_post_batch_labels[:, -o_post_edit_logits.shape[1] - 1:])



            t1i4_before_top_k = torch.topk(torch.nn.functional.softmax(self.check_logits(t1i4_outputs_before), dim=-1),
                                                      k=10,
                                                      dim=-1).indices
            t1i4_after_top_k = torch.topk(torch.nn.functional.softmax(o_post_edit_logits, dim=-1),
                                                      k=10,
                                                      dim=-1).indices


            #t2i4
            t2i4=batch["t2i4"]
            t2i4_outputs = edited_model(t2i4)
            t2i4_labels = batch["edit_outer"]["labels"]
            if not isinstance(t2i4_outputs, torch.Tensor):
                t2i4_logits = t2i4_outputs.logits
            else:
                t2i4_logits = t2i4_outputs

            if t2i4_logits.shape[1] > t2i4_labels.shape[1]:
                t2i4_dict = self.model.edit_loss_fn(self.config, t2i4_logits, t2i4_labels)
            else:
                t2i4_dict = self.model.edit_loss_fn(self.config, t2i4_logits,
                                                         t2i4_labels[:, -t2i4_logits.shape[1] - 1:])


            t2i4_before_top_k = torch.topk(torch.nn.functional.softmax(self.check_logits(t2i4_outputs_before), dim=-1),
                                                      k=10,
                                                      dim=-1).indices
            t2i4_after_top_k = torch.topk(torch.nn.functional.softmax(t2i4_logits, dim=-1),
                                                      k=10,
                                                      dim=-1).indices


            #t2i1
            t2i1=batch["t2i1"]
            t2i1_outputs = edited_model(t2i1)
            t2i1_labels = batch["edit_outer"]["labels"]
            if not isinstance(t2i1_outputs, torch.Tensor):
                t2i1_logits = t2i1_outputs.logits
            else:
                t2i1_logits = t2i1_outputs

            if t2i1_logits.shape[1] > t2i1_labels.shape[1]:
                t2i1_dict = self.model.edit_loss_fn(self.config, t2i1_logits, t2i1_labels)
            else:
                t2i1_dict = self.model.edit_loss_fn(self.config, t2i1_logits,
                                                         t2i1_labels[:, -t2i1_logits.shape[1] - 1:])

            t2i1_before_top_k = torch.topk(torch.nn.functional.softmax(self.check_logits(t2i1_outputs_before), dim=-1),
                                                      k=10,
                                                      dim=-1).indices
            t2i1_after_top_k = torch.topk(torch.nn.functional.softmax(self.check_logits(t2i1_outputs), dim=-1),
                                                      k=10,
                                                      dim=-1).indices


            #t1i2
            t1i2=batch["t1i2"]
            t1i2_outputs = edited_model(t1i2)
            t1i2_labels = batch["edit_outer"]["labels"]
            if not isinstance(t1i2_outputs, torch.Tensor):
                t1i2_logits = t1i2_outputs.logits
            else:
                t1i2_logits = t1i2_outputs

            if t1i2_logits.shape[1] > t1i2_labels.shape[1]:
                t1i2_dict = self.model.edit_loss_fn(self.config, t1i2_logits, t1i2_labels)
            else:
                t1i2_dict = self.model.edit_loss_fn(self.config, t1i2_logits,
                                                         t1i2_labels[:, -t1i2_logits.shape[1] - 1:])

            t1i2_before_top_k = torch.topk(torch.nn.functional.softmax(self.check_logits(t1i2_outputs_before), dim=-1),
                                                      k=10,
                                                      dim=-1).indices
            t1i2_after_top_k = torch.topk(torch.nn.functional.softmax(self.check_logits(t1i2_outputs), dim=-1),
                                                      k=10,
                                                      dim=-1).indices

            #t3i1
            t3i1=batch["t3i1"]
            t3i1_outputs = edited_model(t3i1)
            t3i1_labels = batch["edit_outer"]["labels"]
            if not isinstance(t3i1_outputs, torch.Tensor):
                t3i1_logits = t3i1_outputs.logits
            else:
                t3i1_logits = t3i1_outputs

            if t3i1_logits.shape[1] > t3i1_labels.shape[1]:
                t3i1_dict = self.model.edit_loss_fn(self.config, t3i1_logits, t3i1_labels)
            else:
                t3i1_dict = self.model.edit_loss_fn(self.config, t3i1_logits,
                                                         t3i1_labels[:, -t3i1_logits.shape[1] - 1:])

            t3i1_before_top_k = torch.topk(torch.nn.functional.softmax(self.check_logits(t3i1_outputs_before), dim=-1),
                                                      k=10,
                                                      dim=-1).indices
            t3i1_after_top_k = torch.topk(torch.nn.functional.softmax(self.check_logits(t3i1_outputs), dim=-1),
                                                      k=10,
                                                      dim=-1).indices




        info_dict["t4i4_kl_loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1)) / \
                               post_base_logits_softmax_top_k.view(-1).shape[0] #high is better t4i4
        info_dict["t3i3_kl_loc/acc"] = sum(
            post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1)) / \
                                     post_image_base_logits_softmax_top_k.view(-1).shape[0]  #high is better t3i3
        info_dict["t1i3_kl_loc/acc"] = sum(
            post_tv_logits_softmax_top_k.view(-1) == base_tv_logits_softmax_top_k.view(-1)) / \
                                  post_tv_logits_softmax_top_k.view(-1).shape[0]  #high is better t1i3

        info_dict["t3i1_kl_loc/acc"] = sum(
            t3i1_after_top_k.view(-1) == t3i1_before_top_k.view(-1)) / \
                                  t3i1_after_top_k.view(-1).shape[0]  #high is better t3i1
        info_dict["t1i2_kl_loc/acc"] = sum(
            t1i2_after_top_k.view(-1) == t1i2_before_top_k.view(-1)) / \
                                  t1i2_after_top_k.view(-1).shape[0]  #high is better t1i2

        info_dict["t2i1_kl_loc/acc"] = sum(
            t2i1_after_top_k.view(-1) == t2i1_before_top_k.view(-1)) / \
                                  t2i1_after_top_k.view(-1).shape[0]  #high is better t2i1

        info_dict["t2i2_kl_loc/acc"] = sum(
            t2i2_after_top_k.view(-1) == t2i2_before_top_k.view(-1)) / \
                                  t2i2_after_top_k.view(-1).shape[0]  #high is better t2i2
        info_dict["t1i4_kl_loc/acc"] = sum(
            t1i4_after_top_k.view(-1) == t1i4_before_top_k.view(-1)) / \
                                  t1i4_after_top_k.view(-1).shape[0]  #high is better t1i4
        info_dict["t2i4_kl_loc/acc"] = sum(
            t2i4_after_top_k.view(-1) == t2i4_before_top_k.view(-1)) / \
                                  t2i4_after_top_k.view(-1).shape[0]  #high is better t2i4



        info_dict["t1i3_acc_loc/acc"] = 1- TV_post_edit_dict["acc"].item()  #low is better t1i3
        info_dict['t2i2_acc_loc/acc'] = 1-close_post_edit_dict["acc"].item()  #low is better t2i2
        info_dict['t1i4_acc_loc/acc'] = 1-o_post_edit_dict["acc"].item()  #low is better t1i4
        # info_dict['t3i1_acc_loc/acc'] = t3i1_dict["acc"].item()  #low is better t3i1
        info_dict['t1i2_acc_loc/acc'] = 1-t1i2_dict["acc"].item()  #low is better t1i2
        info_dict['t2i1_acc_loc/acc'] = 1-t2i1_dict["acc"].item()  #low is better t2i1
        info_dict['t2i4_acc_loc/acc'] = 1-t2i4_dict["acc"].item()  #low is better t2i4



        info_dict['edit/acc'] = post_edit_dict["acc"].item()  #high is better
        info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
        info_dict['edit/prob'] = post_edit_dict["prob"].item()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item() #high is better


        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item() #high is better
        info_dict["time/edit"] = edit_time

        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()

        show_s(post_edit_outputs, batch["edit_outer"], tokenizer,
               tilte=f'edit_outer_after_{self.config.loss}{self.config.choose}{self.config.model_name}',
               indexs=indexs,image_tokens=image_tokens)

        show_s(post_image_base_outputs, batch["loc_image"], tokenizer,
               tilte=f't3i3_after_{self.config.loss}{self.config.choose}{self.config.model_name}',indexs=indexs,image_tokens=image_tokens)#t3i3

        show_s(close_post_edit_outputs, batch["close_edit"], tokenizer,
               tilte=f't2i2_after_{self.config.loss}{self.config.choose}{self.config.model_name}',indexs=indexs,image_tokens=image_tokens)#t2i2

        show_s(t1i2_outputs, batch["t1i2"], tokenizer,
               tilte=f't1i2_after_{self.config.loss}{self.config.choose}{self.config.model_name}',indexs=indexs,image_tokens=image_tokens)#t1i2

        show_s(t2i1_outputs, batch["t2i1"], tokenizer,
               tilte=f't2i1_after_{self.config.loss}{self.config.choose}{self.config.model_name}',indexs=indexs,image_tokens=image_tokens)#t2i1

        show_s(tv_loc_image_outputs, batch["tv_loc_image"], tokenizer,
               tilte=f't1i3_after_{self.config.loss}{self.config.choose}{self.config.model_name}',indexs=indexs,image_tokens=image_tokens)#t1i3

        show_s(t3i1_outputs, batch["t3i1"], tokenizer,
               tilte=f't3i1_after_{self.config.loss}{self.config.choose}{self.config.model_name}',indexs=indexs,image_tokens=image_tokens)#t3i1


        # kl_dict = {}
        #loc_image kl
        # base_image_outputs.logits
        # post_image_base_outputs.logits

        # kl_all=kl_value(  base_image_outputs.logits,post_image_base_outputs.logits).item()
        # kl_image=kl_value(  base_image_outputs.logits[0:,:32],post_image_base_outputs.logits[0:,:32]).item()
        # kl_text=kl_value(  base_image_outputs.logits[0:,32:],post_image_base_outputs.logits[0:,32:]).item()
        # kl_dict['I-loc']=[kl_all,kl_image,kl_text]
        #close kl
        # before_outer_outputs.logits
        # close_post_edit_outputs.logits
        #
        # kl_all = kl_value(before_outer_outputs.logits, close_post_edit_outputs.logits).item()
        # kl_image = kl_value(before_outer_outputs.logits[0:, :32], close_post_edit_outputs.logits[0:, :32]).item()
        # kl_text = kl_value(before_outer_outputs.logits[0:, 32:], close_post_edit_outputs.logits[0:, 32:]).item()
        #
        # kl_dict['close-loc'] = [kl_all, kl_image, kl_text]
        #
        # #outer kl
        # # ot_post_edit_outputs.logits
        # # post_edit_outputs.logits
        # kl_all = kl_value(ot_post_edit_outputs.logits, post_edit_outputs.logits).item()
        # kl_image = kl_value(ot_post_edit_outputs.logits[0:, :32], post_edit_outputs.logits[0:, :32]).item()
        # kl_text = kl_value(ot_post_edit_outputs.logits[0:, 32:], post_edit_outputs.logits[0:, 32:]).item()
        #
        # kl_dict['outer-loc'] = [kl_all, kl_image, kl_text]
        #
        # torch.save(kl_dict,
        #            f'./multmodal/VLKEB-main/results/score_blip/{indexs}_{self.config.loss}{self.config.choose}_KL.pkl')

        if not self.config.eval_only:
            info_dict['loss/edit'] = l_edit if type(l_edit) == int else l_edit.item()
            info_dict['loss/image_edit'] = l_image_edit if type(l_image_edit) == int else l_image_edit.item()
            info_dict['loss/loc'] = l_loc if type(l_loc) == int else l_loc.item()
            l_base = torch.tensor(0.0)
            l_total = l_total_edit + self.config.cbase * l_base

            info_dict["loss/total"] = l_total.item()
            info_dict["loss/total_edit"] = l_total_edit.item()
        else:
            info_dict['loss/edit'] = 0#l_edit if type(l_edit) == int else l_edit.item()
            info_dict['loss/image_edit'] = 0#l_image_edit if type(l_image_edit) == int else l_image_edit.item()
            info_dict['loss/loc'] = 0#l_loc if type(l_loc) == int else l_loc.item()
            l_base = torch.tensor(0.0)
            l_total =0# l_total_edit + self.config.cbase * l_base

            info_dict["loss/total"] = 0#l_total.item()
            info_dict["loss/total_edit"] = 0#l_total_edit.item()
        ####################################################################################################

        ################ portability #################
        if batch['port'] is not None:
            port_acc = 0
            for port in batch['port']:
                with torch.no_grad():
                    port_outputs = edited_model(port)
                    port_labels = port["labels"]
                    if not isinstance(port_outputs, torch.Tensor):
                        port_logits = port_outputs.logits
                    else:
                        port_logits = port_outputs
                    if port_logits.shape[1] > port_labels.shape[1]:
                        port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                    else:
                        port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                    port_acc += port_dict["acc"].item()
            port_acc /= len(batch['port'])
            info_dict['port/acc'] = port_acc
        ################ portability #################
        
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"

        t2i4_acc_loc = f"{stats['t2i4_acc_loc/acc_val']:<12.5f}"
        t2i1_acc_loc = f"{stats['t2i1_acc_loc/acc_val']:<12.5f}"
        t1i2_acc_loc = f"{stats['t1i2_acc_loc/acc_val']:<12.5f}"
        t1i4_acc_loc = f"{stats['t1i4_acc_loc/acc_val']:<12.5f}"
        t2i2_acc_loc = f"{stats['t2i2_acc_loc/acc_val']:<12.5f}"
        t1i3_acc_loc = f"{stats['t1i3_acc_loc/acc_val']:<12.5f}"
        t2i4_kl_loc = f"{stats['t2i4_kl_loc/acc_val']:<12.5f}"
        t1i4_kl_loc = f"{stats['t1i4_kl_loc/acc_val']:<12.5f}"
        t2i2_kl_loc = f"{stats['t2i2_kl_loc/acc_val']:<12.5f}"
        t2i1_kl_loc = f"{stats['t2i1_kl_loc/acc_val']:<12.5f}"
        t1i2_kl_loc = f"{stats['t1i2_kl_loc/acc_val']:<12.5f}"
        t3i1_kl_loc = f"{stats['t3i1_kl_loc/acc_val']:<12.5f}"
        t1i3_kl_loc = f"{stats['t1i3_kl_loc/acc_val']:<12.5f}"
        t4i4_kl_loc = f"{stats['t4i4_kl_loc/acc_val']:<12.5f}"



        LOG.info(
          f"Step {prog} outer_acc: {outer_acc} image_acc: {image_acc} inner_acc: {inner_acc} it_time: {elapsed:.4f} "
          f"t2i4_acc_loc: {t2i4_acc_loc}, "
          f"t2i1_acc_loc: {t2i1_acc_loc} "
          f"t1i2_acc_loc: {t1i2_acc_loc }"
          f"t1i4_acc_loc: {t1i4_acc_loc} "
          f"t2i2_acc_loc: {t2i2_acc_loc} "
          f"t1i3_acc_loc: {t1i3_acc_loc}"
          f"t2i4_kl_loc: {t2i4_kl_loc}"
          f"t1i4_kl_loc: {t1i4_kl_loc}"
          f"t2i2_kl_loc: {t2i2_kl_loc}"
          f"t2i1_kl_loc: {t2i1_kl_loc}"
          f"t1i2_kl_loc: {t1i2_kl_loc}"
          f"t3i1_kl_loc: {t3i1_kl_loc}"
          f"t3i1_kl_loc: {t3i1_kl_loc}"
          f"t1i3_kl_loc: {t1i3_kl_loc}"
          f"t4i4_kl_loc: {t4i4_kl_loc}"
        )#
        if 'port/acc_val' in stats:
            LOG.info(f"step {prog} port_acc: {stats['port/acc_val']:<12.5f}")

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in tqdm(enumerate(self.val_loader), total=steps, desc="Validation", ncols=100):
            if val_step >= steps:
                break
            _, _, _, _, info_dict = self.edit_step(batch, training=False,indexs=self.config.dataset+str(val_step))
            averager.add(info_dict)

            if (
                log
                and (val_step + 1) % self.config.log_interval == 0
            ):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats