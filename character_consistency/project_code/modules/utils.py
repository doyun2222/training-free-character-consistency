import re
import torch
import numpy as np
from skimage import filters
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional
from diffusers.models.attention import Attention

def single_benchmark(dataset):
    # 벤치마크 데이터 사용하기 위한 함수
    benchmark = {}
    for key, values in dataset.items():
        all_prompts = []
        for index, v in enumerate(values):
            style = v['style']
            subject = v['subject']
            settings = v['settings']
            id_prompt = f"{style} {subject}"
            all_prompts.append((id_prompt, settings, index, subject))
        benchmark[key] = all_prompts
    return benchmark

def _find_subseq(token_lists, token_seq):
    # 전체 토큰 리스트에서 특정 단어의 토큰 리스트가 시작되는 인덱스 반환
    if not token_seq or len(token_seq) > len(token_lists):
        return -1
    first = token_seq[0]
    for i, t in enumerate(token_lists[:len(token_lists)-len(token_seq)+1]):
        if t == first and token_lists[i:i+len(token_seq)] == token_seq:
            return i
    return -1

def create_token_indices_span(prompts, batch_size, concept_token, tokenizer):
    # 마스크 만들기 위해 프롬프트 내에서 뽑아내고자 하는 토큰이 몇 번째 토큰에 있는지와 그 길이 계산

    if isinstance(concept_token, str):
        concept_token = [concept_token]
    
    concept_ids_list = []
    for x in concept_token:
        cand = []
        ids1 = tokenizer.encode(x, add_special_tokens=False)
        ids2 = tokenizer.encode(" " + x, add_special_tokens=False)
        if len(ids2) > 0: cand.append(ids2)
        if len(ids1) > 0: cand.append(ids1)
        concept_ids_list.append(cand)
    
    batch = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')
    tokens = batch["input_ids"]
    
    token_indices = torch.full((len(concept_token), batch_size), -1, dtype=torch.int64)
    span_lens = torch.zeros((len(concept_token), batch_size), dtype=torch.int64)
    
    for i, cand_seqs in enumerate(concept_ids_list):
        for b in range(batch_size):
            hs = tokens[b].tolist()
            found = False
            for needle in cand_seqs:
                j = _find_subseq(hs, needle)
                if j != -1:
                    token_indices[i, b] = j
                    span_lens[i, b] = len(needle)
                    found = True
                    break
            if not found:
                token_indices[i, b] = -1
                span_lens[i, b] = 0
    
    return token_indices, span_lens

# =============== Guidance Schedule ===============

def _parse_guidance_schedule(s: Optional[str], steps: int, fallback: float):
    # 디퓨전 스텝마다 다른 guidance_scale 사용하기 위함 => 예시: "8,5" → [8.0, 5.0] // 디퓨전 스텝과 개수 같아야함
    if s is None:
        return [float(fallback)] * steps
    
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(vals) == 1:
        vals = vals * steps
    return vals

def _scale_prompt_embeds_for_alpha(prompt_embeds: torch.Tensor, alpha: float) -> torch.Tensor:
    # cfg 강도를 alpha 값에 따라 조절
    if prompt_embeds is None:
        return None
    
    B2 = prompt_embeds.shape[0]
    if B2 % 2 != 0:
        return prompt_embeds
    
    uncond, cond = prompt_embeds.chunk(2, dim=0)
    cond_new = uncond + alpha * (cond - uncond)
    
    return torch.cat([uncond, cond_new], dim=0)

def sanitize_filename(name):
    # 파일명 안전화
    return re.sub(r'[\\/*?:"<>|]', "-", name)

def _proj_out(attn: Attention, x: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
    # 어텐션에서 마지막 프로젝션
    wdtype = attn.to_out[0].weight.dtype
    if x.dtype != wdtype:
        x = x.to(wdtype)
    x = attn.to_out[0](x)
    x = attn.to_out[1](x)
    if x.dtype != original_dtype:
        x = x.to(original_dtype)
    if original_dtype == torch.float16:
        x = x.clip(-65504, 65504)
    return x

def _add_mask_safe(scores: torch.Tensor, attention_mask: Optional[torch.Tensor],
                    attn: Attention, Q: int, K: int) -> torch.Tensor:
    # 어텐션 스코어에 마스크를 더할 때 차원이 맞지 않으면 자동으로 맞춰서 계산
    if attention_mask is None:
        return scores
    
    mask = attention_mask
    if mask.dtype != scores.dtype:
        mask = mask.to(scores.dtype)
    
    while mask.dim() < 4:
        mask = mask.unsqueeze(1)
    
    B, H = scores.shape[0], scores.shape[1]
    if mask.shape[0] == B * H:
        mask = mask.view(B, H, *mask.shape[1:])
    
    if mask.shape[1] not in (1, H):
        mask = mask.mean(dim=1, keepdim=True)
    
    if mask.shape[-2] not in (1, Q):
        mask = mask.mean(dim=-2, keepdim=True)
    
    if mask.shape[-1] != K:
        if mask.shape[-1] > K:
            mask = mask[..., :K]
        else:
            pad = K - mask.shape[-1]
            mask = F.pad(mask, (0, pad))
    
    return scores + mask

