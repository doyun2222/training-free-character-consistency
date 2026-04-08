# 간소화된 버전: mask_dropout + shared attention + guidance_schedule
# (residual, concept_bias, ca_score_scale, token_weight_scale 제거)
# + Frame별 Attention Map 시각화 기능 추가
#내 key에는 mask X
#dreamsim 0.28대

import os
import re
import time
import yaml
import torch
import argparse
from tqdm import tqdm
from typing import Optional
from diffusers import SanaSprintPipeline
import numpy as np
import math
from skimage import filters
from collections import defaultdict
import torch.nn.functional as F
from diffusers.models.attention import Attention


# === [추가됨] 시각화 라이브러리 ===
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# ===============================

# =============== DEBUG SWITCHES ===============

DEBUG = True
PRINT_EVERY = 1
TOPK_FALLBACK_RATIO = 0.5
OTSU_SCALER = 1.0

# ★★★ Attention Map 시각화 옵션 ★★★
SAVE_ATTN_MAPS = False  # True로 변경하면 frame별 attention map 저장
ATTN_MAP_SAVE_DIR = "./attn_maps_debug"  # 저장 디렉토리

def _dbg_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# =============== 유틸 ===============

def attn_map_to_binary(attention_map, scaler=1.0):
    attention_map_np = attention_map.detach().float().cpu().numpy()
    try:
        threshold_value = filters.threshold_otsu(attention_map_np) * scaler
    except Exception:
        threshold_value = float(np.median(attention_map_np)) * scaler
    binary_mask = (attention_map_np >= threshold_value).astype(np.uint8)
    return binary_mask

def _find_subseq(haystack, needle):
    if not needle or len(needle) > len(haystack):
        return -1
    first = needle[0]
    for i, t in enumerate(haystack[:len(haystack)-len(needle)+1]):
        if t == first and haystack[i:i+len(needle)] == needle:
            return i
    return -1

def create_token_indices_span(prompts, batch_size, concept_token, tokenizer):
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
    
    if DEBUG:
        print("[DBG] concept strings:", concept_token)
        print("[DBG] token_indices (span start):", token_indices.tolist())
        print("[DBG] span_lens:", span_lens.tolist())
    
    return token_indices, span_lens

# =============== Guidance Schedule ===============

def _parse_guidance_schedule(s: Optional[str], steps: int, fallback: float):
    """
    guidance_schedule 문자열을 파싱해서 리스트로 변환
    예: "8,5" → [8.0, 5.0]
    """
    if s is None:
        return [float(fallback)] * steps
    
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(vals) == 1:
        vals = vals * steps
    if len(vals) != steps:
        raise ValueError(f"--guidance_schedule length ({len(vals)}) must equal num_inference_steps ({steps})")
    return vals

def _scale_prompt_embeds_for_alpha(prompt_embeds: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    prompt_embeds: [2B, ...] (uncond, cond) 순서라고 가정
    cond' = uncond + alpha * (cond - uncond)
    alpha = g_next / g0
    """
    if prompt_embeds is None:
        return None
    
    B2 = prompt_embeds.shape[0]
    if B2 % 2 != 0:
        return prompt_embeds
    
    uncond, cond = prompt_embeds.chunk(2, dim=0)
    cond_new = uncond + alpha * (cond - uncond)
    
    return torch.cat([uncond, cond_new], dim=0)

# =============== 마스커 (간소화) ===============

class CrossSelfKVMasker:
    """
    cross-attn weights 수집 → concept 스팬으로 Q-스코어 생성 → Otsu bin → mask
    """
    def __init__(self, mask_dropout: float = 0.0, max_history: int = 20):
        self.mask_dropout = float(mask_dropout)
        self.max_history = int(max_history)
        self.step_store = []
        self.token_indices = None
        self.span_lens = None
        self.last_masks = {}
        self._collect_calls = 0
        self.shape_hist = defaultdict(int)
        self._prev_agg_collect_calls = 0

    def reset(self):
        self.step_store.clear()
        self.last_masks.clear()
        self._collect_calls = 0
        self.shape_hist.clear()
        self._prev_agg_collect_calls = 0
        if DEBUG:
            print("[DBG] Masker.reset(): cleared histories")

    def set_token_indices(self, token_indices: torch.Tensor, span_lens: Optional[torch.Tensor] = None):
        self.token_indices = token_indices
        self.span_lens = span_lens
        if DEBUG:
            print(f"[DBG] Masker.set_token_indices(): idx shape {tuple(token_indices.shape) if token_indices is not None else None}")

    @torch.no_grad()
    def collect(self, weights_bhqk: torch.Tensor):
        if weights_bhqk.dim() == 4:
            attn_bqk = weights_bhqk.mean(dim=1)
        elif weights_bhqk.dim() == 3:
            attn_bqk = weights_bhqk
        else:
            return
        
        self.step_store.append(attn_bqk.detach())
        self._collect_calls += 1
        
        if len(self.step_store) > self.max_history:
            self.step_store = self.step_store[-self.max_history:]

    @torch.no_grad()
    def aggregate(self) -> bool:
        if not self.step_store or self.token_indices is None:
            if DEBUG:
                print("[DBG] aggregate(): skipped (no step_store or no token_indices)")
            return False
        
        attn_maps = torch.stack(self.step_store, dim=0).mean(dim=0)
        B, Q, K = attn_maps.shape
        device = attn_maps.device
        
        if DEBUG:
            print(f"[DBG] aggregate(): attn_maps shape = (B={B}, Q={Q}, K={K})")
        
        masks = []
        num_concepts = self.token_indices.shape[0]
        
        for b in range(B):
            per_concept_bins = []
            valid_concepts = 0
            
            for c in range(num_concepts):
                idx = int(self.token_indices[c, b].item())
                span = int(self.span_lens[c, b].item()) if (self.span_lens is not None) else 1
                
                if 0 <= idx < K:
                    valid_concepts += 1
                    end = min(idx + max(1, span), K)
                    span_matrix = attn_maps[b, :, idx:end]
                    vec = span_matrix.mean(dim=-1)
                else:
                    continue
                
                # Otsu threshold
                vec_np = vec.detach().float().cpu().numpy()
                try:
                    threshold_val = filters.threshold_otsu(vec_np)
                except Exception:
                    threshold_val = np.median(vec_np)
                
                binv_np = (vec_np >= (threshold_val * OTSU_SCALER)).astype(np.uint8)
                
                # top-k fallback
                if binv_np.sum() == 0:
                    scores = vec.detach().float().cpu().numpy()
                    k = max(1, int(TOPK_FALLBACK_RATIO * len(scores)))
                    top_idx = np.argpartition(scores, -k)[-k:]
                    binv_np[top_idx] = 1
                    if DEBUG:
                        print(f"[DBG] Otsu empty → top-{int(100*TOPK_FALLBACK_RATIO)}% fallback")
                
                binv = torch.from_numpy(binv_np).to(device).bool().view(-1)
                per_concept_bins.append(binv)
            
            if valid_concepts > 0:
                m = torch.stack(per_concept_bins, dim=0).max(dim=0).values
            else:
                vec_all = attn_maps[b].mean(dim=-1)
                binv_np = attn_map_to_binary(vec_all, scaler=OTSU_SCALER)
                if binv_np.sum() == 0:
                    scores = vec_all.detach().float().cpu().numpy()
                    k = max(1, int(TOPK_FALLBACK_RATIO * len(scores)))
                    top_idx = np.argpartition(scores, -k)[-k:]
                    binv_np[top_idx] = 1
                m = torch.from_numpy(binv_np).to(device).bool().view(-1)
            
            # ★★★ mask_dropout 적용 ★★★
            if self.mask_dropout > 0.0:
                drop = (torch.rand_like(m.float()) < self.mask_dropout)
                m[drop] = False
                if DEBUG and b == 0:
                    print(f"[DBG] mask_dropout={self.mask_dropout}: dropped {drop.sum().item()} positions")
            
            masks.append(m)
        
        mask_bn = torch.stack(masks, dim=0)
        self.last_masks[Q] = mask_bn
        
        if DEBUG:
            print(f"[DBG] aggregate(): mask per prompt (sum True) = {mask_bn.sum(dim=1).tolist()}")
        
        return True

    def get_mask_for_seq_len(self, seq_len: int, device=None):
        m = self.last_masks.get(seq_len, None)
        if m is None:
            return None
        return m.to(device) if device is not None else m

# =============== 프로세서 ===============

class SanaLinearAttnProcessor2_0_SharedKVAll:
    def __init__(self, my_name: str = "",eps: float = 1e-15, masker: Optional[CrossSelfKVMasker] = None,
                 save_attn_maps: bool = False, attn_save_dir: str = "./attn_maps_debug",
                 use_adaptive_residual: bool = False, **kwargs):
        self.my_name = my_name
        self.eps = float(eps)
        self.masker = masker
        self._printed_shapes = False
        # ★★★ Attention map 저장 설정 ★★★
        self._save_attn_maps = save_attn_maps
        self._attn_save_dir = attn_save_dir
        self.use_adaptive_residual = use_adaptive_residual
        self.current_dump_path = None 
        self.global_step_counter = 0 # 파일명 중복 방지용 카운터
        # 추가 파라미터 저장
        for key, value in kwargs.items():
            setattr(self, key, value)


    def _proj_out(self, attn: Attention, x: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
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


    def _add_mask_safe(self, scores: torch.Tensor, attention_mask: Optional[torch.Tensor],
                       attn: Attention, Q: int, K: int) -> torch.Tensor:
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


    def _cross_attention_vanilla(self, attn: Attention, hidden_states: torch.Tensor,
                                encoder_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor],
                                original_dtype: torch.dtype) -> torch.Tensor:
        B, Q, _ = hidden_states.shape
        _, K, _ = encoder_hidden_states.shape
        H = attn.heads
        
        Qp = attn.to_q(hidden_states)
        Kp = attn.to_k(encoder_hidden_states)
        Vp = attn.to_v(encoder_hidden_states)
        
        if attn.norm_q is not None:
            Qp = attn.norm_q(Qp)
        if attn.norm_k is not None:
            Kp = attn.norm_k(Kp)
        
        Dh = Qp.shape[-1] // H
        Dv = Vp.shape[-1] // H
        
        q = Qp.view(B, Q, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = Kp.view(B, K, H, Dh).permute(0, 2, 1, 3).contiguous()
        v = Vp.view(B, K, H, Dv).permute(0, 2, 1, 3).contiguous()
        
        scale = getattr(attn, "scale", None) or (1.0 / math.sqrt(Dh))
        q = q * scale
        
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = self._add_mask_safe(scores, attention_mask, attn, Q, K)
        
        weights = torch.softmax(scores, dim=-1)
        
        # ★★★ 수집 ★★★
        if self.masker is not None:
            self.masker.collect(weights)
        
        out = torch.matmul(weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Q, H * Dv)
        out = self._proj_out(attn, out, original_dtype)
        
        if original_dtype == torch.float16:
            out = out.clip(-65504, 65504)
        return out


    def _self_attention_vanilla_softmax(self, attn: Attention, hidden_states: torch.Tensor,
                                      attention_mask: Optional[torch.Tensor],
                                      original_dtype: torch.dtype) -> torch.Tensor:
        B, Tq, _ = hidden_states.shape
        H = attn.heads
        
        Q = attn.to_q(hidden_states)
        K = attn.to_k(hidden_states)
        V = attn.to_v(hidden_states)
        
        if attn.norm_q is not None:
            Q = attn.norm_q(Q)
        if attn.norm_k is not None:
            K = attn.norm_k(K)
        
        Dh = Q.shape[-1] // H
        Dv = V.shape[-1] // H
        
        q = Q.view(B, Tq, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = K.view(B, Tq, H, Dh).permute(0, 2, 1, 3).contiguous()
        v = V.view(B, Tq, H, Dv).permute(0, 2, 1, 3).contiguous()
        
        scale = getattr(attn, "scale", None) or (1.0 / math.sqrt(Dh))
        q = q * scale
        
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = self._add_mask_safe(scores, attention_mask, attn, Q=Tq, K=Tq)
        
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Tq, H * Dv)
        out = self._proj_out(attn, out, original_dtype)
        
        if original_dtype == torch.float16:
            out = out.clip(-65504, 65504)
        return out


    def _self_attention_linear_sharedkv(self, attn: Attention, hidden_states: torch.Tensor,
                                   attention_mask: Optional[torch.Tensor],
                                   original_dtype: torch.dtype) -> torch.Tensor:
        B, Tq, _ = hidden_states.shape
        H = attn.heads
   
        Q = attn.to_q(hidden_states)
        K = attn.to_k(hidden_states)
        V = attn.to_v(hidden_states)
   
        if attn.norm_q is not None:
            Q = attn.norm_q(Q)
        if attn.norm_k is not None:
            K = attn.norm_k(K)
   
        q = Q.view(B, Tq, H, -1).permute(0, 2, 1, 3).contiguous()
        k = K.view(B, Tq, H, -1).permute(0, 2, 1, 3).contiguous()
        v = V.view(B, Tq, H, -1).permute(0, 2, 1, 3).contiguous()
   
        phi_q = F.relu(q).float()
        phi_k_full = F.relu(k).float()
        v_full = v.float()
   
        # --- Mask 생성 ---
        k_mask_src = self.masker.get_mask_for_seq_len(Tq, device=phi_k_full.device) if self.masker is not None else None
        if k_mask_src is None:
            k_mask_src = torch.ones((B, Tq), dtype=torch.bool, device=phi_k_full.device)
   
        k_mask = k_mask_src.unsqueeze(1).unsqueeze(-1).to(phi_q.dtype)
       
        # 1. Key Masking
        phi_k_masked = phi_k_full * k_mask
       
        # 2. Query Masking
        phi_q_masked = phi_q * k_mask
 
        # --- Linear Attention Matrices (S, z) ---
        # A. Unmasked Full
        phi_k_full_T = phi_k_full.transpose(-1, -2)
        S_frame_full = torch.matmul(phi_k_full_T, v_full)       # (B, H, D, D)
        z_frame_full = phi_k_full_T.sum(dim=-1, keepdim=True)   # (B, H, D, 1)

        # B. Masked Version
        phi_k_masked_T = phi_k_masked.transpose(-1, -2)
        S_frame_masked_k = torch.matmul(phi_k_masked_T, v_full)       # (B, H, D, D)
        z_frame_masked_k = phi_k_masked_T.sum(dim=-1, keepdim=True)   # (B, H, D, 1)
   
        # ========================================================================
        # [수정됨] Masked Query 기반 Column-wise Cosine Similarity
        # ========================================================================
        S_frame_T = S_frame_masked_k.transpose(-1, -2) # (C, H, V, K)
        # 1. 분자: 내적(Dot Product) 계산

        proxy_output_sum = torch.einsum('b h n k, c h v k -> b c h k', phi_q_masked, S_frame_T)
        raw_dot_product = proxy_output_sum / Tq
 
        # 1. Temperature 설정 (하이퍼파라미터로 빼는 것을 권장)
        # 값이 낮을수록(예: 0.1) 가장 유사한 컨텍스트 하나에 집중하고(Sharp),
        # 값이 높을수록(예: 2.0) 컨텍스트들을 골고루 섞습니다(Flat).
        temperature = 60

        # 2. 헤드 차원 평균 (Head Aggregation)
        # (B, C, H, V) -> (B, C, 1, V)
        # 기존 로직과 동일하게 헤드 간의 합의된 점수를 사용합니다.
        # 만약 헤드별로 다르게 가중치를 주고 싶다면 이 줄을 지우시면 됩니다.
        scores = raw_dot_product.mean(dim=2, keepdim=True)
        #channel_weights_probs = F.softmax(scores / temperature, dim=1)
        min_val = scores.amin(dim=1, keepdim=True) 
        max_val = scores.amax(dim=1, keepdim=True)
        channel_weights_probs = (scores - min_val) / (max_val - min_val + 1e-6)
        #위에거 softmax 주석처리하고 min-max로 변경
        """
        if getattr(self, 'current_dump_path', None) is not None:
            try:
                # 1. 텐서 전처리
                # Shape: (B, C, 1, V) -> (B, C, V) (이미 헤드 차원은 1이므로 제거)
                probs_squeezed = channel_weights_probs.squeeze(2)
                
                # 2. 데이터 펼치기 (Batch 별로 모든 Context와 Spatial 정보를 일렬로)
                # (B, C, V) -> (B, -1)
                B_probs = probs_squeezed.shape[0]
                flattened_probs = probs_squeezed.view(B_probs, -1).detach().float().cpu().numpy()

                data_list = []
                for i in range(B_probs):
                    values = flattened_probs[i]
                    
                    # 샘플링 (데이터가 너무 많으면 그래프 그릴 때 느려지므로 5000개만 추출)
                    if len(values) > 5000:
                         values = np.random.choice(values, 5000, replace=False)
                    
                    for v in values:
                        # 0~1 사이의 확률값입니다.
                        data_list.append({'Batch_Index': f'Image {i}', 'Probability': v})

                df_probs = pd.DataFrame(data_list)

                # 3. 시각화 (KDE Plot)
                plt.figure(figsize=(10, 5))
                
                # x축 범위는 확률이므로 0.0 ~ 1.0 고정
                sns.kdeplot(data=df_probs, x='Probability', hue='Batch_Index', fill=True, palette="Set2", alpha=0.3, clip=(0.0, 1.0))
                
                plt.title(f'Softmax Probability Distribution (Step {self.global_step_counter})')
                plt.xlabel('Probability (Confidence)')
                plt.ylabel('Density')
                plt.xlim(0, 1) # 확률은 0과 1 사이
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # 4. 저장
                # 파일명 변경: softmax_dist_...
                safe_name = getattr(self, 'my_name', 'layer').replace(".", "_")
                save_name = f"softmax_dist_{safe_name}_step_{self.global_step_counter:03d}.png"
                
                save_full_path = os.path.join(self.current_dump_path, save_name)
                plt.savefig(save_full_path)
                plt.close()

            except Exception as e:
                print(f"[Warning] Failed to plot softmax distribution: {e}")
            self.global_step_counter += 1
        """


        H_dim = raw_dot_product.shape[2]
        channel_weights = channel_weights_probs.expand(-1, -1, H_dim, -1)


        # 5. KV Matrix Aggregation (가중 합)
        S_base = torch.einsum('b c h k, c h k v -> b h k v', channel_weights, S_frame_masked_k)
       
        # z(분모) Aggregation
        z_base = torch.einsum('b c h k, c h k i -> b h k i', channel_weights, z_frame_masked_k)
 
        # Self Correction
        indices = torch.arange(B, device=phi_q.device)
        self_channel_weights = channel_weights[indices, indices, :, :] # (B, H, D)
        self_channel_weights = self_channel_weights.unsqueeze(-1) # (B, H, K, 1)
 
        self_boost = 1.0
 
        S_combined = S_base + self_channel_weights * (self_boost * S_frame_full - S_frame_masked_k)
       
        z_combined = z_base + self_channel_weights * (self_boost * z_frame_full - z_frame_masked_k)
 
        # Final Linear Attention
        num = torch.matmul(phi_q, S_combined)
        den = torch.matmul(phi_q, z_combined)
        attn_out = num / (den + self.eps)
   
        out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, Tq, -1)
        out = self._proj_out(attn, out, original_dtype)
   
        if original_dtype == torch.float16:
            out = out.clip(-65504, 65504)
        return out


    def __call__(self, attn: Attention, hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        
        is_cross = getattr(attn, "is_cross_attention", None)
        if is_cross is None:
            is_cross = (encoder_hidden_states is not None) and (encoder_hidden_states is not hidden_states)
        
        if is_cross:
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            return self._cross_attention_vanilla(
                attn, hidden_states, encoder_hidden_states, attention_mask, original_dtype
            )
        else:
            return self._self_attention_linear_sharedkv(
                attn, hidden_states, attention_mask, original_dtype
            )

# =============== 프로세서 설치 ===============

def install_shared_linear_attn_processors(
    pipeline,
    replace_self: bool = True,
    replace_cross: bool = True,
    only_block_indices_self=None,
    only_block_indices_cross=None,
    debug: bool = False,
    masker: Optional[CrossSelfKVMasker] = None,
    save_attn_maps: bool = False,
    attn_save_dir: str = "./attn_maps_debug",
    **processor_kwargs,
) -> tuple[int, int]:
    ap = getattr(pipeline.transformer, "attn_processors", None)
    if ap is None:
        raise AttributeError("pipeline.transformer.attn_processors가 없습니다.")
    
    pat = re.compile(r"transformer_blocks\.(\d+)\.(attn1|attn2)\.")
    allow_self = None if only_block_indices_self is None else set(only_block_indices_self)
    allow_cross = None if only_block_indices_cross is None else set(only_block_indices_cross)
    
    new_map = {}
    rep_self = 0
    rep_cross = 0
    
    for name, proc in ap.items():
        m = pat.search(name)
        if not m:
            new_map[name] = proc
            continue
        
        blk = int(m.group(1))
        kind = m.group(2)
        
        if kind == "attn1" and (allow_self is not None) and (blk not in allow_self):
            if debug: print(f"[keep] {name} (self: block {blk} not in allow_self)")
            new_map[name] = proc
            continue
        
        if kind == "attn2" and (allow_cross is not None) and (blk not in allow_cross):
            if debug: print(f"[keep] {name} (cross: block {blk} not in allow_cross)")
            new_map[name] = proc
            continue
        
        do_replace = (kind == "attn1" and replace_self) or (kind == "attn2" and replace_cross)
        if do_replace:
            if debug:
                print(f"[replace] {name} -> SanaLinearAttnProcessor2_0_SharedKVAll(masker=..., save_attn_maps={save_attn_maps})")
            new_map[name] = SanaLinearAttnProcessor2_0_SharedKVAll(
                my_name=name,
                masker=masker,
                save_attn_maps=save_attn_maps,
                attn_save_dir=attn_save_dir,
                **processor_kwargs
            )
            if kind == "attn1": rep_self += 1
            else: rep_cross += 1
        else:
            if debug: print(f"[keep] {name} (keep default {kind})")
            new_map[name] = proc
    
    pipeline.transformer.set_attn_processor(new_map)
    
    if debug:
        scope_self = "ALL" if allow_self is None else f"{sorted(allow_self)}"
        scope_cross = "ALL" if allow_cross is None else f"{sorted(allow_cross)}"
        print(f"[INFO] replaced self(attn1): {rep_self} (scope_self={scope_self}), "
              f"cross(attn2): {rep_cross} (scope_cross={scope_cross})")
    
    return rep_self, rep_cross

def enable_precollect_cross_before_self(pipeline):
    masker = getattr(pipeline, "_masker", None)
    if masker is None:
        return
    
    for blk in getattr(pipeline.transformer, "transformer_blocks", []):
        attn2 = getattr(blk, "attn2", None)
        norm2 = getattr(blk, "norm2", None)
        
        if attn2 is None or norm2 is None:
            continue
        
        orig_forward = blk.forward
        
        def wrapped_forward(*f_args, _attn2=attn2, _norm2=norm2, _orig=orig_forward, _masker=masker, **f_kwargs):
            hidden_states = f_kwargs.get("hidden_states", None) or (f_args[0] if f_args else None)
            encoder_hidden_states = f_kwargs.get("encoder_hidden_states", None)
            attention_mask = f_kwargs.get("attention_mask", None)
            
            if hidden_states is not None:
                Q = int(hidden_states.shape[1])
                if encoder_hidden_states is not None and _masker.get_mask_for_seq_len(Q, device=hidden_states.device) is None:
                    if DEBUG:
                        print(f"[DBG][PRECOLLECT] no mask for Q={Q} → run cross-attn(collect_only) once")
                    with torch.no_grad():
                        h2 = _norm2(hidden_states)
                        _ = _attn2(h2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)
                    _masker.aggregate()
                    _masker.step_store.clear()
            
            return _orig(*f_args, **f_kwargs)
        
        blk.forward = wrapped_forward

def single_benchmark(dataset):
    benchmark = {}
    for key, values in dataset.items():
        all_prompts = []
        for index, v in enumerate(values):
            style = v['style']
            subject = v['subject']
            settings = v['settings']
            concept_token = v['concept_token']
            id_prompt = f"{style} {subject}"
            all_prompts.append((id_prompt, settings, index, subject))
        benchmark[key] = all_prompts
    return benchmark

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "-", name)

def main(args):
    # 1) Benchmark 로드
    with open(args.single_benchmark_dir, 'r') as f:
        dataset = yaml.safe_load(f)
    s_bench = single_benchmark(dataset)
    
    # 2) Pipeline 로드
    pipeline = SanaSprintPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    
    # 3) Masker 초기화
    masker = CrossSelfKVMasker(mask_dropout=args.mask_dropout, max_history=20)
    pipeline._masker = masker
    
    # 4) 프로세서 설치 (self: 선형 shared attention, cross: 기본)
    rep_self, rep_cross = install_shared_linear_attn_processors(
        pipeline,
        replace_self=True,
        replace_cross=True,
        only_block_indices_self=[7,8,9,10,11,12,13,14,15,16,17,18,19],
        only_block_indices_cross=None,
        debug=True,
        masker=masker,
        save_attn_maps=args.save_attn_maps,
        attn_save_dir=args.attn_save_dir,
        eps=1e-15,
    )
    
    print(f"[INFO] Replaced self-attn: {rep_self}, cross-attn: {rep_cross}")
    print(f"[INFO] mask_dropout: {args.mask_dropout}")
    print(f"[INFO] save_attn_maps: {args.save_attn_maps}, save_dir: {args.attn_save_dir}")
    
    # 5) Pre-collect 활성화
    enable_precollect_cross_before_self(pipeline)
    
    # 6) VAE tiling
    pipeline.vae.enable_tiling()
    
    # ★★★ Guidance Schedule 파싱 ★★★
    cfg_sched = _parse_guidance_schedule(args.guidance_schedule, args.num_inference_steps, args.guidance_scale)
    g0 = float(cfg_sched[0])
    print(f"[CFG] schedule={cfg_sched}, base(g0)={g0}")
    
    # 7) 콜백: 다음 스텝용 prompt_embeds를 alpha = g_next / g0 로 스케일링
    def on_step_end(pipeline, i, t, callback_kwargs):
        # ★★★ Guidance schedule 적용 ★★★
        nxt = i + 1
        if nxt < len(cfg_sched):
            g_next = float(cfg_sched[nxt])
            alpha = (g_next / g0) if g0 != 0.0 else 0.0
            
            pe = callback_kwargs.get("prompt_embeds", None)
            if pe is not None:
                new_pe = _scale_prompt_embeds_for_alpha(pe, alpha)
                callback_kwargs["prompt_embeds"] = new_pe
                tqdm.write(f"[CFG] step={i} done → set next alpha={alpha:.4f} (g_next={g_next}, base={g0})")
            else:
                tqdm.write(f"[CFG][warn] step={i} no prompt_embeds in callback; cannot set next alpha")
        else:
            tqdm.write(f"[CFG] step={i} done → last step; keep alpha")
        
        # Masker aggregate
        m = getattr(pipeline, "_masker", None)
        if m is not None:
            ok = m.aggregate()
            m.step_store.clear()
            if DEBUG:
                tqdm.write(f"[DBG] on_step_end(step {i}): aggregate() -> {ok}")
        
        return callback_kwargs
    
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    total_images = 0
    total_generation_time = 0.0
    total = sum(len(v) for v in s_bench.values())
    
    with tqdm(total=total, desc="Image Generating") as pbar:
        for superclass, prompt_data in s_bench.items():
            for id_prompt, frame_prompt_list, index, subject_str in prompt_data:
                prompts = [f"{id_prompt} {frame}" for frame in frame_prompt_list]
                batch_size = len(prompts)
                
                # 배치마다 token_indices 갱신
                tokenizer = getattr(pipeline, "tokenizer", None) or getattr(pipeline, "tokenizer_2", None)
                if tokenizer is None:
                    raise AttributeError("Pipeline has no tokenizer attribute")
                
                token_indices, span_lens = create_token_indices_span(
                    prompts, batch_size, concept_token=subject_str, tokenizer=tokenizer
                )
                
                masker.set_token_indices(token_indices.to(args.device), span_lens.to(args.device))
                masker.reset()
                pipeline._last_prompts = list(prompts)
                
                output_subdir = os.path.join(args.output_dir, f"{superclass}_{index}")
                os.makedirs(output_subdir, exist_ok=True)

                for proc in pipeline.transformer.attn_processors.values():
                    if hasattr(proc, 'current_dump_path'):
                        proc.current_dump_path = output_subdir
                        proc.global_step_counter = 0  # 새로운 배치 시작 시 카운터 리셋


                if DEBUG:
                    _dbg_print(f"[DBG] new batch: superclass={superclass} index={index}")
                    for b, pr in enumerate(prompts):
                        _dbg_print(f"[DBG] prompt[{b}]: {pr}")
                
                start_time = time.time()
                
                # ★★★ guidance_scale은 베이스(첫 스텝) 값 ★★★
                result = pipeline(
                    prompts,
                    generator=generator,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=g0,
                    callback_on_step_end=on_step_end,
                    callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
                )
                
                batch_elapsed = time.time() - start_time
                time_per_image = batch_elapsed / batch_size if batch_size > 0 else batch_elapsed
                
                print(f"Batch {superclass}_{index} 평균 이미지 생성 시간: {time_per_image:.4f}초")
                
                total_images += batch_size
                total_generation_time += batch_elapsed
                
                
                
                for j, image in enumerate(result.images):
                    prompt = prompts[j]
                    safe_prompt = sanitize_filename(prompt)
                    file_name = safe_prompt + ".png"
                    image.save(os.path.join(output_subdir, file_name))
                
                del result
                import gc; gc.collect()
                torch.cuda.empty_cache()
                
                pbar.update(1)
    
    overall_avg_time = total_generation_time / total_images if total_images > 0 else 0
    print(f"전체 평균 이미지 생성 시간: {overall_avg_time:.4f}초")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--guidance_scale', type=float, default=5.0,
                       help='Base guidance scale (step 0)')
    parser.add_argument('--guidance_schedule', type=str, default=None,
                       help='Guidance schedule as comma-separated values. '
                            'Example: "8,5" for 2 steps, or "7,8,5" for 3 steps. '
                            'If None, uses --guidance_scale for all steps.')
    parser.add_argument('--num_inference_steps', type=int, default=2)
    parser.add_argument('--mask_dropout', type=float, default=0.0,
                       help='Mask dropout rate: fraction of subject positions to randomly deactivate')
    
    # ★★★ Attention map 시각화 옵션 ★★★
    parser.add_argument('--save_attn_maps', action='store_true',
                       help='Save frame-wise shared attention maps to file')
    parser.add_argument('--attn_save_dir', type=str, default='./attn_maps_debug_1222',
                       help='Directory to save attention maps')
    
    parser.add_argument('--pretrained_model', type=str,
                       default='Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers')
    parser.add_argument('--output_dir', type=str, default='./results_sprint_c')
    parser.add_argument('--single_benchmark_dir', type=str,
                       default='/home/ngo/consistory+.yaml')
    
    args = parser.parse_args()
    main(args)