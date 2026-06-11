import torch
import numpy as np
from skimage import filters
from collections import defaultdict
from typing import Optional

# Otsu 알고리즘이 마스크를 하나도 못 찾았을 때 전체의 몇 %를 강제로 활성화할지 결정
TOPK_FALLBACK_RATIO = 0.5

class CrossSelfKVMasker:
    # cross attention 가중치를 수집하고 특정 단어(concept_token)에 해당하는 
    # 어텐션 영역을 계산하여 이진(hard) 마스크 및 부드러운(soft) 마스크를 생성
    def __init__(self, mask_dropout: float = 0.0, max_history: int = 20, temperature: float=0.0005):
        self.mask_dropout = float(mask_dropout)
        self.max_history = int(max_history)
        self.step_store = []
        self.token_indices = None
        self.span_lens = None
        self.last_masks = {}
        self._collect_calls = 0
        self.shape_hist = defaultdict(int)
        self._prev_agg_collect_calls = 0
        self.temperature = float(temperature)

    def reset(self):
        # 배치마다 초기화
        self.step_store.clear()
        self.last_masks.clear()
        self._collect_calls = 0
        self.shape_hist.clear()
        self._prev_agg_collect_calls = 0


    def set_token_indices(self, token_indices: torch.Tensor, span_lens: Optional[torch.Tensor] = None):
        # 마스킹에 사용할 토큰 인덱스와 길이 설정
        self.token_indices = token_indices
        self.span_lens = span_lens


    @torch.no_grad()
    def collect(self, weights_bhqk: torch.Tensor):
        # 디퓨전 과정에서 각 레이어에서 나오는 어텐션 가중치 모음
        if weights_bhqk.dim() == 4:
            attn_bqk = weights_bhqk.mean(dim=1)
        elif weights_bhqk.dim() == 3:
            attn_bqk = weights_bhqk
        else:
            return
        # 현재 스텝의 어텐션 맵을 저장, max_history를 넘으면 오래된 것부터 삭제
        self.step_store.append(attn_bqk.detach())
        self._collect_calls += 1

        if len(self.step_store) > self.max_history:
            self.step_store = self.step_store[-self.max_history:]

    @torch.no_grad()
    def aggregate(self) -> bool:
        # 모은 어텐션 맵들 사용해서 마스크 생성
        if not self.step_store or self.token_indices is None:
            return False
        
        # 어텐션 맵 평균 계산
        attn_maps = torch.stack(self.step_store, dim=0).mean(dim=0)
        B, Q, K = attn_maps.shape
        device = attn_maps.device
        
        hard_masks = []
        soft_masks = []
        num_concepts = self.token_indices.shape[0]
        
        for b in range(B):
            hard_masks_stack = []
            soft_masks_stack = []
            valid_concepts = 0
            
            for c in range(num_concepts):
                # 대상 단어의 토큰 위치, 길이를 가져옴
                idx = int(self.token_indices[c, b].item())
                span = int(self.span_lens[c, b].item()) if (self.span_lens is not None) else 1
                
                if 0 <= idx < K:
                    # 단어가 차지하는 모든 토큰(idx ~ idx+span)의 어텐션 값을 평균냄
                    # 여기서 단어는 텍스트에서 캐릭터를 나타내는 부분임
                    valid_concepts += 1
                    end = min(idx + max(1, span), K)
                    span_matrix = attn_maps[b, :, idx:end]
                    vec = span_matrix.mean(dim=-1)
                else:
                    continue
                
                # hard mask 만들기 위해 otsu 알고리즘 사용
                vec_f = vec.detach().float()
                vec_np = vec_f.cpu().numpy()
                try:
                    threshold_val = filters.threshold_otsu(vec_np)
                except Exception:
                    threshold_val = np.median(vec_np)
                
                binv_np = (vec_np >= threshold_val).astype(np.uint8)
                
                # 마스크가 전부 0이면 상위 n퍼센트 값을 마스크로 사용
                if binv_np.sum() == 0:
                    scores = vec.detach().float().cpu().numpy()
                    k = max(1, int(TOPK_FALLBACK_RATIO * len(scores)))
                    top_idx = np.argpartition(scores, -k)[-k:]
                    binv_np[top_idx] = 1
                
                # hard mask
                binv = torch.from_numpy(binv_np).to(device).bool().view(-1)
                hard_masks_stack.append(binv)

                # soft mask
                soft_m = torch.sigmoid((vec_f.to(device)) / self.temperature)
                soft_masks_stack.append(soft_m.view(-1))
            

            h_mask = torch.stack(hard_masks_stack, dim=0).max(dim=0).values
            s_mask = torch.stack(soft_masks_stack, dim=0).max(dim=0).values

            # 마스크 드롭아웃 적용
            if self.mask_dropout > 0.0:
                drop = (torch.rand_like(s_mask.float(), device=device) < self.mask_dropout)
                h_mask[drop] = False
                s_mask[drop] = 0.0
            
            soft_masks.append(s_mask)
            hard_masks.append(h_mask)
        
        self.last_masks[Q] = {
            'hard': torch.stack(hard_masks, dim=0),
            'soft': torch.stack(soft_masks, dim=0)
        }
        return True

    def get_mask_for_seq_len(self, seq_len: int, mask_type: str = 'soft', device=None):
        # 필요한 마스크 전달
        data = self.last_masks.get(seq_len, None)
        if data is None:
            return None
        
        # data가 딕셔너리인 경우 (hard, soft가 분리되어 저장된 경우)
        if isinstance(data, dict):
            m = data.get(mask_type)
        else:
            # 단일 텐서로 저장된 경우 (이 경우는 과거 버전에서 저장된 마스크일 가능성 때문)
            m = data

        if m is None:
            return None
        return m.to(device) if device is not None else m