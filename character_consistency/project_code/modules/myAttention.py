import torch
import torch.nn.functional as F
import math
import re
import inspect
from .utils import _proj_out, _add_mask_safe
from .mask import CrossSelfKVMasker
from diffusers.models.attention import Attention
from typing import Optional

class SanaLinearAttnProcessor2_0_SharedKVAll:
    # 캐릭터 일관성 유지하기 위한 커스텀 어텐션 프로세서
    def __init__(self, my_name: str = "",eps: float = 1e-15, masker: Optional[CrossSelfKVMasker] = None,
                 use_adaptive_residual: bool = False, **kwargs):
        self.my_name = my_name
        self.eps = float(eps)
        self.masker = masker
        self._printed_shapes = False
        self.use_adaptive_residual = use_adaptive_residual
        self.current_dump_path = None 
        self.global_step_counter = 0
        self.curr_superclass = "default"
        self.curr_index = 0
        self.current_prompts = []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _cross_attention_vanilla(self, attn: Attention, hidden_states: torch.Tensor,
                                encoder_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor],
                                original_dtype: torch.dtype) -> torch.Tensor:
        # 마스크 모으기 위한 cross attention
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
        scores = _add_mask_safe(scores, attention_mask, attn, Q, K)
        
        weights = torch.softmax(scores, dim=-1)
        
        # 마스크 만들기 위해 어텐션 가중치 모아놓음
        if self.masker is not None:
            self.masker.collect(weights)
        
        out = torch.matmul(weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Q, H * Dv)
        out = _proj_out(attn, out, original_dtype)
        
        if original_dtype == torch.float16:
            out = out.clip(-65504, 65504)
        return out

    def _self_attention_linear_sharedkv(self, attn: Attention, hidden_states: torch.Tensor,
                                    attention_mask: Optional[torch.Tensor],
                                    original_dtype: torch.dtype) -> torch.Tensor:
        # 캐릭터 일관성 방법이 적용된 커스텀 self attention
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
        phi_k = F.relu(k).float()
        v_full = v.float()

        phi_k_T = phi_k.transpose(-1, -2)
        S_frame = torch.matmul(phi_k_T, v_full)
        z_frame = phi_k_T.sum(dim=-1, keepdim=True)

        hard_mask = self.masker.get_mask_for_seq_len(Tq, mask_type='hard', device=phi_k.device)
        

        S_shared = S_frame.sum(dim=0, keepdim=True)
        z_shared = z_frame.sum(dim=0, keepdim=True)
        # -----------------------------------------------------------------------


        if self.global_step_counter != 0: # 두번째 디퓨전 스텝에서 사용하는 방법
            obj_hard_exp = hard_mask.unsqueeze(1).unsqueeze(-1).to(phi_q.dtype)
            bg_hard_exp  = (~hard_mask).unsqueeze(1).unsqueeze(-1).to(phi_q.dtype)

            # 객체 부분
            phi_q_obj = phi_q * obj_hard_exp
            num_obj   = torch.matmul(phi_q_obj, S_shared)
            den_obj   = torch.matmul(phi_q_obj, z_shared)
            attn_out_obj = num_obj / (den_obj + self.eps)

            # 배경 부분
            phi_q_bg = phi_q * bg_hard_exp
            num_bg   = torch.matmul(phi_q_bg, S_frame)
            den_bg   = torch.matmul(phi_q_bg, z_frame)
            attn_out_bg = num_bg / (den_bg + self.eps)

            # 합성
            attn_out_shared = attn_out_obj * obj_hard_exp + attn_out_bg * bg_hard_exp

            attn_out_shared_flat = attn_out_shared.permute(0, 2, 1, 3).contiguous().view(B, Tq, -1)
            out = _proj_out(attn, attn_out_shared_flat, original_dtype)

        else: # 첫번째 디퓨전 스텝에서 사용하는 방법
            q_mask = hard_mask.unsqueeze(1).unsqueeze(-1).to(phi_q.dtype)

            # 쿼리에 마스크씌우고 linear attention
            phi_q_masked = phi_q * q_mask
            num_masked = torch.matmul(phi_q_masked, S_frame)
            den_masked = torch.matmul(phi_q_masked, z_frame)
            o_mask_standard_linear = num_masked / (den_masked + self.eps)

            o_hat_list = []
            Dv = o_mask_standard_linear.shape[-1]
            scale = 1.0 / math.sqrt(Dv)

            for b in range(B):
                curr_o = o_mask_standard_linear[b]  # [H, Tq, Dv]
                others_indices = [i for i in range(B) if i != b]

                if len(others_indices) > 0:
                    # [B-1, H, Tq, Dv] -> [H, B-1, Tq, Dv] -> [H, (B-1)*Tq, Dv]
                    others_kv = o_mask_standard_linear[others_indices]
                    kv_source = others_kv.permute(1, 0, 2, 3).reshape(H, -1, Dv)
                    # [H, Tq, Dv] @ [H, Dv, (B-1)*Tq] -> [H, Tq, (B-1)*Tq]
                    sim = torch.matmul(curr_o, kv_source.transpose(-1, -2)) * scale
                    attn_weights = torch.softmax(sim, dim=-1)
                    o_hat_b = torch.matmul(attn_weights, kv_source) # [H, Tq, Dv]
                else:
                    exit(0)

                o_hat_list.append(o_hat_b)

            o_hat_unproj = torch.stack(o_hat_list, dim=0)  # [B, H, Tq, Dv]

            
            obj_hard_exp = hard_mask.unsqueeze(1).unsqueeze(-1).to(phi_q.dtype)
            bg_hard_exp  = (~hard_mask).unsqueeze(1).unsqueeze(-1).to(phi_q.dtype)

            # 객체 부분
            phi_q_obj = phi_q * obj_hard_exp
            num_obj   = torch.matmul(phi_q_obj, S_shared)
            den_obj   = torch.matmul(phi_q_obj, z_shared)
            attn_out_obj = num_obj / (den_obj + self.eps)

            # 배경 부분
            phi_q_bg = phi_q * bg_hard_exp
            num_bg   = torch.matmul(phi_q_bg, S_frame)
            den_bg   = torch.matmul(phi_q_bg, z_frame)
            attn_out_bg = num_bg / (den_bg + self.eps)

            # 합성
            attn_out_shared = attn_out_obj * obj_hard_exp + attn_out_bg * bg_hard_exp

            # 두 방법 섞기
            a = 0.1
            blend_mask = hard_mask.unsqueeze(1).unsqueeze(-1).to(attn_out_shared.dtype)
            blended_unproj = attn_out_shared + blend_mask * a * (o_hat_unproj - attn_out_shared)
            blended_flat = blended_unproj.permute(0, 2, 1, 3).contiguous().view(B, Tq, -1)
            out = _proj_out(attn, blended_flat, original_dtype)

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
    # 각 layer에 커스텀 어텐션 프로세서 설치
    pipeline,
    replace_self: bool = True,
    replace_cross: bool = True,
    only_block_indices_self=None,
    only_block_indices_cross=None,
    debug: bool = False,
    masker: Optional[CrossSelfKVMasker] = None,
    **processor_kwargs,
) -> tuple[int, int]:
    # 파이프라인의 트랜스포머에서 현재 설정된 어텐션 프로세서들을 가져옴
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
        # 패턴에 맞지 않는 이름은 기존 프로세서 유지
        if not m:
            new_map[name] = proc
            continue
        
        blk = int(m.group(1)) # 블록 인덱스
        kind = m.group(2) # 어텐션 종류
        
        # self attention 교체 조건 확인
        if kind == "attn1" and (allow_self is not None) and (blk not in allow_self):
            if debug: print(f"[keep] {name} (self: block {blk} not in allow_self)")
            new_map[name] = proc
            continue
        
        # cross attention 교체 조건 확인
        if kind == "attn2" and (allow_cross is not None) and (blk not in allow_cross):
            if debug: print(f"[keep] {name} (cross: block {blk} not in allow_cross)")
            new_map[name] = proc
            continue
        
        # 커스텀 어텐션으로 교체
        do_replace = (kind == "attn1" and replace_self) or (kind == "attn2" and replace_cross)
        if do_replace:
            new_map[name] = SanaLinearAttnProcessor2_0_SharedKVAll(
                my_name=name,
                masker=masker,
                **processor_kwargs
            )
            if kind == "attn1": rep_self += 1
            else: rep_cross += 1
        else:
            if debug: print(f"[keep] {name} (keep default {kind})")
            new_map[name] = proc
    
    pipeline.transformer.set_attn_processor(new_map)
    
    return rep_self, rep_cross

def enable_precollect_cross_before_self(pipeline):
    # 마스크는 cross attention을 활용해서 만들어지고 그 마스크는 self attention에서 사용됨
    # but 어텐션 순서는 self → cross → self → cross ... 이므로 첫 self attention이 마스크 없이 실행됨
    # 이 함수를 호출하면 cross attention이 먼저 실행되도록 강제해서 첫 self attention도 마스크를 활용할 수 있게 됨
    masker = getattr(pipeline, "_masker", None)
    if masker is None:
        return
    
    for i, blk in enumerate(getattr(pipeline.transformer, "transformer_blocks", [])):
        attn2 = getattr(blk, "attn2", None)
        norm2 = getattr(blk, "norm2", None)
        
        if attn2 is None or norm2 is None:
            continue
        
        orig_forward = blk.forward
        forward_signature = inspect.signature(orig_forward)
        

        def wrapped_forward(*args, _i=i, _orig=orig_forward, _attn2=attn2, _norm2=norm2, _masker=masker, _sig=forward_signature, **kwargs):
            try:
                bound_args = _sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                hidden_states = bound_args.arguments.get('hidden_states')
                encoder_hidden_states = bound_args.arguments.get('encoder_hidden_states')
                attention_mask = bound_args.arguments.get('attention_mask')
                
            except TypeError:
                return _orig(*args, **kwargs)

            # 마스크 생성
            if hidden_states is not None:
                Q = int(hidden_states.shape[1])
                
                # 텍스트 정보(encoder_hidden_states)가 있고
                # 아직 이거에 대한 마스크가 생성되지 않았다면 실행
                if encoder_hidden_states is not None:
                    if _masker.get_mask_for_seq_len(Q, device=hidden_states.device) is None:
                        # 실행
                        with torch.no_grad():
                            h2 = _norm2(hidden_states)
                            _ = _attn2(h2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)
                        _masker.aggregate()
                        _masker.step_store.clear()
                else:
                    # 텍스트가 안 들어온 경우
                    if _i == 0:
                        print(f"[DBG] Block {_i}: encoder_hidden_states is NONE! (Unconditional pass or Architecture design)")

            return _orig(*args, **kwargs)
        
        blk.forward = wrapped_forward
