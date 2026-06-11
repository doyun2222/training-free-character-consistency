import os
import yaml
import torch
import argparse
import time
from tqdm import tqdm
from diffusers import SanaSprintPipeline
from pathlib import Path
from modules.mask import CrossSelfKVMasker
from modules.utils import create_token_indices_span, _parse_guidance_schedule, _scale_prompt_embeds_for_alpha, sanitize_filename, single_benchmark
from modules.myAttention import install_shared_linear_attn_processors, enable_precollect_cross_before_self


def main(args):
    # 벤치마크 로드
    with open(args.single_benchmark_dir, 'r') as f:
        dataset = yaml.safe_load(f)
    s_bench = single_benchmark(dataset)
    
    # 파이프라인 로드
    pipeline = SanaSprintPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    
    # 마스커 초기화
    masker = CrossSelfKVMasker(mask_dropout=args.mask_dropout, max_history=20, temperature=args.temperature)
    pipeline._masker = masker
    
    # 프로세서 설치
    rep_self, rep_cross = install_shared_linear_attn_processors(
        pipeline,
        replace_self=True,
        replace_cross=True,
        only_block_indices_self=[7,8,9,10,11,12,13,14,15,16,17,18,19],
        only_block_indices_cross=None,
        debug=True,
        masker=masker,
        eps=1e-15,
    )
    
    # self attention전에 cross attentino을 미리 실행해서 마스크 만드는 기능 사용
    enable_precollect_cross_before_self(pipeline)
    
    # 메모리 부족 방지
    pipeline.vae.enable_tiling()
    
    # 가이던스 스케줄 파싱
    cfg_sched = _parse_guidance_schedule(args.guidance_schedule, args.num_inference_steps, args.guidance_scale)
    g0 = float(cfg_sched[0])
    print(f"[CFG] schedule={cfg_sched}, base(g0)={g0}")
    
    # (스텝마다 다른 가이던스 스케일 사용할때 유효)다음 스텝용 prompt_embeds를 alpha = g_next / g0 로 스케일링
    def on_step_end(pipeline, i, t, callback_kwargs):
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
        
        m = getattr(pipeline, "_masker", None)
        if m is not None:
            ok = m.aggregate()
            m.step_store.clear()

        if hasattr(pipeline.transformer, "attn_processors"):
            for proc in pipeline.transformer.attn_processors.values():
                if hasattr(proc, 'global_step_counter'):
                    proc.global_step_counter += 1
        
        return callback_kwargs
    
    # 이미지 생성 루프
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    total_images = 0
    total_generation_time = 0.0
    total = sum(len(v) for v in s_bench.values())
    
    with tqdm(total=total, desc="Image Generating") as pbar:
        for superclass, prompt_data in s_bench.items():
            for id_prompt, frame_prompt_list, index, subject_str in prompt_data:
                prompts = [f"{id_prompt} {frame}" for frame in frame_prompt_list]
                batch_size = len(prompts)
                
                # 현재 배치 프롬프트 내에서 캐릭터를 나타내는 토큰이 몇 번째 토큰인지 위치 파악(마스크 위해)
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
                    if hasattr(proc, 'curr_superclass'):
                        proc.curr_superclass = str(superclass)
                        proc.curr_index = int(index)
                        proc.current_prompts = prompts
                        # 저장 경로 재확인 (args에서 받은 경로)
                        proc._attn_save_dir = Path(args.output_dir)

                
                start_time = time.time()
                
                # 이미지 생성
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
                       help='Base guidance scale')
    parser.add_argument('--guidance_schedule', type=str, default=None,
                       help='ex: "8,5" for 2 steps, or "7,8,5" for 3 steps.')
    parser.add_argument('--num_inference_steps', type=int, default=2)
    parser.add_argument('--mask_dropout', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.0005,
                       help='temperature for attention mask generation in aggregate()')
    
    parser.add_argument('--pretrained_model', type=str,
                       default='Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers')
    parser.add_argument('--output_dir', type=str, default='./results_sprint_c')
    parser.add_argument('--single_benchmark_dir', type=str,
                       default='/home/ngo/consistory+.yaml')
    
    args = parser.parse_args()
    main(args)