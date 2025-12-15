import math
import torch
import comfy.ldm.common_dit
import comfy.model_management as mm

from torch import Tensor
from einops import repeat
from typing import Optional
from unittest.mock import patch

from comfy.ldm.flux.layers import timestep_embedding, apply_mod
from comfy.ldm.lightricks.model import precompute_freqs_cis
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from comfy.ldm.wan.model import sinusoidal_embedding_1d


# SUPPORTED_MODELS_COEFFICIENTS = {
#     "flux": [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
#     "wan2.1_t2v_1.3B": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
#     "wan2.1_t2v_14B": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
#     "wan2.1_i2v_480p_14B": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
#     "wan2.1_i2v_720p_14B": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
# }

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result



def get_Lemica_path(model_type, lemica_step=17, num_inference_steps=50):
    """
    Generate a boolean list based on model_type and lemica_step.
    
    Checks included:
        1. Whether lemica_step exists in calc_dict.
        2. Whether the length of calc_list equals lemica_step.
    """

    # Select dictionary based on model type
    
    if "wan2.1" in model_type:   
        if "t2v" in model_type:
            if '1.3B' in model_type:
                calc_dict = {
                    25: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 21, 24, 29, 35, 40, 44, 47, 48, 49],
                    20: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 20, 26, 33, 40, 45, 48, 49],
                    17: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 18, 25, 32, 39, 46, 49],
                    14: [0, 1, 2, 3, 4, 5, 7, 10, 16, 23, 30, 37, 44, 49],
                }
            else:
                calc_dict = {
                    25: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 18, 21, 25, 29, 34, 38, 42, 45, 47, 48, 49],
                    20: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 18, 23, 29, 36, 42, 46, 48, 49],
                    17: [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 16, 22, 29, 36, 43, 47, 49],
                    14: [0, 1, 2, 3, 4, 5, 7, 10, 17, 24, 31, 38, 45, 49],
                }
        else:
            calc_dict = {
                25: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 21, 24, 27, 30, 32, 34, 36, 37, 38, 39],
                20: [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25, 29, 32, 35, 37, 38, 39],
                17: [0, 1, 2, 3, 4, 5, 7, 9, 12, 15, 19, 24, 29, 33, 36, 38, 39],
                14: [0, 1, 2, 3, 4, 6, 8, 11, 15, 22, 29, 34, 37, 39],
            }
    elif "z_image" in model_type:
        calc_dict = {
            8: [0, 1, 2, 3, 5, 7, 8, 9],
            7: [0, 1, 2, 4, 6, 8, 9],
            6: [0, 1, 2, 5, 8, 9],
        }
    elif "qwen-image" in model_type:
        calc_dict = {
            25: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26, 33, 36, 42, 49],
            17: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 22, 29, 36, 42, 49],
            10: [0, 1, 3, 7, 14, 21, 28, 35, 42, 49]
        }
    else:
        raise ValueError(f"Unknown type {model_type}")
      
    #  Check if lemica_step exists in calc_dict
    if lemica_step not in calc_dict:
        raise ValueError(f"cache_type {lemica_step} not in calc_dict")

    calc_list = calc_dict[lemica_step]

    # Check if calc_list length matches lemica_step
    if len(calc_list) != lemica_step:
        raise ValueError(
            f"Length mismatch: len(calc_list)={len(calc_list)} != lemica_step={lemica_step}"
        )

    # Generate bool_list based on calc_list indices
    bool_list = [i in calc_list for i in range(num_inference_steps)]
    # Duplicate each value (for 2x inference step expansion)
    # bool_list = [b for b in bool_list for _ in range(2)]

    return bool_list



def lemica_wanmodel_forward(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        patches_replace = transformer_options.get("patches_replace", {})
        # rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        # coefficients = transformer_options.get("coefficients")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        model_type = transformer_options.get("model_type")
        enable_lemica = transformer_options.get("enable_lemica", True)
        cache_device = transformer_options.get("cache_device")
        
        if not hasattr(self, "_lemica_cnt"):
            self._lemica_cnt = 0
        cnt = self._lemica_cnt        
        
        bool_list = transformer_options.get("bool_list")        
        num_steps = transformer_options.get("num_steps")    

        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        blocks_replace = patches_replace.get("dit", {})

        # enable lemica
        modulated_inp = e0.to(cache_device) if "ret_mode" in model_type else e.to(cache_device)
        if not hasattr(self, 'lemica_state'):
            self.lemica_state = {
                0: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None},
                1: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None}
            }

        # def update_cache_state(cache, modulated_inp):
        #     if cache['previous_modulated_input'] is not None:
        #         try:
        #             cache['accumulated_rel_l1_distance'] += poly1d(coefficients, ((modulated_inp-cache['previous_modulated_input']).abs().mean() / cache['previous_modulated_input'].abs().mean()))
        #             if cache['accumulated_rel_l1_distance'] < rel_l1_thresh:
        #                 cache['should_calc'] = False
        #             else:
        #                 cache['should_calc'] = True
        #                 cache['accumulated_rel_l1_distance'] = 0
        #         except:
        #             cache['should_calc'] = True
        #             cache['accumulated_rel_l1_distance'] = 0
        #     cache['previous_modulated_input'] = modulated_inp
            

    
        b = int(len(x) / len(cond_or_uncond))

        for i, k in enumerate(cond_or_uncond):
            #update_cache_state(self.lemica_state[k], modulated_inp[i*b:(i+1)*b])
            self.lemica_state[k]['should_calc'] = bool(bool_list[cnt])
            
            # print('--cnt--', cnt)
            # # print('bool --', bool_list)
            # print('--k--', self.lemica_state[k]['should_calc'])

        
        
        if enable_lemica:
            should_calc = False
            for k in cond_or_uncond:
                should_calc = (should_calc or self.lemica_state[k]['should_calc'])
        else:
            should_calc = True

        if not should_calc:
            for i, k in enumerate(cond_or_uncond):
                x[i*b:(i+1)*b] += self.lemica_state[k]['previous_residual'].to(x.device)
        else:
            ori_x = x.to(cache_device)
            for i, block in enumerate(self.blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap, "transformer_options": transformer_options})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
            for i, k in enumerate(cond_or_uncond):
                self.lemica_state[k]['previous_residual'] = (x.to(cache_device) - ori_x)[i*b:(i+1)*b]

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        
        
        # LeMiCa end
        cnt += 1
        if cnt >= num_steps:
            cnt = 0    
        self._lemica_cnt = cnt
        
        return x

    
def lemica_qwen_image_forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        guidance: torch.Tensor = None,
        transformer_options={},
        **kwargs
    ):
        # rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        # coefficients = transformer_options.get("coefficients")
        enable_lemica = transformer_options.get("enable_lemica", True)
        cache_device = transformer_options.get("cache_device")
        
        
        bool_list = transformer_options.get("bool_list")        
        num_steps = transformer_options.get("num_steps")  
        
        
        if not hasattr(self, "_lemica_cnt"):
            self._pair_cnt=0
            self._lemica_cnt=0  
        
        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        # Align with upstream Qwen-Image API: use process_img + pe_embedder
        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size), ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size)))
        txt_ids = torch.linspace(txt_start, txt_start + encoder_hidden_states.shape[1], steps=encoder_hidden_states.shape[1], device=x.device, dtype=x.dtype).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        # image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)
        image_rotary_emb = self.pe_embedder(ids).to(x.dtype).contiguous()
        
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        # lemica logic - use first transformer block's input as modulated input for change detection
        # This is similar to how FLUX uses the first double_block's modulated input
        if len(self.transformer_blocks) > 0:
            # Get the first block to calculate modulated input
            first_block = self.transformer_blocks[0]
            # Use the processed hidden_states as modulated input (after img_in transformation)
            modulated_inp = hidden_states.to(cache_device)
        else:
            modulated_inp = hidden_states.to(cache_device)
        
        # CFG-aware caching - maintain separate states for positive and negative prompts
        # Detect CFG mode by checking encoder_hidden_states sequence length
        is_positive_prompt = encoder_hidden_states.shape[1] > 50  # Long sequence = positive, short = negative
        cache_key = 'positive' if is_positive_prompt else 'negative'
        

        if not hasattr(self, 'lemica_states'):
            self.lemica_states = {
                'positive': {'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_encoder_residual': None, 'previous_hidden_residual': None},
                'negative': {'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_encoder_residual': None, 'previous_hidden_residual': None}
            }
        
        cache_state = self.lemica_states[cache_key]

        # LeMiCa
        self._pair_cnt += 1
        should_calc = bool_list[self._lemica_cnt]
        if cache_key == 'negative':
            self._lemica_cnt += 1 
            if self._lemica_cnt == num_steps:
                self._lemica_cnt = 0
                self._pair_cnt = 0

        if not enable_lemica:
            should_calc = True
        # print('---',bool_list) 
        # print('***',cache_key, self._lemica_cnt, should_calc,num_steps) 
        
        if not should_calc:
            # Use CFG-aware cached residuals
            if (cache_state['previous_encoder_residual'] is not None and 
                cache_state['previous_hidden_residual'] is not None):
                # Check if cached residuals have compatible shapes
                if (cache_state['previous_encoder_residual'].shape == encoder_hidden_states.shape and 
                    cache_state['previous_hidden_residual'].shape == hidden_states.shape):
                    pass  # Using cached computation
                    encoder_hidden_states += cache_state['previous_encoder_residual'].to(encoder_hidden_states.device)
                    hidden_states += cache_state['previous_hidden_residual'].to(hidden_states.device)
                else:
                    # Shape mismatch, force recalculation
                    pass  # Shape mismatch, forcing recalculation
                    should_calc = True
            else:
                # No cached residuals available, force recalculation
                pass  # No cached residuals available
                should_calc = True
        
        # Process through transformer_blocks if calculation is needed
        if should_calc:
            # Store original states for residual calculation
            ori_encoder_hidden_states = encoder_hidden_states.to(cache_device)
            ori_hidden_states = hidden_states.to(cache_device)
            
            # Process through transformer_blocks (Qwen-Image architecture)
            for block in self.transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
            
            # Store residuals for future use in CFG-aware cache state
            cache_state['previous_encoder_residual'] = (encoder_hidden_states.to(cache_device) - ori_encoder_hidden_states)
            cache_state['previous_hidden_residual'] = (hidden_states.to(cache_device) - ori_hidden_states)
            pass  # Residuals calculated and stored

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Optional debug
        # if transformer_options.get("debug_lemica", False):
        #     try:
        #         print("[lemica][qwen-image-edit] shapes:",
        #               "x=", tuple(x.shape),
        #               "orig_shape=", tuple(orig_shape),
        #               "after_proj=", tuple(hidden_states.shape),
        #               "main_num_embeds=", int(main_num_embeds))
        #     except Exception:
        #         pass

        # Use only main image tokens for reconstruction
        hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]    
    
class Lemica:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the lemica will be applied to."}),
                #"model_type": (["flux", "flux-kontext", "ltxv", "lumina_2", "hunyuan_video", "hidream_i1_full", "hidream_i1_dev", "hidream_i1_fast", "wan2.1_t2v_1.3B", "wan2.1_t2v_14B", "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B", "wan2.1_t2v_1.3B_ret_mode", "wan2.1_t2v_14B_ret_mode", "wan2.1_i2v_480p_14B_ret_mode", "wan2.1_i2v_720p_14B_ret_mode"], {"default": "flux", "tooltip": "Supported diffusion model."}),
                # "rel_l1_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."}),
                
                # --------------------------lemica-------------------------
                "model_type": (["qwen-image","wan2.1_t2v_1.3B", "wan2.1_t2v_14B", "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B"], {"default": "wan2.1_t2v_1.3B", "tooltip": "Supported diffusion model."}),                
                "lemica_budget":([25, 20, 17, 14, 10], {"default": 25, "tooltip": "LeMiCa compute budget (NFE)."}),
                # --------------------------lemica end -------------------------
                
                # "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of the steps that will apply lemica."}),
                # "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of the steps that will apply lemica."}),
                "cache_device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "Device where the cache will reside."}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_lemica"
    CATEGORY = "Lemica"
    TITLE = "Lemica"
    
    def apply_lemica(self, model, model_type: str, lemica_budget: int, cache_device: str):

        
        print('***test***', model)
        
        
        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
        # new_model.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        # new_model.model_options["transformer_options"]["coefficients"] = SUPPORTED_MODELS_COEFFICIENTS[model_type]

        # ------------------------LeMiCa---------------------------        
        
        if "wan2.1" in model_type:
            if "t2v" in model_type:
                num_steps = 50
            else:
                num_steps = 40
        elif "qwen-image" in model_type:
            num_steps = 50
        
        # new_model.model_options["transformer_options"]["num_steps"] = num_steps*2
        new_model.model_options["transformer_options"]["num_steps"] = num_steps
        
        print('**lemica_budget**',lemica_budget)
        bool_list = get_Lemica_path(model_type, lemica_step=lemica_budget, num_inference_steps=num_steps)
        new_model.model_options["transformer_options"]["bool_list"] = bool_list
        # ------------------------LeMiCa end---------------------------
        
        new_model.model_options["transformer_options"]["model_type"] = model_type
        new_model.model_options["transformer_options"]["cache_device"] = mm.get_torch_device() if cache_device == "cuda" else torch.device("cpu")
        
        diffusion_model = new_model.get_model_object("diffusion_model")
        
        if "wan2.1" in model_type:
            is_cfg = True
            
            context = patch.multiple(
                diffusion_model,
                forward_orig=lemica_wanmodel_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
            
        elif "qwen-image" in model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward=lemica_qwen_image_forward.__get__(diffusion_model, diffusion_model.__class__)
            )  
        else:
            raise ValueError(f"Unknown type {model_type}")
        
        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            
            c = kwargs["c"]
            # referenced from https://github.com/kijai/ComfyUI-KJNodes/blob/d126b62cebee81ea14ec06ea7cd7526999cb0554/nodes/model_optimization_nodes.py#L868
            sigmas = c["transformer_options"]["sample_sigmas"]
            steps  = len(sigmas)

            
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                current_step_index = 0
                for i in range(len(sigmas) - 1):
                    # walk from beginning of steps until crossing the timestep
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break
                        
            # ------------------Lemica------------------
            # if current_step_index == 0 and hasattr(diffusion_model, "_lemica_cnt"):
            #     diffusion_model._lemica_cnt = 0     
            if current_step_index == 0:            
                if "wan2.1" in model_type:  
                    assert lemica_budget in [25,20,17,14], f"[LeMiCa] {model_type} requires 25,20,17,14, got {lemica_budget}"
                    if "t2v" in model_type:
                        assert steps == 50, f"[LeMiCa] {model_type} requires steps=50, got {steps}"
                    if "i2v" in model_type:
                        assert steps == 40, f"[LeMiCa] {model_type} requires steps=40, got {steps}"  
                elif "qwen-image" in model_type:
                    assert lemica_budget in [25,17,10], f"[LeMiCa] {model_type} requires 25,17,10, got {lemica_budget}"
                    assert steps == 50, f"[LeMiCa] {model_type} requires steps=50, got {steps}"  
            
            # ------------------Lemica------------------
            if current_step_index == 0:
                if is_cfg:
                    # uncond -> 1, cond -> 0
                    if hasattr(diffusion_model, 'lemica_state') and \
                        diffusion_model.lemica_state[0]['previous_modulated_input'] is not None and \
                        diffusion_model.lemica_state[1]['previous_modulated_input'] is not None:
                            delattr(diffusion_model, 'lemica_state')
                else:
                    if hasattr(diffusion_model, 'lemica_state'):
                        delattr(diffusion_model, 'lemica_state')
                    if hasattr(diffusion_model, 'accumulated_rel_l1_distance'):
                        delattr(diffusion_model, 'accumulated_rel_l1_distance')
            
            c["transformer_options"]["enable_lemica"] = True
            # current_percent = current_step_index / (len(sigmas) - 1)
            # c["transformer_options"]["current_percent"] = current_percent
            # if start_percent <= current_percent <= end_percent:
            #     c["transformer_options"]["enable_lemica"] = True
            # else:
            #     c["transformer_options"]["enable_lemica"] = False
                
            with context:
                return model_function(input, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)
    
def patch_optimized_module():
    try:
        from torch._dynamo.eval_frame import OptimizedModule
    except ImportError:
        return

    if getattr(OptimizedModule, "_patched", False):
        return

    def __getattribute__(self, name):
        if name == "_orig_mod":
            return object.__getattribute__(self, "_modules")[name]
        if name in (
            "__class__",
            "_modules",
            "state_dict",
            "load_state_dict",
            "parameters",
            "named_parameters",
            "buffers",
            "named_buffers",
            "children",
            "named_children",
            "modules",
            "named_modules",
        ):
            return getattr(object.__getattribute__(self, "_orig_mod"), name)
        return object.__getattribute__(self, name)

    def __delattr__(self, name):
        return delattr(self._orig_mod, name)

    @classmethod
    def __instancecheck__(cls, instance):
        return isinstance(instance, OptimizedModule) or issubclass(
            object.__getattribute__(instance, "__class__"), cls
        )

    OptimizedModule.__getattribute__ = __getattribute__
    OptimizedModule.__delattr__ = __delattr__
    OptimizedModule.__instancecheck__ = __instancecheck__
    OptimizedModule._patched = True

def patch_same_meta():
    try:
        from torch._inductor.fx_passes import post_grad
    except ImportError:
        return

    same_meta = getattr(post_grad, "same_meta", None)
    if same_meta is None:
        return

    if getattr(same_meta, "_patched", False):
        return

    def new_same_meta(a, b):
        try:
            return same_meta(a, b)
        except Exception:
            return False

    post_grad.same_meta = new_same_meta
    new_same_meta._patched = True

class CompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the torch.compile will be applied to."}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "backend": (["inductor","cudagraphs", "eager", "aot_eager"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_compile"
    CATEGORY = "lemica"
    TITLE = "Compile Model"
    
    def apply_compile(self, model, mode: str, backend: str, fullgraph: bool, dynamic: bool):
        patch_optimized_module()
        patch_same_meta()
        torch._dynamo.config.suppress_errors = True
        
        new_model = model.clone()
        new_model.add_object_patch(
                                "diffusion_model",
                                torch.compile(
                                    new_model.get_model_object("diffusion_model"),
                                    mode=mode,
                                    backend=backend,
                                    fullgraph=fullgraph,
                                    dynamic=dynamic
                                )
                            )
        
        return (new_model,)
    

NODE_CLASS_MAPPINGS = {
    "Lemica": Lemica,
    "CompileModel": CompileModel
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
