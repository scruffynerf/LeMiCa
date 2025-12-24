from typing import Any, Dict, Optional, Tuple, Union
from diffusers import DiffusionPipeline,FluxPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import torch, time
import numpy as np
import os, re
import pandas as pd
from torch.utils.flop_counter import FlopCounterMode
import sys
import argparse

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def Lemica_call(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})
   
        
        if self.enable_Lemica:
            if hasattr(self, "bool_list") and len(self.bool_list) > self.cnt:
                should_calc = self.bool_list[self.cnt]
            else:
                should_calc = True

            self.cnt += 1
            self.store.append(should_calc)

            if self.cnt == self.num_steps:
                true_count = self.store.count(True)
                # print(f'*** Total steps: {len(self.store)}, True count: {true_count}')
                self.store = []
                self.cnt = 0        
        
        if self.enable_Lemica:
            if not should_calc:
                hidden_states += self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                for index_block, block in enumerate(self.transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                            block,
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            joint_attention_kwargs,
                        )

                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_block_samples is not None:
                        interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        # For Xlabs ControlNet.
                        if controlnet_blocks_repeat:
                            hidden_states = (
                                hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                            )
                        else:
                            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

                for index_block, block in enumerate(self.single_transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                            block,
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            joint_attention_kwargs,
                        )

                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_single_block_samples is not None:
                        interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]
                self.previous_residual = hidden_states - ori_hidden_states
        else:
            
            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                    )

                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # controlnet residual
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # For Xlabs ControlNet.
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                    )

                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]
            

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", nargs="?", const="medium", default=None, type=str)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    num_inference_steps = 50

    speed = {"slow": 26, "medium": 20, "fast": 15, "ultra": 10}
    calc_dict = {
        26: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30, 35, 39, 42, 44, 45, 46, 47, 48, 49],
        20: [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 15, 19, 24, 31, 38, 43, 46, 47, 48, 49],
        15: [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 27, 34, 41, 47, 49],
        10: [0, 1, 3, 7, 14, 21, 28, 35, 42, 49],
    }

    # ===== Parse --cache =====
    lemica_step = None
    key = None
    if args.cache is not None:
        key = str(args.cache).strip().lower()
        if key in speed:
            lemica_step = speed[key]
        else:
            try:
                lemica_step = int(key)
            except ValueError:
                raise ValueError(
                    f'Invalid --cache "{args.cache}". Use slow/medium/fast/superfast or an integer like 26.'
                )

    # ===== Load model =====
    model_path = "black-forest-labs/FLUX.1-dev"
    pipeline = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")

    # ===== Cache control + tag =====
    if lemica_step is None:
        pipeline.transformer.__class__.enable_Lemica = False
        tag = "nocache"
    else:
        if lemica_step not in calc_dict:
            raise ValueError(f"cache step {lemica_step} not in calc_dict. Available: {list(calc_dict.keys())}")

        calc_list = calc_dict[lemica_step]
        bool_list = [i in calc_list for i in range(num_inference_steps)]

        FluxTransformer2DModel.forward = Lemica_call
        cls = pipeline.transformer.__class__
        cls.enable_Lemica = True
        cls.cnt = 0
        cls.num_steps = num_inference_steps
        cls.bool_list = bool_list
        cls.previous_residual = None
        cls.store = []

        # key is what user typed, lemica_step is resolved step
        tag = f"cached_step{lemica_step}"

    print(f"[INFO] tag = {tag}")

    # ===== Inference and save =====
    prompt = "A black colored car."
    t0 = time.perf_counter()
    img = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    t1 = time.perf_counter()
    elapsed = t1 - t0

    # ===== Save =====
    out = f"output_{tag}_seed{args.seed}.png"
    img.save(out)

    print(f"[INFO] saved: {out}")
    print(f"[INFO] elapsed time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()