<!-- ## **TeaCache4FLUX** -->
# LeMiCa4Qwen-Image

[LeMiCa](https://github.com/UnicomAI/LeMiCa) already supports accelerated inference for [Qwen-Image](https://github.com/QwenLM/Qwen-Image) and provides three optional acceleration paths based on the balance between quality and speed.
 

![visualization](../assets/qw-image.jpg)

## 📊 Inference Latency 
#### Comparisons on a Single H800


|      Qwen-Image       |        LeMiCa (slow, B=25)       |    LeMiCa (Medium, B=17)    |     LeMiCa (fast, B=10)    |
|:-----------------------:|:----------------------------:|:--------------------:|:---------------------:|
|         ~32.68 s           |        ~18.1 s                 |     ~13.3 s            |       ~9.75 s             |

## 🛠️ Installation & Usage 

Please refer to [Qwen-Image](https://github.com/QwenLM/Qwen-Image)
```shell
# Required for Qwen2.5-VL support
pip install transformers>=4.51.3

# Install the necessary diffusers library component
pip install git+https://github.com/huggingface/diffusers
```
LeMiCa provides three acceleration modes that balance speed and image quality.
You can configure them using the --cache argument:

```bash
python inference_qwenimage.py
python inference_qwenimage.py --cache slow
python inference_qwenimage.py --cache medium
python inference_qwenimage.py --cache fast
```

## 📖 Citation
If you find **LeMiCa** useful in your research or applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry:

```bibtex
@inproceedings{gao2025lemica,
  title     = {LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation},
  author    = {Huanlin Gao and Ping Chen and Fuyuan Shi and Chao Tan and Zhaoxiang Liu and Fang Zhao and Kai Wang and Shiguo Lian},
  journal   = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2511.00090}
}
```

## Acknowledgements

We would like to thank the contributors to the  [Qwen-Image](https://github.com/QwenLM/Qwen-Image),  [TeaCache](https://github.com/ali-vilab/TeaCache) and [Diffusers](https://github.com/huggingface/diffusers).