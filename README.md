# ADEM-VL

Official code of paper: ADEM-VL: Adaptive and Embedded Fusion for Efficient Vision-Language Tuning.

Zhiwei Hao, Jianyuan Guo, Li Shen, Yong Luo, Han Hu*, Yonggang Wen

## Preparation 
```bash
conda create -n adem python=3.8 -y
conda activate adem

# install pytorch
conda install pytorch==1.13.1 torchvision==0.14.1 -c pytorch

# install dependencies
pip install -r requirements.txt
```
**Data Preparation**

*the data preparation instruction is borrowed from [LaVIN](https://github.com/luogen1996/LaVIN/tree/main)*.

- For ScienceQA, please prepare the dataset from the [official repo](https://github.com/lupantech/ScienceQA).
- For Multimodal Chatbot, download the images in _train2014_ split from [MSCOCO](http://images.cocodataset.org/zips/train2014.zip), and obtain the prepared 52k text-only and 158k text-image instruction-following data from [here](https://drive.google.com/file/d/1gORDPruqwXbgy6NYmhpDXO7t089yzsg3/view?usp=share_link).
- Obtain the weights of LLaMA from [this form](https://forms.gle/jk851eBVbX1m5TAv5)  (official) or Download [LLaMA-7B](https://huggingface.co/nyanko7/LLaMA-7B/tree/main) and [LLaMA-13B](https://huggingface.co/TheBloke/llama-13b) from HuggingFace (unofficial).

After that, the file structure should look like:

```bash
ADEM-VL/
  |-- adem
  |-- train.py
  ......
  |-- data/
      |-- problem.json
      |-- pid_splits.json
      |-- captions.json
      |-- all_data.json
      |-- images
          |-- train2014      # MSCOCO 2014
          |-- val2014        # MSCOCO 2014
          |-- train          # ScienceQA train image
          |-- val            # ScienceQA val image
          |-- test           # ScienceQA test image
      |-- weights
          |-- tokenizer.model
              |--7B
                  |-- params.json
                  |-- consolidated.00.pth
              |--13B
                  |-- params.json
                  |-- consolidated.00.pth
                  |-- consolidated.01.pth
```
## Fine-tuning
Reproduce the performance of LaVIN-7B.

**ScienceQA**

```shell
torchrun --nproc_per_node 8 train.py --data_root /path/to/data/ --clip_root /path/to/data/weights/clip/ --caption_file /path/to/data/captions.json --llama_model_path /path/to/data/weights/ --llm_model 7B --max_seq_len 512 --batch_size 2 --accum_iter 2 --epochs 20 --warmup_epochs 2 --blr 9e-3 --weight_decay 0.02 --adapter_dim 12 --alpha 0.1 --beta 0.01 --drop_ratio 0.1 --down_sample_num 256 64 --dataset sqa
```

**COCO caption**

```shell
torchrun --nproc_per_node 8 train.py --data_root /path/to/data/ --clip_root /path/to/data/weights/clip/ --caption_file /path/to/data/captions.json --llama_model_path /path/to/data/weights/ --llm_model 7B --max_seq_len 512 --batch_size 2 --accum_iter 2 --epochs 5 --warmup_epochs 0.1 --blr 9e-3 --weight_decay 0.02 --adapter_dim 12 --alpha 0.1 --beta 0.01 --drop_ratio 0.1 --down_sample_num 256 64 --dataset coco_caption
```

**Instruction following**

```shell
torchrun --nproc_per_node 8 train.py --data_root /path/to/data/ --clip_root /path/to/data/weights/clip/ --caption_file /path/to/data/captions.json --llama_model_path /path/to/data/weights/ --llm_model 7B --max_seq_len 512 --batch_size 2 --accum_iter 2 --epochs 15 --warmup_epochs 0.2 --blr 9e-3 --weight_decay 0.02 --adapter_dim 12 --alpha 0.1 --beta 0.01 --drop_ratio 0.1 --down_sample_num 256 64 --dataset instruction
```

To train on fewer GPUs, you can reduce the number of gpus in the scripts and increase gradient accumulation via ```--accum_iter``` to guarantee the total batch size of 32.

## Evaluation

Evaluate fine-tuned model on each tasks.

**ScienceQA**

```shell
python eval_sqa.py --data_root /path/to/data/ --clip_root /path/to/data/weights/clip/ --model 7B --adapter_path ./output_dir --alpha 0.1 --beta 0.01 --drop_ratio 0.1 --down_sample_num 256 64
```

**COCO caption**

```shell
# prepare required packages
pip install pycocoevalcap pycocotools

python eval_caption.py --data_root /path/to/data/ --clip_root /path/to/data/weights/clip/ --model 7B --adapter_path ./output_dir --alpha 0.1 --beta 0.01 --drop_ratio 0.1 --down_sample_num 256 64
```

**Instruction following**

- **MME**

1. Download MME images and eval_tool from the [MME repo](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/Evaluation/README.md).
2. Run the following command to obtain model predictions:

```shell
python eval_instruction.py --data_root /path/to/data/ --clip_root /path/to/data/weights/clip/ --model 7B --adapter_path ./output_dir --alpha 0.1 --beta 0.01 --drop_ratio 0.1 --down_sample_num 256 64
```

3. Calculate MME results by executing the calculation script comes from the MME eval_tool.

- **More tasks**

Evaluation on more tasks can be achieved in a similar way as MME based on tookits like [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and [vlm-evaluation](https://github.com/TRI-ML/vlm-evaluation).

## Model Zoo
| Model    | Task                  | Results                  | Weights                                                      | Training log                                                 |
| -------- | --------------------- | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LLaMA-7B | ScienceQA             | Averaged accuracy=94.01  | [[Link]](https://github.com/Hao840/ADEM-VL/releases/download/checkpoint/checkpoint_7B_sqa.pth) | [[Link]](https://github.com/Hao840/ADEM-VL/releases/download/checkpoint/train_log_7B_sqa.txt) |
| LLaMA-7B | COCO caption          | BLEU-4=38.5, CIDEr=130.1 | [[Link]](https://github.com/Hao840/ADEM-VL/releases/download/checkpoint/checkpoint_7B_caption.pth) | [[Link]](https://github.com/Hao840/ADEM-VL/releases/download/checkpoint/train_log_7B_caption.txt) |
| LLaMA-7B | Instruction following | MME-P=969.7, MME-C=258.9 | [[Link]](https://github.com/Hao840/ADEM-VL/releases/download/checkpoint/checkpoint_7B_instruction.pth) | [[Link]](https://github.com/Hao840/ADEM-VL/releases/download/checkpoint/train_log_7B_instruction.txt) |

## Citation
If you find this work helpful, please cite our paper:
```BibTeX
coming soon
```

## Acknowledgement
This repo borrows some data and codes from [LaVIN](https://github.com/luogen1996/LaVIN/tree/main), [MemVP](https://github.com/JieShibo/MemVP), and [BLIP](https://github.com/salesforce/BLIP). Thanks for their great works.
