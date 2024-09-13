import argparse
from dataclasses import dataclass
import json
import os
import re

import torch.utils.data

from adem.build import create_model
from adem.tokenizer import Tokenizer
from util.coco_karpathy_dataset import coco_caption_eval, coco_karpathy_caption_eval
from util.misc import MetricLogger


@dataclass
class ModelArgs:
    llama_model_path = './data/weights/'
    llm_model = '7B'
    max_seq_len = 512
    hidden_proj = 128
    cpu_load = False
    alpha = 0.1
    adapter_dim = 12
    gradient_checkpointing = False
    is_train = False
    data_root = './data/'
    clip = 'ViT-L/14'
    clip_root = './clip'
    down_sample_num = [256, 64]
    no_cls = False
    drop_ratio = 0.1


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--clip', type=str, default='ViT-L/14')
parser.add_argument('--clip_root', type=str, default='./clip')
parser.add_argument('--llm_model', type=str, default='7B')
parser.add_argument('--adapter_path', type=str, default='./output_dir')
parser.add_argument('--log_dir', type=str, default='./output_dir')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--down_sample_num', type=int, nargs='+', default=[256, 64])
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--drop_ratio', type=float, default=0.1)
parser.add_argument('--no_cls', action='store_true')

args = parser.parse_args()
log_dir = args.log_dir if args.log_dir is not None else './logs'
os.makedirs(log_dir, exist_ok=True)
llama_model_path = os.path.join(args.data_root, "weights/")

model_args = ModelArgs()
model_args.llama_model_path = llama_model_path
model_args.llm_model = args.llm_model
model_args.alpha = args.alpha
model_args.beta = args.beta
model_args.data_root = args.data_root
model_args.clip = args.clip
model_args.clip_root = args.clip_root
model_args.down_sample_num = args.down_sample_num
model_args.no_cls = args.no_cls
model_args.drop_ratio = args.drop_ratio

llama = create_model(model_args)
adapter = torch.load(os.path.join(args.adapter_path, 'checkpoint-4.pth'))['model']
sd = {}
for k in adapter:
    sd[k.replace('module.', '')] = adapter[k]
_IncompatibleKeys = llama.load_state_dict(sd, False)
print(_IncompatibleKeys)

tokenizer = Tokenizer(model_path=os.path.join(args.llama_model_path, 'tokenizer.model'))

dataset_test = coco_karpathy_caption_eval(image_root=os.path.join(args.data_root, 'images'),
                                          ann_root=os.path.join(args.data_root, 'coco_caption'))

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
)

llama.eval()

pattern = re.compile(r'picture of (.+)')

metric_logger = MetricLogger(delimiter="  ")
header = 'Caption generation:'
print_freq = 100

result = []
prompt = 'a picture of'
for image, image_id in metric_logger.log_every(data_loader_test, print_freq, header):

    captions = llama.generate(
        [prompt] * image.size(0), images=image, indicators=[1] * image.size(0), max_gen_len=20, tokenizer=tokenizer,
        temperature=0.0
    )

    matched_caption = []
    for c in captions:
        pred = pattern.findall(c)
        if len(pred) >= 1:
            pred = pred[0]
        else:
            print(c)
            pred = c
        matched_caption.append(pred)

    for caption, img_id in zip(matched_caption, image_id):
        result.append({"image_id": img_id.item(), "caption": caption})

result_file = os.path.join(log_dir, 'test_result.json')
json.dump(result, open(result_file, 'w'))

coco_test = coco_caption_eval(os.path.join(args.data_root, 'coco_caption/'), result_file, split='val')

log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()}}

with open(os.path.join(log_dir, "evaluate.txt"), "a") as f:
    f.write(json.dumps(log_stats) + "\n")
