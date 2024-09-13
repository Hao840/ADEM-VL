import argparse
from dataclasses import dataclass
import os

from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torchvision.transforms import transforms

from adem.build import create_model
from adem.tokenizer import Tokenizer


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
adapter = torch.load(os.path.join(args.adapter_path, 'checkpoint-14.pth'))['model']
sd = {}
for k in adapter:
    sd[k.replace('module.', '')] = adapter[k]
_IncompatibleKeys = llama.load_state_dict(sd, False)
print(_IncompatibleKeys)

tokenizer = Tokenizer(model_path=os.path.join(args.llama_model_path, 'tokenizer.model'))
vis_processor = transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC), transforms.ToTensor(),
                                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

task_meta = {
    'artwork': dict(qa='questions_answers_YN', vis='images', suffix='jpg'),
    'celebrity': dict(qa='questions_answers_YN', vis='images', suffix='jpg'),
    'code_reasoning': dict(qa='', vis='', suffix='png'),
    'color': dict(qa='', vis='', suffix='jpg'),
    'commonsense_reasoning': dict(qa='', vis='', suffix='png'),
    'count': dict(qa='', vis='', suffix='jpg'),
    'existence': dict(qa='', vis='', suffix='jpg'),
    'landmark': dict(qa='questions_answers_YN', vis='images', suffix='jpg'),
    'numerical_calculation': dict(qa='', vis='', suffix='png'),
    'OCR': dict(qa='', vis='', suffix='jpg'),
    'position': dict(qa='', vis='', suffix='jpg'),
    'posters': dict(qa='questions_answers_YN', vis='images', suffix='jpg'),
    'scene': dict(qa='questions_answers_YN', vis='images', suffix='jpg'),
    'text_translation': dict(qa='', vis='', suffix='png'),
}

os.makedirs(os.path.join(log_dir, 'output'))
for task_idx, task in enumerate(task_meta):
    qa_path = os.path.join(args.data_root, 'MME_Benchmark_release_version', task, task_meta[task]['qa'])
    vis_path = os.path.join(args.data_root, 'MME_Benchmark_release_version', task, task_meta[task]['vis'])
    suffix = task_meta[task]['suffix']

    results = []
    for qa_name in os.listdir(qa_path):
        if not qa_name.split('.')[-1] == 'txt':
            continue
        vis_name = qa_name.split('.')[0] + f'.{suffix}'
        image = Image.open(os.path.join(vis_path, vis_name)).convert('RGB')
        image = vis_processor(image).unsqueeze(0)

        with open(os.path.join(qa_path, qa_name)) as f:
            items = [l[:-1] if '\n' in l else l for l in f.readlines()]
        for item in items:
            q, gt = item.split('\t')

            prompt = f'Instruction: {q}\nResponse:'

            output = llama.generate([prompt], images=image, indicators=[1], max_gen_len=10, tokenizer=tokenizer,
                                    temperature=0.0)

            output = output[0].replace('\n', '')

            results.append('\t'.join([vis_name, q, gt, output]) + '\n')

    with open(os.path.join(log_dir, f'output/{task}.txt'), mode='w') as f:
        f.writelines(results)
