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

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--down_sample_num', type=int, nargs='+', default=[256, 64])
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--drop_ratio', type=float, default=0.1)
parser.add_argument('--no_cls', action='store_true')

args = parser.parse_args()

model_args = ModelArgs()
model_args.llama_model_path = os.path.join(args.data_root, "weights/")
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

tokenizer = Tokenizer(model_path=os.path.join(args.llama_model_path, 'weights/tokenizer.model'))
vis_processor = transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC), transforms.ToTensor(),
                                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

image = '/cache/data/data/cocoimages/test/COCO_test2014_000000000069.jpg'
prompt = 'Describe this image.'
prompt = f'Instruction: {prompt}\nResponse:'

if image is not None:
    raw_image = Image.open(image).convert('RGB')
    image = vis_processor(raw_image).unsqueeze(0).cuda()
    indicator = 1
else:
    image = torch.Tensor(torch.zeros(3, 224, 224)).cuda()
    indicator = 0

outputs = llama.generate(
    prompts=[prompt],
    images=[image],
    indicators=[indicator],
    max_gen_len=384,
    tokenizer=tokenizer,
    temperature=0.1,
    top_p=0.75,
)
