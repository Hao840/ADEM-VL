import argparse
from dataclasses import dataclass
import json
import os
import re

import pandas as pd
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from adem.build import create_model
from adem.tokenizer import Tokenizer
from util.base_prompt import build_prompt


@dataclass
class PromptArgs:
    prompt_format = 'QCM-A'
    use_caption = True
    options = ["A", "B", "C", "D", "E"]


def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def get_scores(result_file, data_file):
    # read result file
    results = json.load(open(result_file))
    num = len(results)
    assert num == 4241

    sqa_data = json.load(open(data_file))

    # construct pandas data
    sqa_pd = pd.DataFrame(sqa_data).T
    res_pd = sqa_pd[sqa_pd['split'] == 'test']  # test set

    # update data
    for index, row in res_pd.iterrows():
        res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
        res_pd.loc[index, 'has_text'] = True if row['hint'] else False
        res_pd.loc[index, 'has_image'] = True if row['image'] else False
        res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False

        label = row['answer']
        pred = int(results[index])
        res_pd.loc[index, 'pred'] = pred
        res_pd.loc[index, 'true_false'] = (label == pred)

    # accuracy scores
    acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100

    scores = {
        'acc_natural':
            get_acc_with_contion(res_pd, 'subject', 'natural science'),
        'acc_social':
            get_acc_with_contion(res_pd, 'subject', 'social science'),
        'acc_language':
            get_acc_with_contion(res_pd, 'subject', 'language science'),
        'acc_has_text':
            get_acc_with_contion(res_pd, 'has_text', True),
        'acc_has_image':
            get_acc_with_contion(res_pd, 'has_image', True),
        'acc_no_context':
            get_acc_with_contion(res_pd, 'no_context', True),
        'acc_grade_1_6':
            get_acc_with_contion(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
        'acc_grade_7_12':
            get_acc_with_contion(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
        'acc_average':
            "{:.2f}".format(acc_average),
    }

    return scores


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return -1  # return random.choice(range(len(choices)))


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


def main():
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
    adapter = torch.load(os.path.join(args.adapter_path, 'checkpoint-19.pth'))['model']
    sd = {}
    for k in adapter:
        sd[k.replace('module.', '')] = adapter[k]
    _IncompatibleKeys = llama.load_state_dict(sd, False)
    print(_IncompatibleKeys)

    tokenizer = Tokenizer(model_path=os.path.join(llama_model_path, 'tokenizer.model'))

    split = 'test'
    print('split: ', split)
    problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
    captions = json.load(open(os.path.join(args.data_root, 'captions.json')))["captions"]
    image_path = os.path.join(args.data_root, 'images', split)
    qids = pid_splits['%s' % (split)]
    total_items = len(qids)
    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    print('total_items: ', total_items)

    image_transforms = transforms.Compose(
        [transforms.Resize((224, 224), interpolation=Image.BICUBIC), transforms.ToTensor(),
         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    prompt_args = PromptArgs()

    pattern = re.compile(r'([A-Z])')

    answers = []
    preds = []

    print_freq = 100
    with tqdm(total=total_items // args.batch_size + 1, ncols=0) as pbar:
        for i in range(total_items // args.batch_size + 1):
            if i % print_freq == 0:
                pbar.update(print_freq)

            batch_qids = qids[i * args.batch_size:(i + 1) * args.batch_size]
            if len(batch_qids) == 0:
                break
            indicators = []
            prompts = []
            images = []
            for qid in batch_qids:
                prompt, _ = build_prompt(problems, qid, prompt_args)
                prompt += 'The answer is'
                answer = problems[qid]["answer"]
                if problems[qid]['image'] is not None:
                    image = Image.open(os.path.join(image_path, qid, 'image.png')).convert('RGB')
                    image = image_transforms(image)
                    indicator = 1
                else:
                    image = torch.Tensor(torch.zeros(3, 224, 224).float())
                    indicator = 0
                prompts.append(prompt)
                answers.append(answer)
                images.append(image)
                indicators.append(indicator)

            images = torch.stack(images)
            results = llama.generate(
                prompts, images=images, indicators=indicators, max_gen_len=1, tokenizer=tokenizer, temperature=0.0
            )

            for result in results:
                pred = pattern.findall(result)

                if len(pred) >= 1:
                    pred = pred[0]  # 'A', 'B', ...
                else:
                    # print(result)
                    pred = "FAILED"
                preds.append(pred)

    # evaluations
    results = {}
    correct = 0
    for i, prediction in enumerate(preds):
        pred_idx = get_pred_idx(prediction, problems[qids[i]]["choices"],
                                prompt_args.options)  # 0, 1, ..., 4
        if pred_idx == answers[i]:
            correct += 1
        results[qids[i]] = pred_idx
    acc = correct / len(results) * 100
    print('overall accuracy: ', acc)

    with open(os.path.join(log_dir, 'preds.json'), 'w') as f:
        json.dump(results, f)

    scores = get_scores(os.path.join(log_dir, 'preds.json'), os.path.join(args.data_root, 'problems.json'))
    print(scores)
    with open(os.path.join(log_dir, 'eval_log.txt'), 'w') as f:
        f.write(str(scores))


if __name__ == '__main__':
    main()
