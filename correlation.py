import asyncio
from dataset import Flickr8k, MSVD
import scipy
from evaluation import get_all_metrics, PTBTokenizer, evaluate_gveval
from torch.utils.data import DataLoader
import torch
import json
import numpy as np
from tqdm import tqdm
import pickle
import os

def collate_fn(batch):
    print(f"collate_fn: {len(batch)}")
    if isinstance(batch, tuple) and isinstance(batch[0], list):
        return batch
    elif isinstance(batch, list):
        transposed = list(zip(*batch))
        return [collate_fn(samples) for samples in transposed]
    return torch.utils.data.default_collate(batch)

def correllation(score, human_scores):
    kendalltau_b = 100 * scipy.stats.kendalltau(score, human_scores, variant='b')[0]
    kendalltau_c = 100 * scipy.stats.kendalltau(score, human_scores, variant='c')[0]
    print('Kendall Tau-b: %.3f \t  Kendall Tau-c: %.3f' % (kendalltau_b, kendalltau_c))

def compute_scores(imgs, gen, gts, gvevals):
    all_scores = dict()
    gts = PTBTokenizer.tokenize(gts)
    gen = PTBTokenizer.tokenize(gen)
    
    all_scores_metrics = get_all_metrics(gts, gen, return_per_cap=True)
    
    for k, v in all_scores_metrics.items():
        if k == 'BLEU':
            all_scores['BLEU-1'] = v[0]
            all_scores['BLEU-4'] = v[-1]
        else:
            all_scores[k] = v
    all_scores['G-VEval'] = [instance['gveval'] for instance in gvevals.values()]
    instance = {}
    for i, (k, v) in enumerate(gvevals.items()):
        instance[k] = {
            'image': imgs[k],
            'references': gts[k],
            'candidate': v['candidate'],
            'reason': v['reason'],
            'G-VEval': v['gveval']
        }
        for t, met in all_scores.items():
            instance[k][t] = met[i]
        
    return all_scores, instance
    

async def evaluate_flickr8k(json_file, root, setting='ref-only', batch_size=10):
    dataset = Flickr8k(json_file=json_file, root=root, load_images=False)
    all_scores = []
    
    image_ids = [item[0] for item in dataset]
    candidates = [item[1] for item in dataset]
    references = [item[2] for item in dataset]
    human_scores = [item[3] for item in dataset]
    
    imgs = {}
    gen = {}
    gts = {}
    scores = {}

    for i, (im_i, gts_i, gen_i, score_i) in enumerate(zip(image_ids, references, candidates, human_scores)):
        imgs['%d' % (i)] = im_i
        gen['%d' % (i)] = [gen_i, ]
        gts['%d' % (i)] = gts_i
        scores['%d' % (i)] = score_i
    
    gveval_scores = await evaluate_gveval(gen, gts, imgs, visual='img', setting=setting, accr=False, resolution='low', batch_size=batch_size)
    all_scores, instance = compute_scores(imgs, gen, gts, gveval_scores)
    human_scores = []
    for k in instance.keys():
        human_scores.append(scores[k])

    for k, v in all_scores.items():
        kendalltau_b = 100 * scipy.stats.kendalltau(v, human_scores, variant='b')[0]
        kendalltau_c = 100 * scipy.stats.kendalltau(v, human_scores, variant='c')[0]
        print('%s \t Kendall Tau-b: %.3f \t  Kendall Tau-c: %.3f'
              % (k, kendalltau_b, kendalltau_c))
    
    return all_scores

async def evaluate_msvd(json_file, root, setting='ref-only', batch_size=3):
    dataset = MSVD(json_file=json_file, root=root, load_videos=False)
    all_scores = []
    
    image_ids = [item[0] for item in dataset]
    candidates = [item[1] for item in dataset]
    references = [item[2] for item in dataset]
    human_scores = [item[3] for item in dataset]
    
    imgs = {}
    gen = {}
    gts = {}
    scores = {}

    for i, (im_i, gts_i, gen_i, score_i) in enumerate(zip(image_ids, references, candidates, human_scores)):
        imgs['%d' % (i)] = im_i
        gen['%d' % (i)] = [gen_i, ]
        gts['%d' % (i)] = gts_i
        scores['%d' % (i)] = score_i
    
    gveval_scores = await evaluate_gveval(gen, gts, imgs, visual='vid', setting=setting, accr=True, resolution='high', batch_size=batch_size)
    
    all_scores, instance = compute_scores(imgs, gen, gts, gveval_scores)
    human_scores = []
    for k in instance.keys():
        human_scores.append(scores[k])
    extended_scores = {}
    human_scores = []
    for i, k in enumerate(instance.keys()):
        for s in scores[k]:
            human_scores.append(s)
            for metric in all_scores.keys():
                if metric not in extended_scores:
                    extended_scores[metric] = []
                extended_scores[metric].append(all_scores[metric][i])

    dimensions = ["Acc", "Com", "Con", "Rel", "Avg"]
    for k, v in all_scores.items():
        if 'G-VEval' in k:
            for i, accr in enumerate(dimensions):
                print(f"{k}-{accr}", end=" ")
                correllation([item[i] for item in extended_scores[k]], [h[i] for h in human_scores])
        else:   
            print(k, end=" ")
            correllation(extended_scores[k], [h[4] for h in human_scores])
    
    return all_scores

if __name__ == "__main__":
    # Example usage for Flickr8k
    json_file_expert = 'flickr8k.json'
    json_file_cf = 'crowdflower_flickr8k.json'
    root_flickr8k = 'data/flickr8k/'

    print("Evaluating flickr8k-expert dataset:")
    all_scores_expert = asyncio.run(evaluate_flickr8k(json_file_expert, root_flickr8k, setting='combined'))

    print("\nEvaluating flickr8k-cf dataset:")
    all_scores_cf = asyncio.run(evaluate_flickr8k(json_file_cf, root_flickr8k, setting='combined'))

    # Example usage for MSVD
    json_file_msvd = 'MSVD-Eval.json'
    root_msvd = 'data/YouTubeClips/'

    print("\nEvaluating MSVD dataset:")
    all_scores_msvd = asyncio.run(evaluate_msvd(json_file_msvd, root_msvd, setting='ref-only'))
