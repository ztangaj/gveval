'''
Automatic generation evaluation metrics wrapper
The most useful function here is
get_all_metrics(refs, cands)
'''
from .tokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from .gveval.scorer import Scorer
from tqdm import tqdm
import pickle
import os
import asyncio

def get_all_metrics(refs, cands, return_per_cap=False):
    metrics = []
    names = []

    pycoco_eval_cap_scorers = [(Bleu(4), 'BLEU'),
                               (Meteor(), 'METEOR'),
                               (Rouge(), 'ROUGE'),
                               (Cider(), 'CIDER'),
                            #    (Spice(), 'SPICE')
                               ]

    for scorer, name in pycoco_eval_cap_scorers:
        overall, per_cap = pycoco_eval(scorer, refs, cands)
        if return_per_cap:
            metrics.append(per_cap)
        else:
            metrics.append(overall)
        names.append(name)

    metrics = dict(zip(names, metrics))
    return metrics


def pycoco_eval(scorer, refs, cands):
    '''
    scorer is assumed to have a compute_score function.
    refs is a list of lists of strings
    cands is a list of predictions
    '''
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores


async def evaluate_gveval(candidates, references, images, checkpoint_file=None, visual='img', setting='ref-only', accr=False, resolution='low', batch_size=2):
    scorer = Scorer()
    output_json = {}
    start_index = 0
    
    # Load checkpoint if exists
    if checkpoint_file is not None:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                output_json = checkpoint['output_json']
                start_index = checkpoint['processed_items']
    
    # Skip already processed items
    for _ in range(start_index // len(candidates)):
        next(iter(candidates))
        
    async def process_batch(batch):
        batch_results = {}
        for k, im_i, gen_i, gts_i in batch:
            try:
                if setting == 'ref-free':
                    results = await scorer.gveval_batch([gen_i], [[]], imgs=[im_i], visual=visual, setting=setting, accr=accr, resolution=resolution)
                else:
                    refs = [[gts_i[0]]] if setting == '1-ref' else [gts_i]
                    results = await scorer.gveval_batch([gen_i], refs, imgs=[im_i], visual=visual, setting=setting, accr=accr, resolution=resolution)
                for result in results:
                    if result is not None:
                        batch_results[k] = {
                            'candidate': gen_i,
                            'references': gts_i,
                            'gveval': result['final_score'],
                            'reason': result['reason']
                        }
            except Exception as e:
                print(f"Error processing item: {str(e)}")
                continue
        return batch_results
    
    tasks = []
    batches = []
    image_keys = list(images.keys())
    for i in range(start_index, len(image_keys), batch_size):
        batch_keys = image_keys[i:i+batch_size]
        batch = [(k, images[k], candidates[k], references[k]) for k in batch_keys]
        batches.append(batch)
        tasks.append(process_batch(batch))
    
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing G-VEval"):
        batch_results = await f
        results.append(batch_results)
    
    # Flatten the list of batches to maintain the original order
    for batch_results in results:
        output_json.update(batch_results)
        
    return output_json
    