import json
import random
from tqdm import tqdm
import json
import yaml
from dataset import FoilDatset
from PIL import Image
from pathlib import Path
from copy import deepcopy
from termcolor import colored
from evaluation import evaluate_gveval
import asyncio

def rprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'white', 'on_red', attrs=["bold"]))

def yprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'yellow',attrs=["bold"]))

def gprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'green',attrs=["bold"]))

def collect_acc(memory, dataset_name, method, acc):
    memory.setdefault(dataset_name, {})
    memory[dataset_name].update({method : acc})
    gprint(f"[{dataset_name}]",method,acc)

def process_gveval_scores(gveval_scores):
    # Convert keys to integers and sort them
    sorted_keys = sorted(int(k) for k in gveval_scores.keys())
    
    # Create a new ordered dictionary
    ordered_scores = {str(k): gveval_scores[str(k)] for k in sorted_keys}
    
    # Drop unpaired items
    paired_scores = {}
    for i in range(0, len(ordered_scores), 2):
        if str(i) in ordered_scores and str(i + 1) in ordered_scores:
            paired_scores[str(i)] = ordered_scores[str(i)]
            paired_scores[str(i + 1)] = ordered_scores[str(i + 1)]
    
    # Convert to list of dictionaries
    result = [paired_scores[k] for k in sorted(paired_scores.keys(), key=int)]
    
    return result

async def gveval(dataset,args):
    data = []
    for data_ in (pbar := tqdm(dataset)):
        pbar.set_description("Prepare dataset ...")
        data.append(data_)
    
    imgs = {}
    gen = {}
    gts = {}
    for i, instant in enumerate(data):
        imgs['%d' % (i)] = str(instant["imgid"])
        gen['%d' % (i)] = [instant["mt"], ]
        gts['%d' % (i)] = instant["refs"]
    
    gveval_scores = await evaluate_gveval(gen, gts, imgs, visual='img', setting='combined', accr=False, resolution='low', batch_size=20)
    gveval_scores = process_gveval_scores(gveval_scores)

    # Define the file path
    file_path = Path('foil.json')

    # Load existing data if the file exists
    if file_path.exists():
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    # Append the new gveval_scores to the existing data
    existing_data.extend(gveval_scores)

    # Save the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
    sys_score = [g['gveval'] for g in gveval_scores]
    
    # _, sys_score = model.predict(data,cuda=True,batch_size=32)
    return sys_score

async def compute_acc(model_fn,dataset,one_ref,**kwargs):
    # Split by buckets because images do not fit in RAM.
    bucket_count = 20
    data = dataset.get_data(one_ref)
    
    print("Compute ...")
    sys_score = []
    for i in range(bucket_count):
        bucket_size = len(data) // bucket_count
        subset = deepcopy(data[i*bucket_size:(i+1)*bucket_size])
        for j, sub in enumerate(pbar := tqdm(subset)):
            pbar.set_description(f"Processing {i+1}/{bucket_count}")
            subset[j].update({"img" : Image.open(sub["imgid"]).convert("RGB")})
        sub_sys_score = await model_fn(subset,**kwargs)
        sys_score.extend(sub_sys_score)
        del subset
    
    assert len(sys_score) == len(data)
    assert len(sys_score) % 2 == 0

    acc = 0.
    N = len(sys_score) // 2
    for i in range(0,2*N,2):
        s1 = sys_score[i] # foil
        s2 = sys_score[i+1] # orig
        
        # sanity check
        assert data[i]["type"] == "foil" and data[i+1]["type"] == "orig"

        if s2 > s1:
            acc += 1.

    acc /= N
    rprint(f"acc: {acc}")
    
    return acc


async def compute_foil(args, memory, tops):
    dataset = FoilDatset()
    dataset_name = "foil"
    for one_ref in [True, False]:
        suffix = "(one_ref)" if one_ref else "(four-ref)"
        dataset_name += suffix
        gveval_acc = await compute_acc(gveval, dataset, one_ref, args=args)
        collect_acc(memory, dataset_name, f"G-VEval{suffix}", gveval_acc)

    # aggregate
    max_acc = ("", 0.)
    for method, acc in memory[dataset_name].items():
        if max_acc[1] < acc:
            max_acc = (method, acc)

    rprint("[TOP]")
    rprint(max_acc)
    tops[dataset_name] = max_acc

    return memory, tops

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute foil accuracy")
    # parser.add_argument('--foil', action='store_true')
    # parser.add_argument('--polos', action='store_true', help="Use Polos model")
    args = parser.parse_args()
    memory = {}
    tops = {}
    memory, tops = asyncio.run(compute_foil(args, memory, tops))
    rprint("Final Results:")
    for dataset_name, (method, acc) in tops.items():
        rprint(f"{dataset_name}: {method} with accuracy {acc}")
