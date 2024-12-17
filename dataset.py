import os
from PIL import Image
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

class Flickr8k(torch.utils.data.Dataset):
    def __init__(self, json_file, root='datasets/flickr8k/', 
                 transform=None, load_images=False):
        self.im_folder = os.path.join(root, 'images')
        self.transform = transform
        self.load_images = load_images

        with open(os.path.join(root, json_file)) as fp:
            data = json.load(fp)

        self.data = list()
        if os.path.basename(json_file) == 'flickr8k.json':
            for i in data:
                cand_len = len(data[i]['human_judgement'])
                if cand_len % 3 != 0:
                    continue
                # for human_judgement in data[i]['human_judgement']:
                for j in range(0, cand_len, 3):
                    human_judgement = data[i]['human_judgement'][j]
                    if np.isnan(human_judgement['rating']) or np.isnan(data[i]['human_judgement'][j+1]['rating']) or np.isnan(data[i]['human_judgement'][j+2]['rating']):
                        print('NaN')
                        continue
                    human_score = data[i]['human_judgement'][j]['rating'] + data[i]['human_judgement'][j+1]['rating'] + data[i]['human_judgement'][j+2]['rating']
                    human_score /= 3
                    d = {
                        'image': data[i]['image_path'].split('/')[-1],
                        'references': [' '.join(gt.split()) for gt in data[i]['ground_truth']],
                        'candidate': ' '.join(human_judgement['caption'].split()),
                        # 'human_score': human_judgement['rating']
                        'human_score': human_score
                    }
                    self.data.append(d)
        else:
            for i in data:
                for human_judgement in data[i]['human_judgement']:
                    if np.isnan(human_judgement['rating']):
                        print('NaN')
                        continue
                    d = {
                        'image': data[i]['image_path'].split('/')[-1],
                        'references': [' '.join(gt.split()) for gt in data[i]['ground_truth']],
                        'candidate': ' '.join(human_judgement['caption'].split()),
                        'human_score': human_judgement['rating']
                    }
                    self.data.append(d)

    def get_image(self, filename):
        img = Image.open(os.path.join(self.im_folder, filename)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_idx = self.data[idx]['image']
        candidate = self.data[idx]['candidate']
        references = self.data[idx]['references']
        score = self.data[idx]['human_score']

        if self.load_images:
            im = self.get_image(im_idx)
        else:
            im = os.path.join(self.im_folder, im_idx)

        return im, candidate, references, score

class MSVD(torch.utils.data.Dataset):
    def __init__(self, json_file, root='data/YouTubeClips/', transform=None, load_videos=False):
        self.root = root
        self.transform = transform
        self.load_videos = load_videos

        with open(json_file) as fp:
            data = json.load(fp)

        self.data = list()
        for k, v in data.items():
            d = {
                'video': os.path.join(root, v['vname']),
                'references': [' '.join(gt.split()) for gt in v['ref']],
                'candidate': ' '.join(v['candidate'].split()),
                'human_score': v['human_score']
            }
            self.data.append(d)


    def get_video(self, filename):
        # Placeholder for video loading logic
        # For example, you could use OpenCV or another library to load the video
        video = filename  # Simplified for this example
        if self.transform:
            video = self.transform(video)
        return video

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_idx = self.data[idx]['video']
        candidate = self.data[idx]['candidate']
        references = self.data[idx]['references']
        score = self.data[idx]['human_score']

        if self.load_videos:
            video = self.get_video(video_idx)
        else:
            video = video_idx

        return video, candidate, references, score
    

class FoilDatset:
    def __init__(self, coco_root_path="data/coco", foil_path="data/foil/foilv1.0_test_2017.json"):
        coco_root_path = Path(coco_root_path)
        coco_path = coco_root_path / Path("captions_val2014.json")
        coco_refs = self._read_coco(coco_path)
        self.data = self._build_foil(foil_path, coco_refs) # data[anno_id][foil or orig] = [anno1, anno2, ...]
        self.coco_root_path = coco_root_path
        self.dataset = {"one_ref" : None, "four_ref" : None}

    def _read_coco(self, coco_annos):
        refs = {}
        with open(coco_annos) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            refs.setdefault(ann['image_id'],[]).append(ann['caption'])
        return refs
    
    def _build_foil(self, path, coco_refs):
        with open(path) as f:
            self.data = json.load(f)
        # For preliminary testing
        images = self.data["images"]
        annos = self.data["annotations"]

        data = {}
        imgid_to_img = {img["id"] : img for img in images}
        for anno in annos:
            anno_id = anno["id"]
            data.setdefault(anno_id, {"foil" : [], "orig" : []})
            key = "foil" if anno["foil"] else "orig"
            anno["image"] = imgid_to_img[anno["image_id"]]
            anno["refs"] = coco_refs[anno["image_id"]]
            data[anno_id][key].append(anno)
        
        return data

    def get_data(self,one_ref):
        key = "one_ref" if one_ref else "four_ref"
        if self.dataset[key] is not None:
            return self.dataset[key]
        
        dataset = []
        for _, data in (pbar := tqdm(self.data.items())):  # data[anno_id][foil or orig] = [anno1, anno2, ...]
            pbar.set_description("Prepare dataset ...")
            foiles, origs = data["foil"], data["orig"]

            assert len(origs) == 1
            N = len(foiles)
            for foil, orig in zip(foiles, [origs[0]]*N):
                refs = foil["refs"]
                refs = [r for r in refs if r != orig["caption"]]
                if one_ref:
                    refs = [refs[0]]
                
                filename = Path(foil["image"]["file_name"])
                img_path = Path("data/coco/val2014") / filename

                dataset.append({
                    "imgid" : img_path,
                    "refs": refs,
                    "mt": foil["caption"],
                    "type": "foil"
                })
                dataset.append({
                    "imgid" : img_path,
                    "refs": refs,
                    "mt": orig["caption"],
                    "type": "orig"
                })
        
        self.dataset[key] = dataset
        return self.dataset[key]