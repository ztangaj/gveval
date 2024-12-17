from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import json
import time
from tqdm import tqdm
import os
import base64
from math import exp
from .utils import select_prompt, video2imgs 

class Scorer:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        # print(f'Successfully loaded your OpenAI API key: {self.openai_api_key}')
        self.client = AsyncOpenAI(api_key=self.openai_api_key)

    @staticmethod
    def encode_image(image_path):
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            return None

    @staticmethod
    def normalize_responses(responses):
        tokens = [f'{i}' for i in range(101)]
        updated_responses = {s: 0 for s in tokens}
        for r in responses:
            for key, value in r.items():
                if key in updated_responses:
                    updated_responses[key] = value
        total_prob = sum(updated_responses.values())
        normalized_responses = {key: value / total_prob for key, value in updated_responses.items()}
        return normalized_responses
    
    @staticmethod
    def extract_and_normalize_responses(all_responses, token):
        for i, r in enumerate(all_responses):
            if token in r.token:
                top_logprobs = all_responses[i+1].top_logprobs
                break
        all_responses = [{top_logprobs[i].token: exp(top_logprobs[i].logprob)} for i in range(len(top_logprobs))]
        return Scorer.normalize_responses(all_responses)


    async def gveval(self, pred, ref, img=None, visual='img', setting='ref-only', accr=False, resolution='low'):
        if visual not in ['img', 'vid']:
            raise ValueError("Invalid value for visual. Allowed values are 'img' or 'vid'.")
        if resolution not in ['low', 'high']:
            raise ValueError("Invalid value for resolution. Allowed values are 'low' or 'high'.")
        try:
            # Process video if visual is 'vid' and img is a video file
            if visual == 'vid' and img and img.lower().endswith(('.mp4', '.avi')):
                video_name = os.path.splitext(os.path.basename(img))[0]
                output_dir = 'processed_videos'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                video_output_path = os.path.join(output_dir, f'{video_name}.png')
                video2imgs(img, video_output_path, num_samples=3, save_combined=True)
                img = video_output_path

            encoded_image = self.encode_image(img) if img else None
            if img and encoded_image is None:
                return None
            prompt_fp = select_prompt(visual, setting, accr)
            assert prompt_fp is not None, f"Prompt file not found for visual={visual}, setting={setting}, accr={accr}"
        
            prompt = open(prompt_fp).read()

            ignore = 3
            instance = {}

            pred = pred[0]
            ref = "'; '".join(ref)
            ref = f"'{ref}"
            # instance['reference'] = ref
            # instance['prediction'] = pred
            if visual == 'vid':
                resolution = 'high'

            cur_prompt = prompt.replace('{{Reference}}', ref).replace('{{Caption}}', pred)
            if img is not None:
                messages = [{"role": "user", "content": [{
                    "type": "text",
                    "text": cur_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                        "detail": resolution
                    }
                }]}]
            else:
                messages = [{"role": "system", "content": cur_prompt}]
            
            while True:
                try:
                    _response = await self.client.chat.completions.create(
                        model='gpt-4o-2024-05-13',
                        messages=messages,
                        temperature=1,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        logprobs=True,
                        top_logprobs=5,
                    )
                    all_responses = _response.choices[0].logprobs.content
                    reason = _response.choices[0].message.content
                    for i, r in enumerate(all_responses):
                        if '$' in r.token:
                            top_logprobs = all_responses[i+1].top_logprobs
                            break
                    if accr:
                        acc_all_responses = self.extract_and_normalize_responses(all_responses, 'α')
                        comp_all_responses = self.extract_and_normalize_responses(all_responses, 'β')
                        conc_all_responses = self.extract_and_normalize_responses(all_responses, 'ψ')
                        rel_all_responses = self.extract_and_normalize_responses(all_responses, 'δ')
                        acc_score = sum([int(i)*w for i, w in acc_all_responses.items()])
                        comp_score = sum([int(i)*w for i, w in comp_all_responses.items()])
                        conc_score = sum([int(i)*w for i, w in conc_all_responses.items()])
                        rel_score = sum([int(i)*w for i, w in rel_all_responses.items()])
                        instance['final_score'] = [acc_score, comp_score, conc_score, rel_score, (acc_score+comp_score+conc_score+rel_score)/4]
                    else:
                        all_responses = [{top_logprobs[i].token: exp(top_logprobs[i].logprob)} for i in range(len(top_logprobs))]
                        all_responses = self.normalize_responses(all_responses)

                        instance['final_score'] = sum([int(i)*w for i, w in all_responses.items()])
                    instance['reason'] = reason                   
                    return instance
                except Exception as e:
                    print(e)
                    if ("limit" in str(e)):
                        await asyncio.sleep(2)
                    else:
                        ignore -=1
                        if ignore <= 0:
                            print('skip')
                            return None
                        else:
                            print('ignored', ignore)
                    print(f"Error processing response: {e}")
                
        except Exception as e:
            print(f"Error in veval_single for image {img}: {str(e)}")
            return None

    async def gveval_batch(self, gen, gts, imgs=None, visual='img', setting='ref-only', accr=False, resolution='low'):
        tasks = []
        for i, (pred, ref) in enumerate(zip(gen, gts)):
            img = imgs[i] if imgs else None
            task = asyncio.create_task(self.gveval(pred, ref, img, visual, setting, accr, resolution))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = [r for r in results if not isinstance(r, Exception) and r is not None]
        return valid_results
