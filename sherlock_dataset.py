import json 
import torch
from PIL import Image
import numpy as np 
from torch.utils.data import Dataset

class SherlockDataset(Dataset):
    def __init__(self, base_dir=None, 
                 json_file=None, 
                 pref_file=None,
                 ref_accept_logprobs=None,
                 ref_reject_logprobs=None,):
        
        self.base_dir = base_dir
        self.annotations = json.load(open(f"{base_dir}/{json_file}"))
        if pref_file is not None:
            with open(f"{base_dir}/{pref_file}") as f:
                data = f.readlines()

            self.accepted = [json.loads(dt)["accepted"] for dt in data]
            self.rejected = [json.loads(dt)["rejected"] for dt in data]

            if ref_accept_logprobs is not None:
                self.ref_accept_lps = np.load(f"{base_dir}/{ref_accept_logprobs}")
            
            if ref_reject_logprobs is not None:
                self.ref_reject_lps = np.load(f"{base_dir}/{ref_reject_logprobs}")
         
    def __getitem__(self, idx):
        image_url = self.annotations[idx]["inputs"]["image"]["url"]
        if "vcr1images" in image_url:
            path = "/".join(image_url.split("/")[-3:])
        elif "VG_100K" in image_url:
            path = "/".join(image_url.split("/")[-2:])
        
        image = Image.open(f"{self.base_dir}/{path}").convert("RGB")
        clue = self.annotations[idx]["inputs"]["clue"]

        return {
            "image": image,
            "clue": clue,
            "ac": self.accepted[idx],
            "rc": self.rejected[idx],
            "ac_lp": self.ref_accept_lps[idx],
            "rc_lp": self.ref_reject_lps[idx],
        }
    
    def __len__(self):
        return len(self.annotations)

class SherlockDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        images = [x["image"] for x in examples]
        prompts = [f"<image>caption en Given an Image and a Clue: {x['clue']}, What can you infer from the Visual Cues? Answer: " for x in examples]

        # duplicating prompts, images, and indices for accept and reject
        images += images
        prompts += prompts 

        targets = [x["ac"] for x in examples]
        targets += [x["rc"] for x in examples]

        ref_accept_lps = torch.tensor([x["ac_lp"] for x in examples], dtype=torch.float32)
        ref_reject_lps = torch.tensor([x["rc_lp"] for x in examples], dtype=torch.float32)
        
        tokens = self.processor(
            text=prompts, 
            images=images, 
            suffix=targets,
            return_tensors="pt", 
            padding="longest"
        )
        
        return {"tokens": tokens.to(torch.bfloat16), 
                "ref_accept_lps": ref_accept_lps,
                "ref_reject_lps": ref_reject_lps}