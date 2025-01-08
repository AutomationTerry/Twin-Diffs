import os
import csv
from tqdm import tqdm
import torch
import numpy as np
import re

from torch.utils.data import Dataset


# prompt to text embedding
class PromptDataset(Dataset):
    def __init__(self, cfg, text_encoder, text_tokenizer):
        super().__init__()
        self.cfg = cfg
        self.model = cfg.model
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.prompt_root = cfg.prompt.root
        self.instance_file = cfg.prompt.instance_file
        with open(self.instance_file, "r") as f:
            self.instance_list = [fn.strip() for fn in f.readlines()]
        self.max_length = cfg.prompt.max_length
        self.text_embedding_root = cfg.prompt.text_embedding_root
        self.rebuild = cfg.prompt.rebuild
        self.text_embedding_root = cfg.prompt.text_embedding_root
        self.invalid_file_pm = cfg.prompt.invalid_file_pm

        assert (
            os.path.basename(self.prompt_root).split(".")[-1] == "csv"
        ), "Prompt file must be a csv file"
        id_list = []
        prompt_list = []
        assert cfg.prompt.root is not None, "Prompt root is not defined"
        with open(self.prompt_root, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                id_list.append(row[0])
                prompt_list.append(row[1])

        invalid_list = []
        prompts = {}
        assert self.instance_list is not None, "Instance file is not defined"
        print(f"Train set has {len(self.instance_list)} instances.")
        print("Loading prompts...")

        if self.rebuild or not os.path.exists(self.text_embedding_root):
            for i in tqdm(self.instance_list, total=len(self.instance_list)):
                if i not in id_list:
                    print(f"{i} not found in the prompt dataset")
                    invalid_list.append(i)
                    continue
                prompt_path = os.path.join(self.text_embedding_root, f"{i}.npz")
                if i in prompt_list:
                    prompt = prompt_list[i]  # prompt为字典prompt_dict的值
                    prompt = self.clean_prompt(prompt)
                    prompts = self.multi_views_prompt(prompt)
                    prompts_attribute = self.tokenize_prompt(prompts, i)
                    prompts = {"id": i, "prompts_attribute": prompts_attribute}
                    np.savez(prompt_path, **prompts)

            with open(self.invalid_file_pm, "w") as f:
                for i in invalid_list:
                    f.write(f"{i}\n")
                    invalid_len = len(invalid_list)
            if invalid_len > 0:
                answer = input(
                    "please check the prompts data, do you want to terminate the training? (y/n): "
                )
                if answer.lower() == "y":
                    print("Terminating the training...")
                    raise Exception("Invalid prompts data found.")
                else:
                    print("Continuing the training...")
        print("Prompts data is ready.")

    @torch.no_grad()
    def clean_prompt(self, prompt):
        assert isinstance(prompt, str), "Prompt must be strings"
        while any(char in prompt for char in ['"', "'", ","]):
            prompt = prompt.replace('"', "")
            prompt = prompt.replace("'", "")
            prompt = prompt.replace(",", "")
            prompt = prompt.replace(".", "")
            prompt = re.sub(" +", " ", prompt)
        # TODO：promptprocessor实现前后左右视图
        # TODO：prompts = [p + ' Black background.' for p in prompts]
        # TODO：tokenize
        return prompt

    @torch.no_grad()
    def multi_views_prompt(self, prompt):
        assert isinstance(prompt, str), "Prompt must be strings"
        prompts = [
            "Right side view. " + prompt,
            "Right oblique rear view. " + prompt,
            "Rear view. " + prompt,
            "Left oblique rear view. " + prompt,
            "Left view. " + prompt,
            "Left oblique front view. " + prompt,
            "Front view. " + prompt,
            "Right oblique front view. " + prompt,
        ]
        prompts = [p + ". Black background." for p in prompts]
        return prompts

    @torch.no_grad()
    def tokenize_prompt(self, prompts, id) -> dict:
        token = self.text_tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        token_ids = token.input_ids
        token_attention_mask = token.attention_mask
        token_attention_mask = token_attention_mask.to(self.text_encoder.device)

        if self.rebuild or not os.path.exists(
            os.path.join(self.text_embedding_root, f"{id}.npz")
        ):
            text_embedding = self.text_encoder(
                token_ids.to(self.text_encoder.device),
                attention_mask=token_attention_mask,
            )[0].cpu()
            text_embedding_np = text_embedding.float().detach().numpy()
            os.makedirs(
                os.path.dirname(os.path.join(self.text_embedding_root, f"{id}.npz")),
                exist_ok=True,
            )
            with open(os.path.join(self.text_embedding_root, f"{id}.npz"), "wb") as f:
                np.save(f, text_embedding_np)
        text_embedding = np.load(os.path.join(self.text_embedding_root, f"{id}.npz"))
        prompts_attribute = {"caption": prompts, "embedding": text_embedding}
        return prompts_attribute

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, index):
        data = {}
        ids = self.instance_list[index]
        for id in ids:
            prompt_path = os.path.join(self.text_embedding_root, f"{id}.npz")
            text_embedding = np.load(prompt_path)
            data["data_pm"] = text_embedding
        return data
