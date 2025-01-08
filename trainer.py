import os
import logging
from venv import logger
import transformers
import torch
import torch.nn as nn
import diffusers
import math
import accelerate

from termcolor import colored
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from packaging import version

from data.twindiffusions_dataset import TwinDiffusionDataset
from twin_diffusions_main.twin_diffusions import TwinDiffusions
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import (
    ProjectConfiguration,
    InitProcessGroupKwargs,
    DistributedDataParallelKwargs,
    set_seed,
)
from accelerate.logging import get_logger
from huggingface_hub import create_repo
from diffusers.optimization import get_scheduler


class Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.step = 0
        self.start = 0
        self.mode = cfg.get("mode", "text_to_3d")
        self.model = cfg.get("model", "stable diffusion v1.5")
        self.root = cfg.model.get("root", None)
        self.output_dir = cfg.output_dir
        self.logging_dir = cfg.logging_dir
        self.distributed_data_parallel = cfg.get("distributed_data_parallel", False)
        self.gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
        self.mixed_precision = cfg.get("mixed_precision", None)
        self.report_to = cfg.get("report_to", None)
        self.seed = cfg.get("seed", int(42))
        self.push_to_hub = cfg.get("push_to_hub", False)
        self.hub_model_id = cfg.get("hub_model_id", None)
        self.hub_token = cfg.get("hub_token", None)
        self.num_epochs = cfg.get("num_epochs", int(1))
        self.lr_warmup = cfg.get("lr_warmup", int(500))
        self.lr_cycle = cfg.get("lr_cycle", int(1))
        self.negative_prompt = cfg.get("negative_prompt", None)
        self.refer_img_file = cfg.get("refer_img_file", None)
        self.rebulid = cfg.model.get("rebulid", False)
        self.twin_diffusions_path = cfg.model.get("twin_diffusions_path", None)
        self.device = cfg.get("device", "cuda")
        self.graddient_checkpointing = cfg.get("graddient_checkpointing", False)
        self.learning_rate = cfg.get("learning_rate", 5e-6)
        self.adam_weight_decay = cfg.get("adam_weight_decay", 1e-2)
        self.lr_schedule = cfg.get("lr_schedule", "constant")
        self.lr_warmup = cfg.get("lr_warmup", 500)
        self.tracker_project_name = cfg.get("tracker_project_name", "train_lift3d")
        self.resume_from_checkpoint = cfg.get("resume_from_checkpoint", None)
        self.max_norm = cfg.get("max_norm", 1.0)

        logger = get_logger(__name__)
        logging_dir = Path(self.output_dir, self.logging_dir)

        accelerator_project_config = ProjectConfiguration(
            project_dir=self.output_dir,
            logging_dir=logging_dir,
        )
        accelerator_init_process_config = InitProcessGroupKwargs(
            timeout=timedelta(seconds=7200)
        )
        distributed_data_parallel_config = DistributedDataParallelKwargs(
            find_unused_parameters=True
        )

        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with=self.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[
                accelerator_init_process_config,
                distributed_data_parallel_config,
            ],
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )

        logger.info(self.accelerator.state, main_process_only=False)

        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self.accelerator.is_main_process:
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)

            if self.cfg.push_to_hub:
                self.repo_id = create_repo(
                    repo_id=self.hub_model_id or Path(self.output_dir).name,
                    exist_ok=True,
                    token=self.hub_token,
                ).repo_id

        self.weight_dtype = cfg.get("weight_dtype", torch.float32)  #!检查acceralator
        if self.accelerator.mixed_precision == "float16":
            self.weight_dtype = torch.float16

        self.dataset = TwinDiffusionDataset(self.cfg, self.weight_dtype)

        self.batch_size = cfg.get("batch_size", 1)
        self.num_workers = cfg.get("num_workers", 0)
        self.dataloader = iter(
            DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
        )

        if self.max_steps is None:
            self.max_steps, overrode_max_steps = self.overrode_max_steps(cfg)
            print(colored(f"Overrode max_steps: {overrode_max_steps}", "red"))
        print(colored(f"Steps: {self.max_steps}", "yellow"))

        if self.rebulid or self.twin_diffusions_path is None:
            print(colored("Start building TwinDiffusions", "yellow"))
            self.twindiffusions = TwinDiffusions(self.cfg, self.max_steps, self.dataset)
        else:
            print(colored("Loading TwinDiffusions weights", "yellow"))
            self.twindiffusions = torch.load(
                self.twin_diffusions_path,
                low_cpu_mem_usage=False,
                local_files_only=True,
            )
            print(colored("TwinDiffusions is ready.", "yellow"))

        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

            def save_model_hook(models, weights, output_dir):
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    sub_dir = "twin_diffusions"
                    model.save_pretrained(Path(output_dir, sub_dir))
                    i -= 1

            def load_model_hook(models, output_dir):
                while len(models) > 0:
                    model = models.pop()
                    load_model = TwinDiffusions.from_pretrained(
                        Path(output_dir, "twin_diffusions")
                    )
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model

            self.accelerator.register_save_state_pre_hook(save_model_hook)
            self.accelerator.register_load_state_pre_hook(load_model_hook)

        if self.graddient_checkpointing:
            self.twindiffusions.enable_gradient_checkpointing()

        if (
            self.accelerator.unwrap_model(self.twindiffusions).dtype
            != self.weight_dtype
        ):
            self.twindiffusions = self.twindiffusions.to(self.weight_dtype, self.device)

        # TODO: optimizer , param_groups
        # TODO:lr_scheduler = self.dataset.schedule(self.max_steps, self.lr_warmup, self.lr_cycle)

        # initial_values 初始化的点云point_e

        params = []
        for name, param in self.twindiffusions.named_parameters():
            if name.split(".")[0] in ["unet"]:
                continue
            else:
                params.append(param)
        params_to_optimize = []
        if len(params) > 0:
            params_to_optimize.append({"params": params, "lr": self.learning_rate})

        optimizer_class = torch.optim.Adam
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.learning_rate,
            weight_decay=self.adam_weight_decay,
        )

        self.lr_scheduler = get_scheduler(
            self.lr_schedule,
            optimizer=self.optimizer,
            num_training_steps=self.max_steps * self.gradient_accumulation_steps,
            num_warmup_steps=self.lr_warmup * self.gradient_accumulation_steps,
        )

        self.twindiffusions, self.optimizer, self.lr_scheduler, self.dataloader = (
            self.accelerator.prepare(
                self.twindiffusions, self.optimizer, self.lr_scheduler, self.dataloader
            )
        )

        self.num_epochs = math.ceil(
            self.max_steps * self.gradient_accumulation_steps / len(self.dataloader)
        )
        total_batch_size = (
            self.batch_size
            * self.accelerator.num_processes
            * self.graddient_accumulation_steps
        )

        if self.accelerator.is_main_process:
            tracker_config = dict(vars(cfg))
            self.accelerator.init_trackers(
                self.tracker_project_name, config=tracker_config
            )

        logger.info(f"Num examples: {len(self.dataset)}")
        logger.info(f"Num workers: {self.num_workers}")
        logger.info(f"Num batches each epoch: {len(self.dataloader)}")
        logger.info(f"Total batch size: {total_batch_size}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Total steps: {self.max_steps}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

    def train_loop(self):
        global_step = 0
        first_epoch = 0

        if self.resume_from_checkpoint is not None:
            if self.resume_from_checkpoint == "latest":
                dirs = os.listdir(self.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
            else:
                path = os.path.join(self.resume_from_checkpoint)

            if path is not None:
                self.accelerator, print(
                    colored(f"Resuming from checkpoint: {path}", "yellow")
                )
                self.accelerator.load_state(os.path.join(self.output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // (
                    math.ceil(len(self.dataloader) / self.gradient_accumulation_steps)
                )
            else:
                self.accelerator.print(
                    colored("No checkpoint found. Starting a new training loop", "red")
                )
                self.resume_from_checkpoint = None
                initial_global_step = 0
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, self.max_steps),
            initial=initial_global_step,
            desc="steps",
            disable=not self.accelerator.is_local_main_process,
        )
        for epoch in range(first_epoch, self.num_epochs):
            self.train()
            logger.info(colored("Training start.", "yellow"))
            for step, batch in enumerate(self.dataloader):
                with self.accelerator.accumulate(self.twindiffusions):
                    self.optimizer.zero_grad()
                    output = self.twindiffusions(batch, step)
                    losses = output["losses"]  # TODO:losses
                    self.accelerator.backward(losses)
                    if self.accelerator.sync_gradients:
                        params_to_clip = self.twindiffusions.parameters()
                        self.accelerator.clip_grad_norm_(
                            params_to_clip, max_norm=self.max_norm
                        )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        if self.accelerator.is_main_process:
                            if global_step % 100 == 0:
                                save_path = os.path.join(
                                    self.output_dir, f"checkpoint-{global_step}"
                                )
                                self.accelerator.save_state(save_path)
                                logger.info(
                                    colored(
                                        f"Saved checkpoint at step {global_step}",
                                        "yellow",
                                    )
                                )
                            # TODO:validation

                    logs = {
                        "loss": losses.detach().item(),
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }
                    logs.update(losses)
                    progress_bar.set_postfix(**logs)
                    self.accelerator.log(logs, step=global_step)

                    if global_step >= self.max_steps:
                        break

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self.twin_diffusions = self.accelerator.unwrap_model(self.twindiffusions)
            self.twin_diffusions.save_pretrained(self.output_dir)
            if self.push_to_hub:
                raise NotImplementedError("Push to hub is not implemented yet.")

        self.accelerator.end_training()

    def update(self, step):
        self.dataset.update(step)
        # self.renderer.update(step)
        # self.guidance.update(step)
        # self.optimizer.update(step)

    def overrode_max_steps(self, overrode_max_steps=False):
        num_update_steps_epoch = math.ceil(
            len(self.dataloader) / self.gradient_accumulation_steps
        )
        if self.max_steps is None:
            self.max_steps = self.num_epochs * num_update_steps_epoch
            overrode_max_steps = True
        return self.max_steps, overrode_max_steps
