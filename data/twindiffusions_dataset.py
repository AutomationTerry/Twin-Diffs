from transformers import AutoTokenizer, CLIPTextModel
from data.get_3d_dataset import TwinDiffusion3dDataset
from data.get_mv_dataset import TwinDiffusionMVDataset
from data.prompt.get_prompt import PromptDataset
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin


class TwinDiffusionDataset(ModelMixin, ConfigMixin):
    def __init__(self, cfg, weight_dtype):
        super().__init__()

        self.cfg = cfg
        self.model = cfg.model
        self.text_encoder = self.prepare_text_encoder(self)
        self.text_tokenizer = self.prepare_text_encoder(self)
        self.text_encoder.requires_grad_(False)

        self.weight_dtype = weight_dtype
        self.data_3d = TwinDiffusion3dDataset(self.cfg)

        self.data_mv = TwinDiffusionMVDataset(self.cfg, self.weight_dtype)
        self.data_pm = PromptDataset(self.cfg, self.text_encoder, self.text_tokenizer)

        self.pipe = TwinDiffusionMVDataset.get_pipeline(self.cfg)
        self.schedule = self.get_schedule(self.model.name)
        self.noise_schedule = self.get_schedule(self.model.name + " noise")

        self.revision = cfg.model.get("revision", None)
        # TODO:camera

    def prepare_text_encoder(self):
        if self.model.name == "stable diffusion v1.5":
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                self.model.root, subfolder="tokenizer", cache_dir="./.cache"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.model.root, subfolder="text_encoder", cache_dir="./.cache"
            ).to(self.device)
        elif self.model == "stable diffusion xl":
            self.base_text_tokenizer = AutoTokenizer.from_pretrained(
                self.model.base.root, subfolder="tokenizer_2", cache_dir="./.cache"
            )
            self.base_text_encoder = CLIPTextModel.from_pretrained(
                self.model.base.root, subfolder="text_encoder_2", cache_dir="./.cache"
            ).to(self.device)

            self.refiner_text_tokenizer = AutoTokenizer.from_pretrained(
                self.model.refiner.root, subfolder="tokenizer_2", cache_dir="./.cache"
            )
            self.refiner_text_encoder = CLIPTextModel.from_pretrained(
                self.model.refiner.root,
                subfolder="text_encoder_2",
                cache_dir="./.cache",
            ).to(self.device)
        else:
            raise NotImplementedError("Unsupported model: {}".format(self.model))

    def get_schedule(self, name):
        if name == "stable diffusion v1.5":
            self.schedule = DDIMScheduler(
                self.model.root,
                subfolder="scheduler",
                torch_dtype=self.weight_dtype,
                optimizer=self.optimizer,
                local_files_only=True,
            )
            return self.schedule
        elif name == "stable diffusion xl":
            self.base_schedyler = DDIMScheduler(
                self.model.base.root,
                subfolder="scheduler",
                torch_dtype=self.weight_dtype,
                optimizer=self.optimizer,
                local_files_only=True,
            )
            self.refiner_schedyler = DDIMScheduler(
                self.model.refiner.root,
                subfolder="scheduler",
                torch_dtype=self.weight_dtype,
                local_files_only=True,
            )
        elif name == "stable diffusion v1.5 noise":
            self.schedule = DDPMScheduler(
                self.model.root,
                subfolder="scheduler",
                torch_dtype=self.weight_dtype,
                revision=self.revision,
                local_files_only=True,
            )
        elif name == "stable diffusion xl noise":
            self.base_schedyler = DDPMScheduler(
                self.model.base.root,
                subfolder="scheduler",
                revision=self.revision,
                local_files_only=True,
            )
            self.refiner_schedyler = DDPMScheduler(
                self.model.refiner.root,
                subfolder="scheduler",
                revision=self.revision,
                local_files_only=True,
            )

    def update(self, step):
        self.data_3d.update(step)
        self.data_mv.update(step)
        self.data_pm.update(step)

    def get_data_mv(self):
        return self.data_mv

    def __len__(self):
        return len(self.data_3d)

    def __getitem__(self, index):
        data = self.data_3d.__getitem__(index)
        data.update(self.data_mv.__getitem__(index))
        data.update(self.data_pm.__getitem__(index))
        return data
