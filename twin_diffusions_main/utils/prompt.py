# 导入torch库
import torch
# 导入torch.nn.functional库
import torch.nn.functional as F

# 导入transformers库中的AutoTokenizer和CLIPTextModel类
from transformers import AutoTokenizer, CLIPTextModel
# 导入diffusers库中的DDIMScheduler、DDPMScheduler和StableDiffusionPipeline类
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline

# 导入utils.typing模块中的所有内容
from utils.typing import *
# 导入utils.ops模块中的perpendicular_component函数
from utils.ops import perpendicular_component
# 导入utils.misc模块中的C类
from utils.misc import C
# 导入.prompt_processors模块中的BasePromptProcessor和PromptEmbedding类
from .prompt_processors import BasePromptProcessor

class BasePromptProcessor(nn.Module):
    # 定义初始化方法，接受两个参数：cfg和guidance_model
    #完成了对DeepFloydPromptProcessor类的实例进行初始化设置，包括加载配置、创建方向、准备提示和加载嵌入等
    def __init__(self, cfg, guidance_model=None):
        # 调用父类的初始化方法
        super().__init__()
        # 将cfg赋值给self.cfg
        self.cfg = cfg
        # 将cfg的device属性赋值给self.device
        self.device = self.cfg.device
        # 将cfg的pretrained_model_name_or_path属性赋值给self.pretrained_model_name_or_path
        self.pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        # 将cfg的prompt属性赋值给self.prompt
        self.prompt = cfg.prompt
        # 将cfg的negative_prompt属性赋值给self.negative_prompt
        self.negative_prompt = cfg.negative_prompt
        # 将guidance_model赋值给self.guidance_model
        self.guidance_model = guidance_model

        # 将cfg的use_cache属性赋值给self.use_cache
        self.use_cache = cfg.use_cache
        # 如果cfg的use_cache属性为True
        if cfg.use_cache:
            # 将"./.cache/text_prompt_embeddings"赋值给self.cache_dir
            self.cache_dir = "./.cache/text_prompt_embeddings"
            # 创建self.cache_dir指定的目录，如果目录已存在，则不会抛出异常
            os.makedirs(self.cache_dir, exist_ok=True)

        # 定义一个名为self.directions的属性，类型为List[DirectionConfig]
        self.directions: List[DirectionConfig]
        # 如果cfg的view_dependent_prompt_front属性为True
        if cfg.view_dependent_prompt_front:
            # 定义四个方向：side、front、back和overhead，并将它们添加到self.directions列表中
            self.directions = [
                # 定义side方向
                DirectionConfig(
                    "side",  # 名称
                    lambda s: f"side view of {s}",  # 描述生成器
                    lambda s: s,  # 负面提示生成器
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),  # 视图选择器
                ),
                # 定义front方向
                DirectionConfig(
                    "front",  # 名称
                    lambda s: f"front view of {s}",  # 描述生成器
                    lambda s: s,  # 负面提示生成器
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),  # 视图选择器
                ),
                # 定义back方向
                DirectionConfig(
                    "back",  # 名称
                    lambda s: f"backside view of {s}",  # 描述生成器
                    lambda s: s,  # 负面提示生成器
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),  # 视图选择器
                ),
                # 定义overhead方向
                DirectionConfig(
                    "overhead",  # 名称
                    lambda s: f"overhead view of {s}",  # 描述生成器
                    lambda s: s,  # 负面提示生成器
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,  # 视图选择器
                ),
            ]
        # 如果cfg的view_dependent_prompt_front属性为False
        else:
            # 定义四个方向：side、front、back和overhead，并将它们添加到self.directions列表中
            self.directions = [
                # 定义side方向
                DirectionConfig(
                    "side",  # 名称
                    lambda s: f"{s}, side view",  # 描述生成器
                    lambda s: s,  # 负面提示生成器
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),  # 视图选择器
                ),
                # 定义front方向
                DirectionConfig(
                    "front",  # 名称
                    lambda s: f"{s}, front view",  # 描述生成器
                    lambda s: s,  # 负面提示生成器
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),  # 视图选择器
                ),
                # 定义back方向
                DirectionConfig(
                    "back",  # 名称
                    lambda s: f"{s}, back view",  # 描述生成器
                    lambda s: s,  # 负面提示生成器
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),  # 视图选择器
                ),
                # 定义overhead方向
                DirectionConfig(
                    "overhead",  # 名称
                    lambda s: f"{s}, overhead view",  # 描述生成器
                    lambda s: s,  # 负面提示生成器
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,  # 视图选择器
                ),
            ]

        # 使用字典推导式创建一个字典，键为方向的名称，值为方向在self.directions列表中的索引
        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}

        # 如果cfg的use_prompt_debiasing属性为True
        if cfg.use_prompt_debiasing:
            # TODO: 添加提示去偏
            # 断言self.cfg的prompt_side、prompt_back和prompt_overhead属性都为None
            # 如果使用提示去偏，不应手动分配prompt_side、prompt_back或prompt_overhead
            assert (
                self.cfg.prompt_side is None
                and self.cfg.prompt_back is None
                and self.cfg.prompt_overhead is None
            ), "Do not manually assign prompt_side, prompt_back or prompt_overhead when using prompt debiasing"
            # 获取去偏后的提示
            prompts = self.get_debiased_prompt(self.prompt)
            # 使用列表推导式创建一个列表，包含每个方向的提示
            self.prompts_view_dependent = [
                d.prompt(prompt) for d, prompt in zip(self.directions, prompts)
            ]
        # 如果cfg的use_prompt_debiasing属性为False
        else:
            # 使用列表推导式创建一个列表，包含每个方向的提示
            self.prompts_view_dependent = [
                d.prompt(self.cfg.get(f"prompt_{d.name}", None) or self.prompt)  # type: ignore
                for d in self.directions
            ]

        # 使用列表推导式创建一个列表，包含每个方向的提示和名称，然后将列表中的元素用换行符连接起来，打印结果
        prompts_vd_display = "\n".join(
            [
                f"[{d.name}]:[{prompt}]"
                for prompt, d in zip(self.prompts_view_dependent, self.directions)
            ]
        )
        print(prompts_vd_display)
        # console.print(prompts_vd_display)

        # 使用列表推导式创建一个列表，包含每个方向的负面提示
        self.negative_prompts_view_dependent = [
            d.negative_prompt(self.negative_prompt) for d in self.directions
        ]

        # 准备提示
        self.prepare_prompts()
        # 加载提示嵌入
        self.load_prompt_embeddings()

    # 定义一个名为load_from_cache的方法，接受一个参数：prompt   从缓存中加载提示的嵌入。
    def load_from_cache(self, prompt):
        # 定义缓存路径，包括缓存目录和提示的哈希值
        cache_path = os.path.join(
            self.cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )
        # 如果缓存路径不存在
        if not os.path.exists(cache_path):
            # 抛出一个FileNotFoundError异常
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )

        # 使用torch.load方法从缓存路径加载数据，并返回结果
        return torch.load(cache_path, map_location=self.device)

    # 定义一个名为prepare_text_encoder的方法，没有实现，需要子类实现
    def prepare_text_encoder(self):
        raise NotImplementedError

    # 定义一个名为encode_prompts的方法，接受一个参数：prompts，没有实现，需要子类实现
    def encode_prompts(self, prompts):
        raise NotImplementedError

    # 定义一个名为load_prompt_embeddings的方法
    def load_prompt_embeddings(self):
        # 从缓存中加载提示的嵌入，并赋值给self.text_embedding
        self.text_embedding = self.load_from_cache(self.prompt)[None, ...]
        # 从缓存中加载负面提示的嵌入，并赋值给self.uncond_text_embedding
        self.uncond_text_embedding = self.load_from_cache(self.negative_prompt)[
            None, ...
        ]
        # 从缓存中加载视图依赖的提示的嵌入，并赋值给self.text_embedding_view_dependent
        self.text_embedding_view_dependent = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.prompts_view_dependent],
            dim=0,
        )
        # 从缓存中加载视图依赖的负面提示的嵌入，并赋值给self.uncond_text_embedding_view_dependent
        self.uncond_text_embedding_view_dependent = torch.stack(
            [
                self.load_from_cache(prompt)
                for prompt in self.negative_prompts_view_dependent
            ],
            dim=0,
        )

    # 定义一个名为prepare_prompts的方法
    def prepare_prompts(self):
        # 准备文本编码器，如果self.guidance_model为None，表示从unet和vae独立初始化文本编码器和分词器
        self.prepare_text_encoder(self.guidance_model)
        # 创建一个列表，包含提示、负面提示、视图依赖的提示和视图依赖的负面提示
        prompts = (
            [
                self.prompt,
                self.negative_prompt,
            ]
            + self.prompts_view_dependent
            + self.negative_prompts_view_dependent
        )
        # 创建一个空列表，用于存储需要处理的提示
        prompts_to_process = []
        # 遍历prompts列表
        for prompt in prompts:
            # 如果使用缓存
            if self.use_cache:
                # 定义缓存路径，包括缓存目录和提示的哈希值
                cache_path = os.path.join(
                    self.cache_dir,
                    f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
                )
                # 如果缓存路径存在，跳过当前循环
                if os.path.exists(cache_path):
                    continue
            # 将提示添加到prompts_to_process列表
            prompts_to_process.append(prompt)

        # 如果prompts_to_process列表的长度大于0
        if len(prompts_to_process) > 0:
            # 编码提示，得到提示的嵌入
            prompt_embeddings = self.encode_prompts(prompts_to_process)

            # 遍历prompts_to_process列表和prompt_embeddings列表
            for prompt, embedding in zip(prompts_to_process, prompt_embeddings):
                # 如果使用缓存
                if self.use_cache:
                    # 定义缓存路径，包括缓存目录和提示的哈希值
                    cache_path = os.path.join(
                        self.cache_dir,
                        f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
                    )
                    # 使用torch.save方法保存嵌入到缓存路径
                    torch.save(embedding, cache_path)

    # 定义一个名为get_prompt_embedding的方法，返回一个PromptEmbedding对象
    def get_prompt_embedding(self) -> PromptEmbedding:
        return PromptEmbedding(
            text_embedding=self.text_embedding,
            uncond_text_embedding=self.uncond_text_embedding,
            text_embedding_view_dependent=self.text_embedding_view_dependent,
            uncond_text_embedding_view_dependent=self.uncond_text_embedding_view_dependent,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_negative=self.cfg.use_perp_negative,
            debug=self.cfg.debug,
        )

    # 定义一个名为get_debiased_prompt的方法，接受一个参数：prompt
    #这个方法用于获取去偏后的提示。它首先使用预训练的Bert模型对视角方向的描述进行预测，然后计算每个描述词在不同视角下的概率分布。
    #接着根据配置文件中指定的需要去偏的单词列表，将每个单词从提示中移除，并再次预测各个方向下的概率分布。
    #如果某个单词的去偏前后概率分布的相对变化小于0.95，说明这个单词对于该方向的影响较小，可以考虑去除。最终，返回去偏后的提示列表。
    def get_debiased_prompt(self, prompt):
        # 设置环境变量TOKENIZERS_PARALLELISM为false
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # 从预训练模型中加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path_prompt_debiasing
        )
        # 从预训练模型中加载BertForMaskedLM模型
        model = BertForMaskedLM.from_pretrained(
            self.cfg.pretrained_model_name_or_path_prompt_debiasing
        )

        # 获取方向的名称列表
        views = [d.name for d in self.directions]
        # 对方向的名称列表进行分词，并获取输入id
        view_ids = tokenizer(" ".join(views), return_tensors="pt").input_ids[0]
        # 获取输入id的第1到第5个元素
        view_ids = view_ids[1:5]

        # 定义一个名为modulate的函数，接受一个参数：prompt
        def modulate(prompt):
            # 定义一个模板字符串
            prompt_vd = f"This image is depicting a [MASK] view of {prompt}"
            # 对模板字符串进行分词
            tokens = tokenizer(
                prompt_vd,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            # 获取mask token的索引
            mask_idx = torch.where(tokens.input_ids == tokenizer.mask_token_id)[1]

            # 使用模型对tokens进行预测，得到logits
            logits = model(**tokens).logits
            # 对logits进行softmax操作
            logits = F.softmax(logits[0, mask_idx], dim=-1)
            # 获取logits的第0个元素和view_ids对应的元素
            logits = logits[0, view_ids]
            # 计算probes，即logits的每个元素除以logits的和
            probes = logits / logits.sum()
            # 返回probes
            return probes

        # 将prompt按空格分割，得到一个列表，重复4次，得到一个二维列表
        prompts = [prompt.split(" ") for _ in range(4)]
        # 调用modulate函数，得到full_probe
        full_probe = modulate(prompt)
        # 计算prompt的单词数量
        n_words = len(prompt.split(" "))
        # 获取需要进行去偏的单词的索引
        prompt_debiasing_mask_ids = (
            self.cfg.prompt_debiasing_mask_ids
            if self.cfg.prompt_debiasing_mask_ids is not None
            else list(range(n_words))
        )
        # 获取需要进行去偏的单词
        words_to_debias = [prompt.split(" ")[idx] for idx in prompt_debiasing_mask_ids]
        # 打印需要进行去偏的单词
        console.print(f"Words that can potentially be removed: {words_to_debias}")
        # 遍历需要进行去偏的单词的索引
        for idx in prompt_debiasing_mask_ids:
            # 将prompt按空格分割，得到一个列表
            words = prompt.split(" ")
            # 将列表的前idx个元素和后(idx+1)个元素连接，得到一个新的字符串
            prompt_ = " ".join(words[:idx] + words[(idx + 1) :])
            # 调用modulate函数，得到part_probe
            part_probe = modulate(prompt_)

            # 计算pmi，即full_probe除以part_probe和full_probe的插值
            pmi = full_probe / torch.lerp(part_probe, full_probe, 0.5)
            # 遍历pmi的每个元素
            for i in range(pmi.shape[0]):
                # 如果pmi的第i个元素小于0.95
                if pmi[i].item() < 0.95:
                    # 将prompts的第i个元素的第idx个元素设置为空字符串
                    prompts[i][idx] = ""

        # 将prompts的每个元素按空格连接，得到一个新的列表
        debiased_prompts = [" ".join([word for word in p if word]) for p in prompts]
        # 遍历views和debiased_prompts的每个元素
        for d, debiased_prompt in zip(views, debiased_prompts):
            # 打印去偏后的提示
            console.print(f"Debiased prompt of the {d} view is [{debiased_prompt}]")

        # 删除tokenizer和model
        del tokenizer, model
        # 调用cleanup方法
        self.cleanup()
        # 调用gc.collect方法，进行垃圾回收
        gc.collect()
        # 清空CUDA缓存
        torch.cuda.empty_cache()

        # 返回去偏后的提示
        return debiased_prompts

    # 定义一个名为update的方法，没有实现，需要子类实现
    def update(self, step):
        raise NotImplementedError("Update not implemented")

    # 定义一个名为forward的方法，返回提示的嵌入
    def forward(self):
        return self.get_prompt_embedding()

    # 定义一个名为cleanup的方法，删除tokenizer和text_encoder
    def cleanup(self):
        del self.tokenizer
        del self.text_encoder

class StableDiffusionPromptProcessor(BasePromptProcessor):
    # 定义prepare_text_encoder方法，接受一个guidance_model参数，默认值为None
    def prepare_text_encoder(self, guidance_model=None):
        # 如果guidance_model为None
        if guidance_model is None:
            # 从预训练模型中加载AutoTokenizer，并赋值给self.tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="tokenizer",
                cache_dir="./.cache",
            )
            # 从预训练模型中加载CLIPTextModel，并赋值给self.text_encoder
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                device_map="auto",
                cache_dir="./.cache",
            )
        # 如果guidance_model不为None
        else:
            # 从guidance_model中获取tokenizer，并赋值给self.tokenizer
            self.tokenizer = guidance_model.pipe.tokenizer
            # 从guidance_model中获取text_encoder，并赋值给self.text_encoder
            self.text_encoder = guidance_model.pipe.text_encoder

    # 定义encode_prompts方法，接受一个prompts参数
    def encode_prompts(self, prompts):
        # 禁止计算梯度
        with torch.no_grad():
            # 打印prompts
            print(prompts)
            # 使用tokenizer对prompts进行编码，并赋值给tokens
            tokens = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).to(self.device)
            # 使用text_encoder对tokens.input_ids进行编码，并赋值给text_embeddings
            text_embeddings = self.text_encoder(tokens.input_ids)[0]

        # 返回text_embeddings
        return text_embeddings

    # 定义update方法，接受一个step参数
    def update(self, step):
        # 方法体为空，不执行任何操作
        pass

