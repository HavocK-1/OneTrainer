import os
import traceback

from modules.model.AnimaModel import AnimaModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert_util import convert, reverse_conversion
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
)
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast

import accelerate
from safetensors.torch import load_file


class AnimaModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name, quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: AnimaModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        t5_tokenizer = T5TokenizerFast.from_pretrained(
            base_model_name,
            subfolder="t5_tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            Qwen3Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        # conditioner is always bfloat16 — small adapter, no user dtype control
        text_conditioner = AnimaTextConditioner.from_pretrained(
            base_model_name,
            subfolder="text_conditioner",
            torch_dtype=torch.bfloat16,
        )

        if vae_model_name: #TODO simplify
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        if transformer_model_name:
            transformer = self.__load_transformer_single_file(
                model, weight_dtypes, base_model_name, transformer_model_name, text_conditioner,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                CosmosTransformer3DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.t5_tokenizer = t5_tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.text_conditioner = text_conditioner
        model.vae = vae
        model.transformer = transformer

    def __load_transformer_single_file(
            self,
            model: AnimaModel,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            text_conditioner: AnimaTextConditioner,
    ):
        #avoid loading the transformer in float32:
        torch_dtype = weight_dtypes.transformer.torch_dtype()
        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        if weight_dtypes.transformer.is_gguf():
            # GGUF is loaded by diffusers, which applies the quantization config itself
            return CosmosTransformer3DModel.from_single_file(
                transformer_model_name,
                config=base_model_name,
                subfolder="transformer",
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            )

        transformer_model_name = self._resolve_single_file_model(transformer_model_name)
        state_dict = load_file(transformer_model_name)

        # Original Anima checkpoints nest the DiT under a wrapper prefix ("model.diffusion_model." in
        # Anima trainer exports, "net." in OneTrainer's own ORIGINAL_TRANSFORMER output) and bundle the
        # text conditioner as llm_adapter.*. diffusers' cosmos conversion does not strip the wrapper, so
        # original-format files are converted manually here.
        wrapper_prefix = next(
            (prefix for prefix in ("model.diffusion_model.", "net.") if any(key.startswith(prefix) for key in state_dict)),
            None,
        )
        if wrapper_prefix is None:
            # already in diffusers format
            return CosmosTransformer3DModel.from_single_file(
                transformer_model_name,
                config=base_model_name,
                subfolder="transformer",
                torch_dtype=torch_dtype,
            )

        state_dict = {key.removeprefix(wrapper_prefix): value for key, value in state_dict.items()}

        # the bundled text conditioner keys are identical to the diffusers module, only the llm_adapter. prefix differs
        conditioner_state_dict = {}
        for key in [key for key in state_dict if key.startswith("llm_adapter.")]:
            conditioner_state_dict[key.removeprefix("llm_adapter.")] = state_dict.pop(key)
        if conditioner_state_dict:
            missing, unexpected = text_conditioner.load_state_dict(conditioner_state_dict, strict=False)
            if missing:
                raise Exception(f"could not load text conditioner from {transformer_model_name}: missing keys: {missing}")
            if unexpected:
                print(f"unexpected keys when loading text conditioner from {transformer_model_name}: {unexpected}")

        state_dict = convert(state_dict, reverse_conversion(model.diffusers_to_original()), strict=False)

        for key, value in state_dict.items():
            if value.is_floating_point() and value.dtype != torch_dtype:
                state_dict[key] = value.to(dtype=torch_dtype)

        config = CosmosTransformer3DModel.load_config(base_model_name, subfolder="transformer")
        with accelerate.init_empty_weights():
            transformer = CosmosTransformer3DModel.from_config(config)

        missing, unexpected = transformer.load_state_dict(state_dict, strict=False, assign=True)
        if missing:
            raise Exception(f"could not load transformer from {transformer_model_name}: missing keys: {missing}")
        if unexpected:
            print(f"unexpected keys when loading transformer from {transformer_model_name}: {unexpected}")

        return transformer

    def load( #TODO share code between models
            self,
            model: AnimaModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)


class AnimaLoRALoader(
    LoRALoaderMixin,
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: AnimaModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


AnimaLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.ANIMA: "resources/sd_model_spec/anima-lora.json"},
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
    lora_loader_class=AnimaLoRALoader,
)

AnimaFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.ANIMA: "resources/sd_model_spec/anima.json"},
    model_class=AnimaModel,
    model_loader_class=AnimaModelLoader,
    embedding_loader_class=None,
)
