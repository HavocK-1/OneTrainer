import os
import traceback

from modules.model.Krea2Model import Krea2Model
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert_util import convert, reverse_conversion
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Transformer2DModel,
)
from transformers import Qwen2Tokenizer, Qwen3VLModel

import accelerate
from safetensors.torch import load_file


class Krea2ModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: Krea2Model,
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
            model: Krea2Model,
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

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            Qwen3VLModel,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
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
                model, weight_dtypes, base_model_name, transformer_model_name,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                Krea2Transformer2DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def __load_transformer_single_file(
            self,
            model: Krea2Model,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
    ):
        # diffusers' Krea2Transformer2DModel is not registered as single-file-loadable (no
        # from_single_file), so the native Krea 2 checkpoint namespace is converted manually. The
        # mapping is the reverse of Krea2Model.checkpoint_diffusers_to_original() -- the same
        # conversion the saver applies when writing ORIGINAL_TRANSFORMER files.
        if transformer_model_name.endswith(".gguf"):
            # ponytail: GGUF single-file support stops here; add diffusers' GGUF checkpoint loading when a GGUF Krea 2 transformer is needed
            raise Exception("GGUF transformer overrides are not supported for Krea 2, use a safetensors file")

        transformer_model_name = self.__resolve_transformer_file(transformer_model_name)

        config = Krea2Transformer2DModel.load_config(base_model_name, subfolder="transformer")
        with accelerate.init_empty_weights():
            transformer = Krea2Transformer2DModel.from_config(config)

        state_dict = load_file(transformer_model_name)
        # ComfyUI-style checkpoints carry a "model.diffusion_model." prefix
        if any(k.startswith("model.diffusion_model.") for k in state_dict):
            state_dict = {k.removeprefix("model.diffusion_model."): v for k, v in state_dict.items()}

        state_dict = convert(state_dict, reverse_conversion(model.checkpoint_diffusers_to_original()), strict=False)

        #avoid loading the transformer in float32:
        torch_dtype = weight_dtypes.transformer.torch_dtype()
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        for key, value in state_dict.items():
            if value.is_floating_point() and value.dtype != torch_dtype:
                state_dict[key] = value.to(dtype=torch_dtype)

        missing, unexpected = transformer.load_state_dict(state_dict, strict=False, assign=True)
        if missing:
            raise Exception(f"could not load transformer from {transformer_model_name}: missing keys: {missing}")
        if unexpected:
            print(f"unexpected keys when loading transformer from {transformer_model_name}: {unexpected}")

        return transformer

    @staticmethod
    def __resolve_transformer_file(transformer_model_name: str) -> str:
        # A bare HF repo id ("user/repo") or "repo_id:filename" may point at a single-file
        # transformer-only repo (e.g. HavocK1/See-Krea-2-Turbo, which holds just one .safetensors
        # file and no diffusers config). load_file() needs a local path, so resolve via the hub.
        if os.path.isfile(transformer_model_name):
            return transformer_model_name
        import huggingface_hub
        import re
        # "repo_id:filename" split; a Windows drive-letter path ("D:\...") also contains ":",
        # so only split when the left side looks like a HF repo id ("user/repo").
        match = re.match(r"^([\w.\-]+/[\w.\-]+):(.+)$", transformer_model_name)
        if match:
            repo_id, filename = match.group(1), match.group(2)
            return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
        repo_id = transformer_model_name
        siblings = huggingface_hub.list_repo_files(repo_id)
        candidates = [f for f in siblings if f.endswith(".safetensors")]
        if len(candidates) == 1:
            return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=candidates[0])
        raise Exception(
            f"could not resolve transformer file in {repo_id}: expected exactly one .safetensors file, "
            f"found {candidates}. Use 'repo_id:filename' to pick one."
        )

    def __load_safetensors(
            self,
            model: Krea2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        raise NotImplementedError("Loading of single file Krea 2 models not supported. Use the diffusers model instead. Optionally, transformer-only safetensor files can be loaded by overriding the transformer.")

    def load( #TODO share code between models
            self,
            model: Krea2Model,
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

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
