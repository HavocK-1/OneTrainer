import os
import traceback

from modules.model.moss.pipeline_moss_image import MOSSImageScheduler, MOSSImageTransformer
from modules.model.MossModel import MossModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert_util import convert, reverse_conversion
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import AutoencoderKLFlux2
from transformers import Qwen2Tokenizer, Qwen3_5Model

import accelerate
import torch
from safetensors.torch import load_file


class MossModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: MossModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name,
                quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: MossModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        model.model_type = model_type
        model.tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        model.noise_scheduler = MOSSImageScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        model.text_encoder = self._load_transformers_sub_module(
            Qwen3_5Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        if vae_model_name:
            model.vae = self._load_diffusers_sub_module(
                AutoencoderKLFlux2,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            model.vae = self._load_diffusers_sub_module(
                AutoencoderKLFlux2,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        if transformer_model_name:
            transformer = self.__load_transformer_single_file(
                model, weight_dtypes, base_model_name, transformer_model_name,
            )
            model.transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            model.transformer = self._load_diffusers_sub_module(
                MOSSImageTransformer,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

    def __load_transformer_single_file(
            self,
            model: MossModel,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
    ):
        # single-file transformer override (e.g. a locally fine-tuned safetensors); the text
        # encoder, VAE, scheduler and tokenizer still come from the diffusers base model.
        if transformer_model_name.endswith(".gguf"):
            raise Exception("GGUF transformer overrides are not supported for MOSS, use a safetensors file")

        transformer_model_name = self._resolve_single_file_model(transformer_model_name)

        config = MOSSImageTransformer.load_config(base_model_name, subfolder="transformer")
        with accelerate.init_empty_weights():
            transformer = MOSSImageTransformer.from_config(config)

        state_dict = load_file(transformer_model_name)
        # ComfyUI-style checkpoints carry a "model.diffusion_model." prefix
        if any(k.startswith("model.diffusion_model.") for k in state_dict):
            state_dict = {k.removeprefix("model.diffusion_model."): v for k, v in state_dict.items()}
        # native/Comfy checkpoints fuse proj.q/k/v into proj.qkv; diffusers-named files keep them split
        if any(".proj.qkv" in k for k in state_dict):
            state_dict = convert(state_dict, reverse_conversion(model.checkpoint_diffusers_to_original()), strict=False)

        # avoid loading the transformer in float32:
        torch_dtype = weight_dtypes.transformer.torch_dtype()
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        for key, value in state_dict.items():
            if value.is_floating_point() and value.dtype != torch_dtype:
                state_dict[key] = value.to(dtype=torch_dtype)

        missing, unexpected = transformer.load_state_dict(state_dict, strict=False, assign=True)
        # missing buffers are fine: derived constants (e.g. rope.bands) are recomputed on load
        param_keys = {k for k, _ in transformer.named_parameters()}
        missing = [k for k in missing if k in param_keys]
        if missing:
            raise Exception(f"could not load transformer from {transformer_model_name}: missing keys: {missing}")
        if unexpected:
            print(f"unexpected keys when loading transformer from {transformer_model_name}: {unexpected}")

        return transformer

    def __load_safetensors(
            self,
            model: MossModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        raise NotImplementedError("Loading of single file MOSS models not supported. Use the diffusers model instead. Optionally, a transformer-only safetensors file can be loaded by overriding the transformer.")

    def load(
            self,
            model: MossModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, quantization,
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
