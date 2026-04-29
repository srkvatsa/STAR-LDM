import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
from omegaconf import OmegaConf

from star_ldm.models.transfusion import TransfusionGPT, variance_preserving_map
from star_ldm.diffusion.noise_schedule import log_snr_to_alpha2

class TransfusionGPTInterface:
    def __init__(self, model_path: str, device: str = 'cuda', classifier_path: Optional[str] = None,
                 target_model_name: Optional[str] = None, draft_model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to the STAR-LDM checkpoint directory or ``.pt`` file.
            device: Device to load models onto.
            classifier_path: Optional path to a pretrained
                :class:`~star_ldm.models.classifier.NoiseConditionedMLP` checkpoint
                for classifier-guided generation.
            target_model_name: Optional HuggingFace model name for the speculative
                decoding target model (e.g., ``'gpt2-large'``). If provided, enables
                speculative decoding with the backbone as draft model.
            draft_model_path: Optional path to a trained
                :class:`~star_ldm.decoding.DraftTransformerLM` checkpoint for
                distilled speculative decoding. When provided, uses the tiny draft
                model (~20M params) instead of a separate HF target model. The
                backbone GPT-2 becomes the target, and the draft generates candidates.
        """
        self.model_path = model_path
        if device == 'cuda' and not torch.cuda.is_available():
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.gpt2.config._name_or_path)

        self.classifier = None
        if classifier_path is not None:
            from star_ldm.models.classifier import load_classifier
            self.classifier = load_classifier(classifier_path, device=str(self.device))

        # Speculative decoding: legacy HF target model path
        self.target_model = None
        if target_model_name is not None:
            self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name)
            self.target_model = self.target_model.to(self.device)
            self.target_model.eval()

        # Speculative decoding: distilled draft model path (preferred)
        self.draft_model = None
        if draft_model_path is not None:
            from star_ldm.decoding.draft_model import load_draft_model
            self.draft_model = load_draft_model(draft_model_path, device=str(self.device))

    def _load_model(self, model_path: str) -> 'TransfusionGPT':
        # Check if model_path ends in '.pt'
        if model_path.endswith('.pt'):
            model_dir = os.path.dirname(model_path)
        else:
            model_dir = model_path
            model_path = os.path.join(model_dir, 'model.pt')

        # Grab model directory from model_path
        transfusion_cfg = OmegaConf.load(os.path.join(model_dir, 'args.yaml'))

        model = TransfusionGPT(
            dataset_name=transfusion_cfg.dataset_name,
            transfusion_cfg=transfusion_cfg,
            gpt2_model_name=transfusion_cfg.train.lm_name,
            gamma_min=-15,
            gamma_max=15,
            clf_guidance_dropout=0.1,
            scale_by_std=True,
            global_norm=transfusion_cfg.train.get('global_norm', False),
        )

        # Always load to CPU first to avoid MPS unaligned blit errors
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

        if isinstance(ckpt, dict) and 'ema' in ckpt:
            # Direct training checkpoint — extract EMA weights
            from ema_pytorch import EMA
            ema = EMA(model, beta=0.999, update_every=10, power=3/4, update_after_step=1000)
            ema.load_state_dict(ckpt['ema'], strict=False)
            model = ema.ema_model
        else:
            # Plain state_dict
            state_dict = ckpt
            model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)

        return model

    def generate(self, prompts: List[str], cls_guidance: float = 0.0,
                 cls_target: Optional[float] = None, use_kv_cache: bool = True,
                 use_speculative: bool = False, speculative_k: int = 4,
                 **kwargs) -> List[str]:
        """
        Generate text for a list of prompts.

        Args:
            prompts: List of prompts to generate from.
            cls_guidance: Classifier guidance scale. ``0.0`` disables guidance.
                Positive values steer toward ``cls_target``.
            cls_target: Target class for classifier guidance (``0.0`` or ``1.0``).
                Required when ``cls_guidance != 0``.
            use_kv_cache: If True (default), use KV-cache reuse for faster
                inference. The prefix KV is computed once and reused across all
                diffusion steps.
            use_speculative: If True, use speculative decoding for the final
                text generation step. Requires ``target_model_name`` to have
                been set during initialization.
            speculative_k: Number of draft tokens per speculative round (default 4).
            **kwargs: Additional keyword arguments forwarded to
                :meth:`TransfusionGPT.sample` or
                :meth:`TransfusionGPT.sample_with_kv_cache`.

        Returns:
            List of generated text strings.
        """
        if cls_guidance != 0.0:
            if self.classifier is None:
                raise ValueError(
                    "Classifier guidance requested but no classifier loaded. "
                    "Pass classifier_path when constructing TransfusionGPTInterface."
                )
            if cls_target is None:
                raise ValueError(
                    "cls_target must be specified (0.0 or 1.0) when using classifier guidance."
                )

        if use_speculative and self.target_model is None and self.draft_model is None:
            raise ValueError(
                "Speculative decoding requested but no draft or target model loaded. "
                "Pass draft_model_path or target_model_name when constructing "
                "TransfusionGPTInterface."
            )

        sample_fn = self.model.sample_with_kv_cache if use_kv_cache else self.model.sample

        generations = []
        generate_kwargs = kwargs.pop('generate_kwargs', {})
        for prompt in tqdm(prompts, desc="Generating"):
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)

            if use_speculative:
                if self.draft_model is not None:
                    # Distilled draft model: draft=tiny model, target=backbone GPT-2
                    _, generation = self.model.sample_with_draft_speculative(
                        input_ids,
                        draft_model=self.draft_model,
                        speculative_k=speculative_k,
                        cls_guidance=cls_guidance,
                        classifier=self.classifier,
                        cls_target=cls_target,
                        generate_kwargs=generate_kwargs,
                        **kwargs,
                    )
                else:
                    # Legacy: backbone=draft, separate HF model=target
                    _, generation = self.model.sample_with_speculative(
                        input_ids,
                        target_model=self.target_model,
                        speculative_k=speculative_k,
                        cls_guidance=cls_guidance,
                        classifier=self.classifier,
                        cls_target=cls_target,
                        generate_kwargs=generate_kwargs,
                        **kwargs,
                    )
            else:
                _, generation = sample_fn(
                    input_ids,
                    cls_guidance=cls_guidance,
                    classifier=self.classifier,
                    cls_target=cls_target,
                    generate_kwargs=generate_kwargs,
                    **kwargs,
                )
            generations.extend(generation)
        return generations

    def interactive_demo(self, generate_kwargs: Optional[Dict[str, Any]] = None):
        """
        Run an interactive demo allowing the user to try different generation settings.
        """
        print("STAR-LDM Interactive Demo")
        print("Enter 'quit' to exit")

        while True:
            prompt = input("\nEnter a prompt: ")
            if prompt.lower() == 'quit':
                break

            if generate_kwargs is None:
                generation = self.generate([prompt])[0]
            else:
                generation = self.generate([prompt], **generate_kwargs)[0]

            print(f"Generated text: {generation}")
