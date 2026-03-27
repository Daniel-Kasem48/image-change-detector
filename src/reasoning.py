"""
reasoning.py
------------
VLM-based (Vision Language Model) semantic change reasoning.

Unlike pixel/YOLO detection which compares visual features,
this module sends both images to a VLM and asks it to **reason**
about what actually changed — handling different angles, scales,
lighting, and camera positions that break traditional CV.

This solves the core problem of **Visual Change Captioning** /
**Cross-View Change Detection**: understanding that a cup seen from
above and from the side is the *same* cup, not a removed + added object.

Supported providers
~~~~~~~~~~~~~~~~~~~
- ``huggingface`` — HuggingFace Inference Providers (**free tier** available).
  Uses ``huggingface_hub.InferenceClient`` with vision-capable models like
  ``Qwen/Qwen2.5-VL-7B-Instruct`` or ``meta-llama/Llama-3.2-11B-Vision-Instruct``.
- ``gemini`` — Google Gemini (gemini-2.5-flash, gemini-2.5-pro, etc.)
- ``openai`` — OpenAI GPT-4o / GPT-4.1

Usage::

    from src.reasoning import VLMReasoner

    # Free HuggingFace provider (default)
    reasoner = VLMReasoner(provider="huggingface")
    result = reasoner.compare("before.jpg", "after.jpg")

    # Gemini
    reasoner = VLMReasoner(provider="gemini")
    result = reasoner.compare("before.jpg", "after.jpg")

    print(result["summary"])
    for change in result["changes"]:
        print(change)
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import ChangeDetectorError, ConfigError
from .logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Valid providers & their defaults
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "huggingface": {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "api_key_env": "HF_TOKEN",
    },
    "gemini": {
        "model": "gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
    "openai": {
        "model": "gpt-4.1-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        # Free vision-capable model on OpenRouter (OpenAI-compatible)
        "model": "google/gemini-2.5-flash-preview:free",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}

VALID_PROVIDERS = frozenset(_PROVIDER_DEFAULTS)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ReasoningError(ChangeDetectorError):
    """Raised when VLM reasoning fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReasoningConfig:
    """VLM reasoning configuration.

    Attributes:
        provider: ``'huggingface'`` (free), ``'gemini'``, ``'openai'``, or
                  ``'openrouter'`` (OpenAI-compatible, many free models).
        model: Model identifier.  If empty, a sensible default is chosen
               based on the provider.
        api_key_env: Name of the environment variable holding the API key.
        temperature: Sampling temperature (0 = deterministic).
        max_tokens: Maximum response tokens.
        hf_provider: HuggingFace inference provider backend
                     (``'auto'``, ``'novita'``, ``'together'``, etc.).
                     Only used when *provider* is ``'huggingface'``.
        fallback_providers: Comma-separated providers to try if primary fails
                            (e.g. ``'gemini,openrouter'``).
    """

    provider: str = "huggingface"
    model: str = ""
    api_key_env: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096
    hf_provider: str = "auto"
    fallback_providers: str = ""

    def __post_init__(self) -> None:
        if self.provider not in VALID_PROVIDERS:
            raise ConfigError(
                f"Invalid reasoning provider: {self.provider!r}. "
                f"Use one of {sorted(VALID_PROVIDERS)}.",
                "reasoning.provider",
            )
        if self.temperature < 0 or self.temperature > 2:
            raise ConfigError(
                f"temperature must be 0–2, got {self.temperature}",
                "reasoning.temperature",
            )

    @property
    def resolved_model(self) -> str:
        """Return the model name, falling back to the provider default."""
        return self.model or _PROVIDER_DEFAULTS[self.provider]["model"]

    @property
    def resolved_api_key_env(self) -> str:
        """Return the env-var name, falling back to the provider default."""
        return self.api_key_env or _PROVIDER_DEFAULTS[self.provider]["api_key_env"]

    @property
    def fallback_provider_list(self) -> list[str]:
        """Return the parsed list of fallback providers (may be empty)."""
        if not self.fallback_providers:
            return []
        return [p.strip() for p in self.fallback_providers.split(",") if p.strip()]


# ---------------------------------------------------------------------------
# Structured prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert visual change detection analyst. You will receive two \
images of the same scene or location, possibly taken from different angles, \
cameras, zoom levels, or at different times.

Your task is to determine what ACTUALLY changed between Image A (before) \
and Image B (after).

CRITICAL RULES:
1. If the same object appears in both images but from a different angle, \
zoom level, or camera position, it is NOT a change — report it as \
"unchanged" or "same_object_different_view".
2. Only report genuine changes: objects that were truly added, removed, \
moved to a different location, or modified.
3. Consider the SCENE as a whole — the same room/area photographed \
differently is not a change.
4. Be conservative: if you're unsure whether something changed or just \
looks different due to perspective, say "uncertain" and explain why.
5. Pay attention to the background, floor, walls, furniture to establish \
if it's the same scene.

Respond ONLY with valid JSON in this exact format:
{
  "scene_analysis": {
    "same_scene": true,
    "scene_description": "brief description of the scene/location",
    "viewpoint_change": "description of how the camera angle/position differs",
    "confidence": 0.95
  },
  "changes": [
    {
      "type": "added|removed|moved|modified|unchanged|same_object_different_view",
      "object": "object name",
      "description": "detailed explanation of what changed or why it's the same",
      "confidence": 0.9,
      "in_image_a": true,
      "in_image_b": true
    }
  ],
  "summary": "One paragraph summary of all actual changes (not viewpoint differences)",
  "total_real_changes": 0
}"""

_USER_SUFFIX = (
    "Analyze the changes between these two images. "
    "Remember: same object from different angles is NOT a change. "
    "Respond with JSON only."
)


# ---------------------------------------------------------------------------
# VLMReasoner
# ---------------------------------------------------------------------------

class VLMReasoner:
    """Use a Vision Language Model to reason about changes between two images.

    Unlike YOLO/pixel comparison, the VLM **understands** the scene and can
    distinguish between:

    - Same object from different angles (NOT a change)
    - Actually added/removed/moved objects (REAL changes)
    - Lighting/exposure differences vs structural changes

    Args:
        config: A :class:`ReasoningConfig` instance.  If *None*, one is
                built from the keyword arguments.
        provider: VLM provider override.
        model: Model name override.
        api_key: API key override (otherwise reads from env var).
    """

    def __init__(
        self,
        config: ReasoningConfig | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if config is not None:
            self._cfg = config
        else:
            kwargs: dict[str, Any] = {}
            if provider is not None:
                kwargs["provider"] = provider
            if model is not None:
                kwargs["model"] = model
            self._cfg = ReasoningConfig(**kwargs)

        # Resolve API key
        self._api_key = api_key or os.environ.get(self._cfg.resolved_api_key_env, "")
        if not self._api_key:
            raise ConfigError(
                f"API key not found. Set the {self._cfg.resolved_api_key_env} "
                f"environment variable or pass api_key= to VLMReasoner.",
                "reasoning.api_key",
            )

        logger.debug(
            "VLMReasoner: provider=%s, model=%s",
            self._cfg.provider,
            self._cfg.resolved_model,
        )
        self._last_usage: dict[str, Any] | None = None

    @property
    def last_usage(self) -> dict[str, Any] | None:
        """Token usage metadata from the most recent provider call."""
        return self._last_usage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        image_a: str | Path,
        image_b: str | Path,
    ) -> dict[str, Any]:
        """Send both images to the VLM and get a semantic change analysis.

        Args:
            image_a: Path to the reference (before) image.
            image_b: Path to the comparison (after) image.

        Returns:
            A dict with ``scene_analysis``, ``changes``, ``summary``,
            and ``total_real_changes``.

        Raises:
            ReasoningError: If the VLM call fails.
        """
        image_a = Path(image_a)
        image_b = Path(image_b)

        for p in (image_a, image_b):
            if not p.exists():
                raise ReasoningError(f"Image not found: {p}")

        logger.info(
            "Sending images to %s (%s) for reasoning ...",
            self._cfg.provider,
            self._cfg.resolved_model,
        )

        dispatch = {
            "huggingface": self._call_huggingface,
            "gemini": self._call_gemini,
            "openai": self._call_openai,
            "openrouter": self._call_openrouter,
        }

        # Build the ordered list of providers to attempt: primary first, then fallbacks
        providers_to_try = [self._cfg.provider] + self._cfg.fallback_provider_list
        last_error: Exception | None = None

        for provider in providers_to_try:
            if provider not in dispatch:
                logger.warning("Unknown fallback provider %r — skipping", provider)
                continue

            # Resolve API key for this provider (may differ from primary)
            api_key_env = _PROVIDER_DEFAULTS[provider]["api_key_env"]
            api_key = self._api_key if provider == self._cfg.provider else os.environ.get(api_key_env, "")

            if not api_key:
                logger.warning(
                    "Skipping provider %r: env var %s not set", provider, api_key_env
                )
                continue

            try:
                self._last_usage = None
                logger.info(
                    "Sending images to %s (%s) for reasoning ...",
                    provider,
                    self._cfg.resolved_model if provider == self._cfg.provider
                    else _PROVIDER_DEFAULTS[provider]["model"],
                )
                # Temporarily swap API key for fallback providers
                original_key = self._api_key
                object.__setattr__(self, "_api_key", api_key)
                try:
                    raw_response = dispatch[provider](image_a, image_b)
                finally:
                    object.__setattr__(self, "_api_key", original_key)

                result = self._parse_response(raw_response)
                if self._last_usage:
                    result["usage"] = self._last_usage
                n_changes = result.get("total_real_changes", len(result.get("changes", [])))
                logger.info(
                    "VLM reasoning complete via %s: %d real change(s) detected",
                    provider, n_changes,
                )
                return result

            except ReasoningError as exc:
                last_error = exc
                if providers_to_try.index(provider) < len(providers_to_try) - 1:
                    logger.warning("Provider %r failed (%s) — trying fallback ...", provider, exc)
                else:
                    logger.error("Provider %r failed: %s", provider, exc)

        raise ReasoningError(
            f"All providers failed. Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # HuggingFace provider  (FREE tier via Inference Providers)
    # ------------------------------------------------------------------

    def _call_huggingface(self, image_a: Path, image_b: Path) -> str:
        """Call HuggingFace Inference API with two images.

        Uses ``huggingface_hub.InferenceClient.chat_completion`` with
        OpenAI-compatible multi-modal messages containing base64 images.
        """
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ReasoningError(
                "huggingface_hub is required for HuggingFace reasoning. "
                "Install with: pip install huggingface_hub",
            ) from exc

        # Build the client
        hf_provider = self._cfg.hf_provider
        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if hf_provider and hf_provider != "auto":
            client_kwargs["provider"] = hf_provider

        client = InferenceClient(**client_kwargs)

        img_a_url = self._image_to_data_url(image_a)
        img_b_url = self._image_to_data_url(image_b)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image A (Before):"},
                    {"type": "image_url", "image_url": {"url": img_a_url}},
                    {"type": "text", "text": "Image B (After):"},
                    {"type": "image_url", "image_url": {"url": img_b_url}},
                    {"type": "text", "text": _USER_SUFFIX},
                ],
            },
        ]

        try:
            response = client.chat_completion(
                messages=messages,
                model=self._cfg.resolved_model,
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_tokens,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._last_usage = {
                    "provider": "huggingface",
                    "model": self._cfg.resolved_model,
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            content = response.choices[0].message.content or ""
            logger.debug("HuggingFace raw response length: %d chars", len(content))
            return content
        except Exception as exc:
            raise ReasoningError(
                f"HuggingFace API call failed: {exc}",
            ) from exc

    # ------------------------------------------------------------------
    # Gemini provider
    # ------------------------------------------------------------------

    def _call_gemini(self, image_a: Path, image_b: Path) -> str:
        """Call Google Gemini API with two images."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ReasoningError(
                "google-genai is required for Gemini reasoning. "
                "Install with: pip install google-genai",
            ) from exc

        client = genai.Client(api_key=self._api_key)

        img_a_bytes = image_a.read_bytes()
        img_b_bytes = image_b.read_bytes()
        mime_a = self._get_mime_type(image_a)
        mime_b = self._get_mime_type(image_b)

        try:
            response = client.models.generate_content(
                model=self._cfg.resolved_model,
                contents=[
                    _SYSTEM_PROMPT,
                    "\n\nImage A (Before):",
                    types.Part.from_bytes(data=img_a_bytes, mime_type=mime_a),
                    "\n\nImage B (After):",
                    types.Part.from_bytes(data=img_b_bytes, mime_type=mime_b),
                    f"\n\n{_USER_SUFFIX}",
                ],
                config=types.GenerateContentConfig(
                    temperature=self._cfg.temperature,
                    max_output_tokens=self._cfg.max_tokens,
                ),
            )
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                self._last_usage = {
                    "provider": "gemini",
                    "model": self._cfg.resolved_model,
                    "input_tokens": getattr(usage, "prompt_token_count", None),
                    "output_tokens": getattr(usage, "candidates_token_count", None),
                    "total_tokens": getattr(usage, "total_token_count", None),
                }
            return response.text or ""
        except Exception as exc:
            raise ReasoningError(f"Gemini API call failed: {exc}") from exc

    # ------------------------------------------------------------------
    # OpenAI provider
    # ------------------------------------------------------------------

    def _call_openai(self, image_a: Path, image_b: Path) -> str:
        """Call OpenAI API with two images."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ReasoningError(
                "openai is required for OpenAI reasoning. "
                "Install with: pip install openai",
            ) from exc

        client = OpenAI(api_key=self._api_key)

        img_a_url = self._image_to_data_url(image_a)
        img_b_url = self._image_to_data_url(image_b)

        try:
            response = client.chat.completions.create(
                model=self._cfg.resolved_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Image A (Before):"},
                            {
                                "type": "image_url",
                                "image_url": {"url": img_a_url, "detail": "high"},
                            },
                            {"type": "text", "text": "Image B (After):"},
                            {
                                "type": "image_url",
                                "image_url": {"url": img_b_url, "detail": "high"},
                            },
                            {"type": "text", "text": _USER_SUFFIX},
                        ],
                    },
                ],
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_tokens,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._last_usage = {
                    "provider": "openai",
                    "model": self._cfg.resolved_model,
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise ReasoningError(f"OpenAI API call failed: {exc}") from exc

    # ------------------------------------------------------------------
    # OpenRouter provider  (OpenAI-compatible, many free vision models)
    # ------------------------------------------------------------------

    def _call_openrouter(self, image_a: Path, image_b: Path) -> str:
        """Call OpenRouter API with two images.

        OpenRouter is OpenAI-compatible. Point the base URL to
        ``https://openrouter.ai/api/v1`` and use any vision model slug
        from https://openrouter.ai/models?modality=image+text->text.

        Free vision models (as of March 2026):
        - ``google/gemini-2.5-flash-preview:free``
        - ``openrouter/healer-alpha``  (logs data)
        - ``stepfun/step-3.5-flash:free``
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ReasoningError(
                "openai is required for OpenRouter reasoning. "
                "Install with: pip install openai",
            ) from exc

        client = OpenAI(
            api_key=self._api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        model = (
            self._cfg.resolved_model
            if self._cfg.provider == "openrouter"
            else _PROVIDER_DEFAULTS["openrouter"]["model"]
        )

        img_a_url = self._image_to_data_url(image_a)
        img_b_url = self._image_to_data_url(image_b)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Image A (Before):"},
                            {
                                "type": "image_url",
                                "image_url": {"url": img_a_url, "detail": "high"},
                            },
                            {"type": "text", "text": "Image B (After):"},
                            {
                                "type": "image_url",
                                "image_url": {"url": img_b_url, "detail": "high"},
                            },
                            {"type": "text", "text": _USER_SUFFIX},
                        ],
                    },
                ],
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_tokens,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._last_usage = {
                    "provider": "openrouter",
                    "model": model,
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise ReasoningError(f"OpenRouter API call failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse the VLM's JSON response, handling markdown fencing."""
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Some models (e.g. Qwen) wrap in <think>...</think> tags
        if "<think>" in text:
            think_end = text.find("</think>")
            if think_end != -1:
                text = text[think_end + len("</think>"):].strip()
                # Re-strip fences after removing think block
                if text.startswith("```json"):
                    text = text[7:]
                elif text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse VLM JSON response: %s", exc)
            logger.debug("Raw response:\n%s", raw[:2000])
            result = {
                "scene_analysis": {"same_scene": None, "confidence": 0},
                "changes": [],
                "summary": raw,
                "total_real_changes": -1,
                "parse_error": str(exc),
                "raw_response": raw,
            }

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mime_type(path: Path) -> str:
        """Determine MIME type from file extension."""
        ext = path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        return mime_map.get(ext, "image/jpeg")

    @staticmethod
    def _image_to_data_url(path: Path) -> str:
        """Convert an image file to a base64 data URL."""
        mime = VLMReasoner._get_mime_type(path)
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{data}"
