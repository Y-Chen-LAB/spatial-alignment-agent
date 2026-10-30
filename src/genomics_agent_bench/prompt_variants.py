from __future__ import annotations

from pathlib import Path


DEFAULT_PROMPT_VARIANT = "default"
NO_PACKAGE_PROMPT_VARIANT = "no-package"
FULL_BASIC_PROMPT_VARIANT = "full-basic"
METHOD_LIST_PROMPT_VARIANT = "paper-method-list"
METHOD_PACKAGE_FIRST_PROMPT_VARIANT = "paper-method-package-first"
METHOD_PACKAGE_BALANCED_PROMPT_VARIANT = "paper-method-package-balanced"
FULL_PACKAGE_PROMPT_VARIANT = "full-package"
FULL_PACKAGE_PAPER_PRIOR_PROMPT_VARIANT = "full-package-paper-prior"
NO_METHOD_HELPER_PROMPT_VARIANTS = (
    FULL_BASIC_PROMPT_VARIANT,
)
PACKAGE_BACKED_PROMPT_VARIANTS = (
    FULL_PACKAGE_PROMPT_VARIANT,
    FULL_PACKAGE_PAPER_PRIOR_PROMPT_VARIANT,
)
PROMPT_VARIANTS = (
    DEFAULT_PROMPT_VARIANT,
    FULL_BASIC_PROMPT_VARIANT,
    FULL_PACKAGE_PROMPT_VARIANT,
    FULL_PACKAGE_PAPER_PRIOR_PROMPT_VARIANT,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _asset_path(variant: str) -> Path:
    assets = {
        METHOD_LIST_PROMPT_VARIANT: "spatial_alignment_methods_no_rankings.md",
        METHOD_PACKAGE_FIRST_PROMPT_VARIANT: "spatial_alignment_methods_package_first.md",
        METHOD_PACKAGE_BALANCED_PROMPT_VARIANT: "spatial_alignment_methods_package_first.md",
        FULL_PACKAGE_PROMPT_VARIANT: "spatial_alignment_methods_package_first.md",
        FULL_PACKAGE_PAPER_PRIOR_PROMPT_VARIANT: "spatial_alignment_methods_paper_prior.md",
    }
    try:
        filename = assets[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt variant: {variant}") from exc
    return _repo_root() / "references" / "prompt_assets" / filename


def load_spatial_alignment_prompt_context(variant: str) -> str:
    if variant in {DEFAULT_PROMPT_VARIANT, *NO_METHOD_HELPER_PROMPT_VARIANTS}:
        return ""
    return _asset_path(variant).read_text(encoding="utf-8").strip()


def method_prebootstrap_target(variant: str) -> str | None:
    if variant in {FULL_PACKAGE_PROMPT_VARIANT, FULL_PACKAGE_PAPER_PRIOR_PROMPT_VARIANT}:
        return "paper-all"
    return None


def render_cli_prompt_context(variant: str) -> str:
    return load_spatial_alignment_prompt_context(variant)


def prompt_variant_tag(variant: str) -> str:
    if variant == DEFAULT_PROMPT_VARIANT:
        return ""
    safe = variant.replace("/", "-").replace("_", "-")
    return f"_pv-{safe}"
