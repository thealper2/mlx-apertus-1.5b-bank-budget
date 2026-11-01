"""Typed, validated loading of the YAML files in ``configs/``.

Each loader returns frozen dataclasses. Unknown keys, missing required keys, wrong types
and unsafe combinations raise :class:`ConfigError` with the offending key path.
"""

from __future__ import annotations

import re
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from bank_budget.utils.paths import find_repo_root, resolve

CONFIG_DIR = "configs"
DATA_CONFIG = "data.yaml"
TRAINING_CONFIG = "training.yaml"
EVALUATION_CONFIG = "evaluation.yaml"
PUBLISHING_CONFIG = "publishing.yaml"

_HF_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_FINE_TUNE_TYPES = {"lora", "dora"}
_OPTIMIZERS = {"adam", "adamw"}
_SCHEDULES = {"constant", "cosine"}
_PROFILE_KINDS = {"full", "smoke"}


class ConfigError(ValueError):
    """Raised for invalid or unsafe configuration."""


# --------------------------------------------------------------------------- helpers


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, with a clear error for missing files and syntax errors."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _section(data: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    if key not in data:
        raise ConfigError(f"{where}: missing required section '{key}'")
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{where}.{key} must be a mapping")
    return value


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    """Instantiate a flat dataclass from ``data`` with type and key checks."""
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            if f.default is MISSING and f.default_factory is MISSING:
                raise ConfigError(f"{where}: missing required key '{name}'")
            continue
        kwargs[name] = _check_type(data[name], f.type, f"{where}.{name}")
    return cls(**kwargs)


def _check_type(value: Any, annotation: Any, where: str) -> Any:
    """Check ``value`` against a simple annotation (a type or its string form)."""
    ann = annotation.__name__ if isinstance(annotation, type) else str(annotation)
    ann = ann.replace(" ", "")
    optional = ann.endswith("|None")
    base = ann.removesuffix("|None")
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{where} must not be null")
    if base == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer, got {value!r}")
    elif base == "float":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where} must be a number, got {value!r}")
        value = float(value)
    elif base == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false, got {value!r}")
    elif base == "str":
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{where} must be a non-empty string, got {value!r}")
    elif base == "list[str]" and (
        not isinstance(value, list) or not all(isinstance(v, str) and v for v in value)
    ):
        raise ConfigError(f"{where} must be a list of non-empty strings")
    return value


def _positive(value: int | float, where: str, allow_zero: bool = False) -> None:
    if value < 0 or (value == 0 and not allow_zero):
        raise ConfigError(f"{where} must be {'>= 0' if allow_zero else '> 0'}, got {value}")


def validate_hf_id(repo_id: str, where: str) -> None:
    """Validate a Hugging Face ``namespace/name`` identifier."""
    if not _HF_ID.match(repo_id) or ".." in repo_id or len(repo_id) > 96:
        raise ConfigError(f"{where}: '{repo_id}' is not a valid Hugging Face repo id")


def _validate_revision(revision: str | None, where: str) -> None:
    if revision is not None and not _REVISION.match(revision):
        raise ConfigError(f"{where}: revision must be a 40-character commit hash, got {revision!r}")


# --------------------------------------------------------------------------- data

SPLITS = ("train", "valid", "test_new", "test_known")
TEST_SPLITS = ("test_new", "test_known")


@dataclass(frozen=True)
class DatasetSource:
    repo_id: str
    revision: str | None
    license: str | None
    file: str
    expected_lines: int


@dataclass(frozen=True)
class SplitConfig:
    holdout_share: float
    holdout_max_group_share: float
    holdout_seed: int
    test_per_category: int
    test_new_seed: int
    test_known_seed: int
    valid_per_category: int
    valid_seed: int
    train_shuffle_seed: int


@dataclass(frozen=True)
class SmokeDataConfig:
    dir: str
    mlx_dir: str
    train_size: int
    valid_size: int
    test_per_category: int


@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetSource
    split: SplitConfig
    expected_rows: dict[str, int]
    expected_unique_lines: int
    output_dir: Path
    mlx_dir: Path
    smoke: SmokeDataConfig
    smoke_dir: Path
    smoke_mlx_dir: Path


def _split_mapping(value: Any, kind: type, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(SPLITS):
        raise ConfigError(f"{where} must map exactly {list(SPLITS)}")
    out = {k: _check_type(value[k], kind, f"{where}.{k}") for k in SPLITS}
    if kind is int:
        for k, v in out.items():
            _positive(v, f"{where}.{k}")
    return out


def parse_data_config(raw: dict[str, Any], root: Path) -> DataConfig:
    """Validate the contents of ``data.yaml``."""
    w = DATA_CONFIG
    allowed = {"dataset", "split", "expected_rows", "expected_unique_lines", "output", "smoke"}
    if unknown := sorted(set(raw) - allowed):
        raise ConfigError(f"{w}: unknown section(s) {unknown}")
    source = _build(DatasetSource, _section(raw, "dataset", w), f"{w}.dataset")
    validate_hf_id(source.repo_id, f"{w}.dataset.repo_id")
    _validate_revision(source.revision, f"{w}.dataset.revision")
    if not source.file.endswith(".csv") or source.file.startswith(("/", "..")):
        raise ConfigError(f"{w}.dataset.file must be a relative .csv path")
    _positive(source.expected_lines, f"{w}.dataset.expected_lines")
    split = _build(SplitConfig, _section(raw, "split", w), f"{w}.split")
    for name in ("holdout_share", "holdout_max_group_share"):
        if not 0.0 < getattr(split, name) < 1.0:
            raise ConfigError(f"{w}.split.{name} must be in (0, 1)")
    _positive(split.test_per_category, f"{w}.split.test_per_category")
    _positive(split.valid_per_category, f"{w}.split.valid_per_category")
    expected = _split_mapping(raw.get("expected_rows"), int, f"{w}.expected_rows")
    unique = _check_type(raw.get("expected_unique_lines"), int, f"{w}.expected_unique_lines")
    _positive(unique, f"{w}.expected_unique_lines")
    if sum(expected.values()) > unique or unique > source.expected_lines:
        raise ConfigError(f"{w}: expected_rows / expected_unique_lines are inconsistent")
    output = _section(raw, "output", w)
    out_dir = _check_type(output.get("dir"), str, f"{w}.output.dir")
    mlx_dir = _check_type(output.get("mlx_dir"), str, f"{w}.output.mlx_dir")
    smoke = _build(SmokeDataConfig, _section(raw, "smoke", w), f"{w}.smoke")
    for name in ("train_size", "valid_size", "test_per_category"):
        _positive(getattr(smoke, name), f"{w}.smoke.{name}")
    if (
        smoke.train_size > expected["train"]
        or smoke.valid_size > expected["valid"]
        or smoke.test_per_category > split.test_per_category
    ):
        raise ConfigError(f"{w}.smoke sizes cannot exceed the canonical sizes")
    cfg = DataConfig(
        dataset=source,
        split=split,
        expected_rows=expected,
        expected_unique_lines=unique,
        output_dir=resolve(out_dir, root),
        mlx_dir=resolve(mlx_dir, root),
        smoke=smoke,
        smoke_dir=resolve(smoke.dir, root),
        smoke_mlx_dir=resolve(smoke.mlx_dir, root),
    )
    if cfg.smoke_dir.resolve() == cfg.output_dir.resolve():
        raise ConfigError(f"{w}: smoke.dir must differ from output.dir")
    return cfg


# --------------------------------------------------------------------------- training


@dataclass(frozen=True)
class ModelConfig:
    repo_id: str
    revision: str | None
    license: str | None = None
    upstream: str | None = None


@dataclass(frozen=True)
class LoraParameters:
    rank: int
    scale: float
    dropout: float
    keys: list[str]


@dataclass(frozen=True)
class TrainingProfile:
    """One complete, explicit MLX-LM training recipe."""

    name: str
    kind: str
    data_dir: str
    adapter_path: str
    fine_tune_type: str
    optimizer: str
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    bias_correction: bool
    learning_rate: float
    lr_schedule: str
    warmup_steps: int
    decay_steps: int
    end_learning_rate: float
    iters: int
    batch_size: int
    grad_accumulation_steps: int
    max_seq_length: int
    num_layers: int
    lora: LoraParameters
    mask_prompt: bool
    grad_checkpoint: bool
    memory_limit_gb: float | None  # MLX's scheduling memory limit; memory only
    cache_limit_gb: float | None  # cap on MLX's buffer cache; memory only, results unchanged
    steps_per_report: int
    steps_per_eval: int
    val_batches: int
    save_every: int
    seed: int

    @property
    def examples_per_update(self) -> int:
        return self.batch_size * self.grad_accumulation_steps

    @property
    def optimizer_updates(self) -> int:
        return self.iters // self.grad_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyConfig:
    smoke_max_iters: int = 5
    smoke_max_seq_length: int = 2048
    warn_seq_length_8gb: int = 2048
    min_free_disk_gb: float = 6.0


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig
    profiles: dict[str, TrainingProfile]
    safety: SafetyConfig

    def profile(self, name: str) -> TrainingProfile:
        if name not in self.profiles:
            raise ConfigError(
                f"Unknown training profile '{name}'. Available: {', '.join(sorted(self.profiles))}"
            )
        return self.profiles[name]


def _parse_profile(name: str, raw: dict[str, Any], safety: SafetyConfig) -> TrainingProfile:
    w = f"{TRAINING_CONFIG}.profiles.{name}"
    if not isinstance(raw, dict):
        raise ConfigError(f"{w} must be a mapping")
    raw = dict(raw)
    lora_raw = raw.pop("lora", None)
    if not isinstance(lora_raw, dict):
        raise ConfigError(f"{w}: missing required mapping 'lora'")
    lora = _build(LoraParameters, lora_raw, f"{w}.lora")
    raw["name"] = name
    profile_fields = {f.name for f in fields(TrainingProfile)} - {"lora"}
    if unknown := sorted(set(raw) - profile_fields):
        raise ConfigError(f"{w}: unknown key(s) {unknown}")
    kwargs: dict[str, Any] = {}
    for f in fields(TrainingProfile):
        if f.name == "lora":
            continue
        if f.name not in raw:
            raise ConfigError(f"{w}: missing required key '{f.name}'")
        kwargs[f.name] = _check_type(raw[f.name], f.type, f"{w}.{f.name}")
    profile = TrainingProfile(lora=lora, **kwargs)
    validate_profile(profile, safety)
    return profile


def validate_profile(p: TrainingProfile, safety: SafetyConfig) -> None:
    """Range checks and smoke-profile guard rails for a training profile."""
    w = f"{TRAINING_CONFIG}.profiles.{p.name}"
    if p.kind not in _PROFILE_KINDS:
        raise ConfigError(f"{w}.kind must be one of {sorted(_PROFILE_KINDS)}")
    if p.fine_tune_type not in _FINE_TUNE_TYPES:
        raise ConfigError(f"{w}.fine_tune_type must be one of {sorted(_FINE_TUNE_TYPES)}")
    if p.optimizer not in _OPTIMIZERS:
        raise ConfigError(f"{w}.optimizer must be one of {sorted(_OPTIMIZERS)}")
    if p.lr_schedule not in _SCHEDULES:
        raise ConfigError(f"{w}.lr_schedule must be one of {sorted(_SCHEDULES)}")
    for name in (
        "learning_rate",
        "iters",
        "batch_size",
        "grad_accumulation_steps",
        "max_seq_length",
        "steps_per_report",
        "steps_per_eval",
        "save_every",
    ):
        _positive(getattr(p, name), f"{w}.{name}")
    _positive(p.weight_decay, f"{w}.weight_decay", allow_zero=True)
    _positive(p.warmup_steps, f"{w}.warmup_steps", allow_zero=True)
    _positive(p.adam_eps, f"{w}.adam_eps")
    for name in ("adam_beta1", "adam_beta2"):
        if not 0.0 <= getattr(p, name) < 1.0:
            raise ConfigError(f"{w}.{name} must be in [0, 1)")
    if p.lr_schedule == "cosine":
        _positive(p.decay_steps, f"{w}.decay_steps")
        if not 0.0 <= p.end_learning_rate < p.learning_rate:
            raise ConfigError(f"{w}.end_learning_rate must be in [0, learning_rate)")
    if p.cache_limit_gb is not None:
        _positive(p.cache_limit_gb, f"{w}.cache_limit_gb")
    if p.memory_limit_gb is not None:
        _positive(p.memory_limit_gb, f"{w}.memory_limit_gb")
    if p.num_layers == 0 or p.num_layers < -1:
        raise ConfigError(f"{w}.num_layers must be > 0 or -1 (all layers)")
    if p.val_batches == 0 or p.val_batches < -1:
        raise ConfigError(f"{w}.val_batches must be > 0 or -1 (whole validation set)")
    _positive(p.lora.rank, f"{w}.lora.rank")
    _positive(p.lora.scale, f"{w}.lora.scale")
    if not 0.0 <= p.lora.dropout < 1.0:
        raise ConfigError(f"{w}.lora.dropout must be in [0, 1)")
    if p.learning_rate > 1e-2:
        raise ConfigError(f"{w}.learning_rate={p.learning_rate} is implausibly large for LoRA")
    if p.iters % p.grad_accumulation_steps:
        raise ConfigError(f"{w}.iters must be a multiple of grad_accumulation_steps")
    if p.lr_schedule != "constant" and p.warmup_steps >= p.optimizer_updates:
        raise ConfigError(f"{w}.warmup_steps must be smaller than the number of updates")
    if p.lr_schedule == "constant" and p.warmup_steps:
        raise ConfigError(f"{w}: warmup_steps needs the cosine lr_schedule")
    if p.kind == "smoke":
        if p.iters > safety.smoke_max_iters:
            raise ConfigError(
                f"{w}: smoke profile has iters={p.iters} > safety.smoke_max_iters="
                f"{safety.smoke_max_iters}; a smoke run must stay tiny"
            )
        if p.max_seq_length > safety.smoke_max_seq_length:
            raise ConfigError(f"{w}: smoke profile max_seq_length exceeds the safety limit")


def parse_training_config(raw: dict[str, Any]) -> TrainingConfig:
    """Validate the contents of ``training.yaml``."""
    w = TRAINING_CONFIG
    if unknown := sorted(set(raw) - {"model", "profiles", "safety"}):
        raise ConfigError(f"{w}: unknown section(s) {unknown}")
    model = _build(ModelConfig, _section(raw, "model", w), f"{w}.model")
    validate_hf_id(model.repo_id, f"{w}.model.repo_id")
    _validate_revision(model.revision, f"{w}.model.revision")
    safety = _build(SafetyConfig, raw.get("safety") or {}, f"{w}.safety")
    profiles_raw = _section(raw, "profiles", w)
    if not profiles_raw:
        raise ConfigError(f"{w}.profiles must define at least one profile")
    profiles = {n: _parse_profile(n, p, safety) for n, p in profiles_raw.items()}
    if "reference" not in profiles:
        raise ConfigError(f"{w}.profiles must keep the 'reference' profile")
    paths = [p.adapter_path for p in profiles.values()]
    if len(set(paths)) != len(paths):
        raise ConfigError(f"{w}: every profile needs its own adapter_path")
    return TrainingConfig(model=model, profiles=profiles, safety=safety)


# --------------------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class EvaluationConfig:
    test_files: dict[str, Path]
    smoke_test_files: dict[str, Path]
    statement_file: Path
    output_dir: Path
    max_tokens: int
    warmup_generations: int
    max_examples: int | None
    smoke_max_examples: int


def parse_evaluation_config(raw: dict[str, Any], root: Path) -> EvaluationConfig:
    """Validate the contents of ``evaluation.yaml``."""
    w = EVALUATION_CONFIG
    allowed = {
        "test_files",
        "smoke_test_files",
        "statement_file",
        "output_dir",
        "generation",
        "limits",
    }
    if unknown := sorted(set(raw) - allowed):
        raise ConfigError(f"{w}: unknown key(s) {unknown}")

    def files(key: str) -> dict[str, Path]:
        value = raw.get(key)
        if not isinstance(value, dict) or set(value) != set(TEST_SPLITS):
            raise ConfigError(f"{w}.{key} must map exactly {list(TEST_SPLITS)}")
        return {k: resolve(_check_type(value[k], str, f"{w}.{key}.{k}"), root) for k in TEST_SPLITS}

    gen = _section(raw, "generation", w)
    lim = _section(raw, "limits", w)
    for where, section, keys in (
        ("generation", gen, {"max_tokens", "warmup_generations"}),
        ("limits", lim, {"max_examples", "smoke_max_examples"}),
    ):
        if unknown := sorted(set(section) - keys):
            raise ConfigError(f"{w}.{where}: unknown key(s) {unknown}")
    cfg = EvaluationConfig(
        test_files=files("test_files"),
        smoke_test_files=files("smoke_test_files"),
        statement_file=resolve(
            _check_type(raw.get("statement_file"), str, f"{w}.statement_file"), root
        ),
        output_dir=resolve(_check_type(raw.get("output_dir"), str, f"{w}.output_dir"), root),
        max_tokens=_check_type(gen.get("max_tokens"), int, f"{w}.generation.max_tokens"),
        warmup_generations=_check_type(
            gen.get("warmup_generations"), int, f"{w}.generation.warmup_generations"
        ),
        max_examples=_check_type(lim.get("max_examples"), int | None, f"{w}.limits.max_examples"),
        smoke_max_examples=_check_type(
            lim.get("smoke_max_examples"), int, f"{w}.limits.smoke_max_examples"
        ),
    )
    _positive(cfg.max_tokens, f"{w}.generation.max_tokens")
    _positive(cfg.warmup_generations, f"{w}.generation.warmup_generations", allow_zero=True)
    _positive(cfg.smoke_max_examples, f"{w}.limits.smoke_max_examples")
    if cfg.max_examples is not None:
        _positive(cfg.max_examples, f"{w}.limits.max_examples")
    if cfg.smoke_max_examples > 50:
        raise ConfigError(f"{w}.limits.smoke_max_examples must stay small (<= 50)")
    return cfg


# --------------------------------------------------------------------------- publishing


@dataclass(frozen=True)
class PublishingConfig:
    repo_id: str | None
    private: bool
    mode: str
    adapter_path: Path
    model_card_path: Path
    staging_dir: Path
    commit_message: str
    adapter_license: str
    include: list[str] = field(default_factory=list)


def parse_publishing_config(raw: dict[str, Any], root: Path) -> PublishingConfig:
    """Validate the contents of ``publishing.yaml``."""
    w = PUBLISHING_CONFIG
    allowed = {f.name for f in fields(PublishingConfig)}
    if unknown := sorted(set(raw) - allowed):
        raise ConfigError(f"{w}: unknown key(s) {unknown}")
    repo_id = _check_type(raw.get("repo_id"), str | None, f"{w}.repo_id")
    if repo_id is not None:
        validate_hf_id(repo_id, f"{w}.repo_id")
    mode = _check_type(raw.get("mode", "adapter"), str, f"{w}.mode")
    if mode != "adapter":
        raise ConfigError(f"{w}.mode: only 'adapter' publication is supported")
    include = _check_type(raw.get("include"), list[str], f"{w}.include")
    for name in include:
        if "/" in name or name.startswith("."):
            raise ConfigError(f"{w}.include: '{name}' must be a plain file name")
    return PublishingConfig(
        repo_id=repo_id,
        private=_check_type(raw.get("private", True), bool, f"{w}.private"),
        mode=mode,
        adapter_path=resolve(_check_type(raw.get("adapter_path"), str, f"{w}.adapter_path"), root),
        model_card_path=resolve(
            _check_type(raw.get("model_card_path"), str, f"{w}.model_card_path"), root
        ),
        staging_dir=resolve(_check_type(raw.get("staging_dir"), str, f"{w}.staging_dir"), root),
        commit_message=_check_type(raw.get("commit_message"), str, f"{w}.commit_message"),
        adapter_license=_check_type(raw.get("adapter_license"), str, f"{w}.adapter_license"),
        include=include,
    )


# --------------------------------------------------------------------------- bundle


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    publishing: PublishingConfig


def load_project_config(root: Path | None = None, config_dir: Path | None = None) -> ProjectConfig:
    """Load and validate all four configuration files."""
    root = root or find_repo_root()
    cdir = config_dir or (root / CONFIG_DIR)
    data = parse_data_config(read_yaml(cdir / DATA_CONFIG), root)
    training = parse_training_config(read_yaml(cdir / TRAINING_CONFIG))
    evaluation = parse_evaluation_config(read_yaml(cdir / EVALUATION_CONFIG), root)
    publishing = parse_publishing_config(read_yaml(cdir / PUBLISHING_CONFIG), root)
    return ProjectConfig(root, data, training, evaluation, publishing)
