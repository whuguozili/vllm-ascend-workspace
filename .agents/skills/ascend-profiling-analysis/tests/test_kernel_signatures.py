"""Regression tests for ``knowledge/kernel_signatures.yaml``.

This file pins the contract between Python's ``categories_and_roles``
rule list and the YAML knowledge inventory. The intent is to make
"someone adds a new kernel rule in Python but forgets the YAML" a CI
failure rather than a silent drift.

Two checks:

1. **Structural** — the YAML parses, every category listed under
   ``kernels[].categories`` is a valid value in
   ``semantic_conventions.yaml:op_categories``.
2. **Behavioural** — fed a curated set of profile kernel names
   (taken from real DSV2 / DSV4 / Qwen3 / Mamba traces — see source
   citations next to each case), ``categories_and_roles`` returns the
   categories the YAML claims it should. This is the *executable* form
   of the inventory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

YAML = pytest.importorskip("yaml", reason="pyyaml not installed; kernel sig test skipped")


_SKILL_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _SKILL_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ascend_profile import common  # noqa: E402

KNOWLEDGE_DIR = _SCRIPTS / "ascend_profile" / "knowledge"
KERNEL_SIG_PATH = KNOWLEDGE_DIR / "kernel_signatures.yaml"
SEMCONV_PATH = KNOWLEDGE_DIR / "semantic_conventions.yaml"


def _load_yaml(path: Path) -> dict:
    return YAML.safe_load(path.read_text())


@pytest.fixture(scope="module")
def kernel_sig_doc() -> dict:
    return _load_yaml(KERNEL_SIG_PATH)


@pytest.fixture(scope="module")
def op_categories_enum() -> set[str]:
    doc = _load_yaml(SEMCONV_PATH)
    return set(doc["attributes"]["op_categories"]["values"])


def test_kernel_signatures_file_parses(kernel_sig_doc):
    assert kernel_sig_doc.get("version") == 1
    assert "kernels" in kernel_sig_doc
    assert isinstance(kernel_sig_doc["kernels"], list)
    # Every entry must at least have profile_name + categories.
    for entry in kernel_sig_doc["kernels"]:
        assert "profile_name" in entry, entry
        assert "categories" in entry, entry
        assert isinstance(entry["categories"], list), entry


def test_kernel_signatures_categories_in_enum(kernel_sig_doc, op_categories_enum):
    """Every category mentioned in kernel_signatures.yaml must be a valid enum value."""
    missing: set[str] = set()
    for entry in kernel_sig_doc["kernels"]:
        for cat in entry.get("categories", []):
            if cat not in op_categories_enum:
                missing.add(cat)
    assert not missing, (
        f"kernel_signatures.yaml references categories not declared in "
        f"semantic_conventions.yaml:op_categories: {sorted(missing)}"
    )


def test_deprecated_categories_table(kernel_sig_doc, op_categories_enum):
    """``deprecated_categories`` maps the old ``attention.csa*`` and
    ``attention.sfa*`` placeholders to the canonical paper-neutral
    kernel-level names (``attention.sparse_sharedkv``,
    ``attention.lightning_indexer``, ``attention.kv_compressor``, …).

    The RHS values must exist in the enum; the LHS values must NOT
    (they're deprecated)."""
    deprecated = kernel_sig_doc.get("deprecated_categories", {}) or {}
    for old, new in deprecated.items():
        assert old not in op_categories_enum, (
            f"deprecated category {old!r} still present in op_categories enum; "
            f"remove it from semantic_conventions.yaml"
        )
        assert new in op_categories_enum, (
            f"deprecated_categories: {old!r} maps to {new!r}, but {new!r} is "
            f"not in op_categories enum"
        )


# ----------------------------------------------------------------------------
# Behavioural cases: real kernel names → expected category subset.
# ----------------------------------------------------------------------------
# Each case is `(kernel_name, must_have_categories, must_not_have_categories)`.
# ``task_type`` and ``accelerator_core`` are not used by the rule list for
# attention/MoE decisions, so they're left blank.

_CASES: list[tuple[str, set[str], set[str]]] = [
    # ---- Sparse-attention building blocks (shared by DSA + CSA at the
    #      kernel level; family resolution lives in attention_families.yaml).
    (
        "KVQuantSparseAttnSharedKV",
        {"attention.sparse_sharedkv"},
        {"attention.sparse_sharedkv.metadata", "attention.mla", "attention.gqa_or_mha"},
    ),
    (
        "KVQuantSparseAttnSharedKVMetadata",
        {"attention.sparse_sharedkv.metadata"},
        {"attention.sparse_sharedkv", "attention.mla"},
    ),
    (
        "QuantLightningIndexer",
        {"attention.lightning_indexer"},
        {"attention.mla", "attention.kv_compressor"},
    ),
    (
        "IndexerCompressEpilogV2",
        {"attention.lightning_indexer"},
        {"attention.mla"},
    ),
    (
        "Compressor",
        {"attention.kv_compressor"},
        {"attention.mla", "attention.lightning_indexer"},
    ),
    (
        "KVCompressEpilog",
        {"attention.kv_compressor"},
        set(),
    ),
    (
        "BatchMatmulTranspose",
        {"attention.sparse_attn.v_up_proj", "compute.matmul"},
        {"attention.mla.v_up_proj"},
    ),
    # ---- MLA (DSV2 / V3, also reused by DSA in V3.2 paper §4)
    (
        "MlaPreprocess",
        {"attention.mla", "attention.mla.preprocess"},
        {"attention.sparse_sharedkv", "attention.lightning_indexer"},
    ),
    (
        "MlaProlog",  # CANN canonical op name (per CANN op_list.md)
        {"attention.mla", "attention.mla.preprocess"},
        set(),
    ),
    (
        "MlaPrologV2WeightNz",
        {"attention.mla", "attention.mla.preprocess"},
        set(),
    ),
    (
        "KvRmsNormRopeCache",
        {"attention.mla.kv_norm_rope_cache", "attention.rope"},
        {"attention.sparse_sharedkv"},
    ),
    (
        "TransposeQuantBatchMatmul",
        {"attention.mla.v_up_proj", "compute.matmul"},
        {"attention.sparse_attn.v_up_proj"},
    ),
    # ---- KVComp overlay
    (
        "NpuHammingDistTopK",
        {"attention.kvcomp.topk", "attention.kvcomp"},
        {"attention.sparse_sharedkv", "attention.mla"},
    ),
    (
        "NpuSignBitsPack",
        {"attention.kvcomp.signpack"},
        set(),
    ),
    (
        "NpuReshapeAndCacheBnsd",
        {"attention.kvcomp.cache_write"},
        set(),
    ),
    # ---- Dense GQA / MHA
    (
        "FusedInferAttentionScore",
        {"attention.gqa_or_mha"},
        {"attention.mla", "attention.sparse_sharedkv"},
    ),
    (
        "FusedInferAttentionScoreV2",
        {"attention.gqa_or_mha"},
        {"attention.mla", "attention.sparse_sharedkv"},
    ),
    (
        "UnpadFlashAttention",
        {"attention.gqa_or_mha"},
        set(),
    ),
    # ---- Linear / mamba / GDN
    (
        "CausalConv1d",
        {"attention.linear_or_mamba"},
        {"attention.gqa_or_mha", "attention.mla", "attention.sparse_sharedkv"},
    ),
    # ---- RoPE companions
    (
        "InterleaveRope",
        {"attention.rope.interleave", "attention.rope"},
        set(),
    ),
    (
        "InPlacePartialRotaryMul",
        {"attention.rope.partial", "attention.rope"},
        {"attention.rope.interleave"},
    ),
    # ---- MoE gating top-k (the genuine fused op only)
    (
        "MoeGatingTopKHash",
        {"moe.gating"},
        # MoeGatingTopKHash itself does NOT start with "hc", so the
        # block_head.mhc_prefix rule should not fire here.
        {"block_head.mhc_prefix"},
    ),
    # ---- HC* / MHC* — block_head structural prefix kernels.
    #      They appear in attention prologue AND moe routing prologue, so
    #      they must NOT be filed under moe.gating.
    (
        "HCPreSinkhorn",
        {"block_head.mhc_prefix"},
        {"moe.gating", "attention.mla"},
    ),
    (
        "HCPreInvRMS",
        {"block_head.mhc_prefix"},
        {"moe.gating"},
    ),
    (
        "HCPost",
        {"block_head.mhc_prefix"},
        {"moe.gating"},
    ),
    (
        "MhcRmsNorm",
        {"block_head.mhc_prefix", "normalization", "block_head"},
        {"moe.gating"},
    ),
    # ---- MoE dispatch / combine
    (
        "MoeDistributeDispatchV2",
        {"moe.dispatch"},
        {"moe.dispatch_expert_compute"},
    ),
    (
        "MoeDistributeCombineV2",
        {"moe.combine"},
        {"moe.dispatch_expert_compute"},
    ),
    (
        "DispatchFFNCombine",
        {"moe.dispatch_expert_compute"},
        {"moe.dispatch", "moe.combine"},
    ),
    (
        "DispatchGmmCombineDecode",
        {"moe.dispatch_expert_compute"},
        {"moe.dispatch", "moe.combine"},
    ),
    # ---- MoE expert matmul
    (
        "GroupedMatmul",
        {"moe.expert_matmul", "compute.matmul"},
        set(),
    ),
    # ---- Quant
    (
        "DynamicQuantV2",
        {"quant.dynamic", "compute.aux"},
        {"quant.mx"},
    ),
    (
        "DynamicMxQuant",
        {"quant.mx", "compute.aux"},
        {"quant.dynamic"},
    ),
    (
        "QuantBatchMatmulV3",
        {"compute.matmul", "quant.matmul"},
        set(),
    ),
    # ---- Communication
    (
        "hcom_allReduce",
        {"communication.collective", "communication.allreduce"},
        set(),
    ),
    (
        "hcom_allToAllV",
        {"communication.collective", "communication.alltoallv"},
        set(),
    ),
    # ---- Sampling
    (
        "ApplyTopKTopP",
        {"sampling.top_k_top_p", "sampling_or_selection"},
        set(),
    ),
]


@pytest.mark.parametrize("name,must_have,must_not_have", _CASES, ids=[c[0] for c in _CASES])
def test_kernel_classification_matches_knowledge(
    name: str, must_have: set[str], must_not_have: set[str]
) -> None:
    cats, _roles = common.categories_and_roles(name, "", "")
    cat_set = set(cats)
    missing = must_have - cat_set
    assert not missing, (
        f"kernel {name!r} expected to be tagged with {sorted(missing)}, "
        f"got {sorted(cat_set)}"
    )
    leaked = cat_set & must_not_have
    assert not leaked, (
        f"kernel {name!r} was tagged with categories that should NOT appear: "
        f"{sorted(leaked)} (full set: {sorted(cat_set)})"
    )


def test_deprecated_category_names_not_emitted_by_python() -> None:
    """The earlier drafts coined two non-canonical name families:
    ``attention.csa*`` (used as a generic catch-all) and ``attention.sfa*``
    (used after a wrong subagent reading). Neither set may be emitted any
    more — the kernel rule list now uses the paper-neutral names
    (``attention.sparse_sharedkv``, ``attention.lightning_indexer``,
    ``attention.kv_compressor``, …) and the paper-aligned architecture
    family (``csa`` / ``dsa`` / …) is resolved at the report layer.
    """
    samples = [
        "KVQuantSparseAttnSharedKV",
        "KVQuantSparseAttnSharedKVMetadata",
        "Compressor",
        "KVCompressEpilog",
        "QuantLightningIndexer",
        "IndexerCompressEpilogV2",
        "BatchMatmulTranspose",
    ]
    deprecated = {
        "attention.csa", "attention.csa.compressor", "attention.csa.indexer",
        "attention.csa.metadata",
        "attention.sfa", "attention.sfa.compressor", "attention.sfa.indexer",
        "attention.sfa.metadata", "attention.sfa.v_up_proj",
    }
    for name in samples:
        cats, _ = common.categories_and_roles(name, "", "")
        leaked = set(cats) & deprecated
        assert not leaked, (
            f"kernel {name!r} still tagged with deprecated category "
            f"{sorted(leaked)} — common.py rule list out of sync with "
            f"kernel_signatures.yaml:deprecated_categories"
        )
