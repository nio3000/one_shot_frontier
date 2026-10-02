"""CMR-V1 X2 unit tests (T1-T14).

These tests cover the frozen X2 machinery only:

- T1  RFF seed derivation is deterministic and matches the protocol recipe;
- T2  frozen ``W``/``b`` arrays are byte-identical on rerun;
- T3  frozen RFF dimension is 256;
- T4  ResNet-18 ``layer1..layer4`` extraction produces valid shapes;
- T5  ViT-B/16 ``B3``/``B6``/``B9``/``B12`` extraction produces valid shapes and
      uses the post-block CLS token before the final encoder LayerNorm;
- T6  L2 normalization is correct;
- T7  client aggregation reproduces the centralized message;
- T8  prototype reconstruction is correct;
- T9  the classifier tie rule is smallest canonical class ID;
- T10 balanced accuracy matches the existing project utility;
- T11 ACTION_RELEVANT threshold is exactly 0.01;
- T12 UNIQUE_BEST margin is exactly 0.002;
- T13 protocol version/status are unchanged;
- T14 no X3-X5 artifact exists (the CMR-D closeout adds
      ``evidence/cmr_v1/final_gate_summary.json``, which is not an X3-X5 marker).

All tests are outcome-free: they use synthetic data, random tensors and public
frozen constants, never real X2 BACC values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontier.cmr_feature_extract import (  # noqa: E402
    RESNET_ENCODER_ID,
    VIT_ENCODER_ID,
    X2_N_CLIENTS,
    X2_RAW_LAYER_DIM,
    build_encoder,
    client_partition,
    layer_modules,
    pooled_layer_features,
    verify_encoder_layers,
)
from frontier.cmr_kernel_layer import (  # noqa: E402
    PROTOCOL_ID,
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
    X2_ACTION_RELEVANT_SPREAD_MIN,
    X2_AGGREGATION_TOLERANCE,
    X2_CONDITIONS,
    X2_DATASETS,
    X2_DECISION_FAIL,
    X2_DECISION_PASS,
    X2_ENCODER_LAYERS,
    X2_UNIQUE_BEST_MARGIN_MIN,
    aggregate_client_messages,
    apply_rff,
    array_sha256,
    bacc_from_predictions,
    evaluate_x2_gates,
    generate_rff,
    load_x2_protocol_constants,
    normalize_layer_features,
    predict_prototype_batch,
    reconstruct_prototypes,
    rff_seed,
    summarize_x2_condition,
)

PROTOCOL_YAML = REPO_ROOT / "configs/cross_mechanism_replication_protocol_v1.yaml"
PROTOCOL_MD = REPO_ROOT / "docs/governance/nature/CROSS_MECHANISM_REPLICATION_PROTOCOL_V1.md"
GATE_MATRIX = REPO_ROOT / "docs/phases/cmr_v1/CMR_V1_GATE_MATRIX.csv"

FROZEN_PROTOCOL_YAML_SHA256 = (
    "8004ae02b690163ce41494fbf6f5464c6aee1712709d05428727fa9d658a1e4b"
)
FROZEN_PROTOCOL_MD_SHA256 = (
    "58132ae03f3ab70bb82443a30d7c313a39d068eeb2fb2533d68b5a4b2ab1025a"
)
FROZEN_GATE_MATRIX_SHA256 = (
    "91e4834f5f671062c6f45ed87b6ef72235aeada0eef187eb8df33f66c0940001"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- T1: RFF seed derivation -------------------------------------------------
def test_t1_rff_seed_derivation_matches_protocol_recipe():
    for encoder, layers in X2_ENCODER_LAYERS.items():
        for layer in layers:
            digest = hashlib.sha256(f"20261001|{encoder}|{layer}".encode("utf-8")).digest()
            expected = int.from_bytes(digest[:8], "big") % (2**32)
            assert rff_seed(encoder, layer) == expected


def test_t1_rff_seed_is_deterministic_and_layer_specific():
    seeds = {}
    for encoder, layers in X2_ENCODER_LAYERS.items():
        resolved = [rff_seed(encoder, layer) for layer in layers]
        assert resolved == [rff_seed(encoder, layer) for layer in layers]
        assert len(set(resolved)) == len(resolved)
        seeds[encoder] = resolved
    # Different encoders must not collide at the same frozen layer index.
    assert len(set(seeds[RESNET_ENCODER_ID]) & set(seeds[VIT_ENCODER_ID])) == 0


def test_t1_rff_seed_rejects_layers_outside_action_family():
    with pytest.raises(ValueError):
        rff_seed(RESNET_ENCODER_ID, "layer5")
    with pytest.raises(ValueError):
        rff_seed(VIT_ENCODER_ID, "B1")
    with pytest.raises(ValueError):
        rff_seed("resnet50_imagenet1k_v1", "layer1")


# --- T2 / T3: frozen RFF arrays ---------------------------------------------
def test_t2_rff_arrays_are_byte_identical_on_rerun():
    for encoder, layers in X2_ENCODER_LAYERS.items():
        dim = X2_RAW_LAYER_DIM[encoder][layers[0]]
        for layer in layers:
            w1, b1 = generate_rff(encoder, layer, dim)
            w2, b2 = generate_rff(encoder, layer, dim)
            assert array_sha256(w1) == array_sha256(w2)
            assert array_sha256(b1) == array_sha256(b2)
            assert np.array_equal(w1, w2) and np.array_equal(b1, b2)


def test_t2_rff_npz_roundtrip_is_byte_identical(tmp_path):
    encoder, layer = RESNET_ENCODER_ID, "layer3"
    w, b = generate_rff(encoder, layer, X2_RAW_LAYER_DIM[encoder][layer])
    path = tmp_path / "map.npz"
    np.savez(path, W=w, b=b)
    with np.load(path) as frozen:
        w2 = np.asarray(frozen["W"], dtype=np.float64)
        b2 = np.asarray(frozen["b"], dtype=np.float64)
    assert array_sha256(w) == array_sha256(w2)
    assert array_sha256(b) == array_sha256(b2)


def test_t3_rff_dimension_and_dtype_and_sigma_are_frozen():
    authority = load_x2_protocol_constants(PROTOCOL_YAML)
    assert authority["rff"]["dimension"] == 256
    assert authority["rff"]["sigma"] == 1.0
    assert authority["rff"]["master_seed"] == 20261001
    assert authority["rff"]["generation_dtype"] == "float64"
    for encoder, layers in X2_ENCODER_LAYERS.items():
        dim = X2_RAW_LAYER_DIM[encoder][layers[0]]
        for layer in layers:
            w, b = generate_rff(encoder, layer, dim)
            assert w.shape == (256, dim)
            assert b.shape == (256,)
            assert w.dtype == np.float64 and b.dtype == np.float64
            # sigma = 1.0 means W is standard normal.
            assert abs(float(np.std(w)) - 1.0) < 0.05
            assert 0.0 <= float(np.min(b)) and float(np.max(b)) <= 2.0 * np.pi


def test_t3_apply_rff_matches_closed_form():
    rng = np.random.default_rng(7)
    features = normalize_layer_features(rng.normal(size=(5, 32)))
    w, b = generate_rff(RESNET_ENCODER_ID, "layer1", 32)
    phi = apply_rff(features, w, b)
    expected = np.sqrt(2.0 / 256.0) * np.cos(features @ w.T + b)
    assert np.allclose(phi, expected, rtol=0.0, atol=1e-15)
    with pytest.raises(ValueError):
        apply_rff(features, w[:, :16], b)


# --- T4 / T5: encoder layer extraction --------------------------------------
@pytest.fixture(scope="module")
def resnet_encoder():
    model, _weights, _torch_version, _torchvision_version = build_encoder(RESNET_ENCODER_ID, "cpu")
    return model


@pytest.fixture(scope="module")
def vit_encoder():
    model, _weights, _torch_version, _torchvision_version = build_encoder(VIT_ENCODER_ID, "cpu")
    return model


def test_t4_resnet_layer_extraction_shapes(resnet_encoder):
    import torch

    torch.manual_seed(0)
    batch = torch.randn(2, 3, 224, 224)
    features = pooled_layer_features(resnet_encoder, RESNET_ENCODER_ID, batch)
    assert set(features) == set(X2_ENCODER_LAYERS[RESNET_ENCODER_ID])
    for layer, array in features.items():
        assert array.shape == (2, X2_RAW_LAYER_DIM[RESNET_ENCODER_ID][layer])
        assert array.dtype == np.float32
        assert np.isfinite(array).all()


def test_t4_resnet_features_match_manual_stage_pooling(resnet_encoder):
    import torch

    torch.manual_seed(2)
    batch = torch.randn(3, 3, 224, 224)
    features = pooled_layer_features(resnet_encoder, RESNET_ENCODER_ID, batch)
    captured = {}
    handle = resnet_encoder.layer2.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("stage", out.detach())
    )
    try:
        with torch.inference_mode():
            resnet_encoder(batch)
    finally:
        handle.remove()
    manual = (
        torch.nn.functional.adaptive_avg_pool2d(captured["stage"], 1)
        .flatten(1)
        .to(torch.float32)
        .numpy()
    )
    assert np.array_equal(features["layer2"], manual)


def test_t5_vit_layer_extraction_shapes(vit_encoder):
    import torch

    torch.manual_seed(1)
    batch = torch.randn(2, 3, 224, 224)
    features = pooled_layer_features(vit_encoder, VIT_ENCODER_ID, batch)
    assert set(features) == set(X2_ENCODER_LAYERS[VIT_ENCODER_ID])
    for layer, array in features.items():
        assert array.shape == (2, 768)
        assert array.dtype == np.float32
        assert np.isfinite(array).all()


def test_t5_vit_layer_is_post_block_cls_before_final_layernorm(vit_encoder):
    """The extracted B_k features must be pre-LayerNorm post-block CLS tokens."""
    import torch

    torch.manual_seed(3)
    batch = torch.randn(2, 3, 224, 224)
    features = pooled_layer_features(vit_encoder, VIT_ENCODER_ID, batch)
    module = layer_modules(vit_encoder, VIT_ENCODER_ID)["B12"]
    captured = {}
    handle = module.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("block", out.detach())
    )
    try:
        with torch.inference_mode():
            vit_encoder(batch)
    finally:
        handle.remove()
    pre_ln = captured["block"][:, 0, :].to(torch.float32).numpy()
    assert np.array_equal(features["B12"], pre_ln)
    # The frozen layer must not equal the post-final-LayerNorm representation.
    post_ln = vit_encoder.encoder.ln(captured["block"])[:, 0, :].to(torch.float32).numpy()
    assert not np.allclose(features["B12"], post_ln)


def test_t4_t5_frozen_action_family_rejects_substitution():
    assert verify_encoder_layers(RESNET_ENCODER_ID, ("layer1", "layer2", "layer3", "layer4"))
    assert verify_encoder_layers(VIT_ENCODER_ID, ("B3", "B6", "B9", "B12"))
    with pytest.raises(ValueError):
        verify_encoder_layers(RESNET_ENCODER_ID, ("layer1", "layer2", "layer3"))
    with pytest.raises(ValueError):
        verify_encoder_layers(VIT_ENCODER_ID, ("B4", "B6", "B9", "B12"))
    with pytest.raises(ValueError):
        verify_encoder_layers("vgg16_imagenet1k_v1", ("layer1",))


def test_t5_vit_block_numbering_maps_to_encoder_layers(vit_encoder):
    modules = layer_modules(vit_encoder, VIT_ENCODER_ID)
    for index, layer in enumerate(X2_ENCODER_LAYERS[VIT_ENCODER_ID]):
        assert modules[layer] is vit_encoder.encoder.layers[int(layer[1:]) - 1]
        assert int(layer[1:]) == 3 * (index + 1)


# --- T6: L2 normalization ---------------------------------------------------
def test_t6_l2_normalization_is_correct():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(64, 32)) * 5.0
    normalized = normalize_layer_features(features)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(normalized, features / np.linalg.norm(features, axis=1, keepdims=True))
    with pytest.raises(ValueError):
        normalize_layer_features(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        bad = np.ones((3, 4))
        bad[0, 0] = np.nan
        normalize_layer_features(bad)


# --- T7 / T8: aggregation and prototypes ------------------------------------
def test_t7_client_aggregation_equals_centralized_message():
    rng = np.random.default_rng(23)
    labels = np.repeat(np.arange(5), 40)
    features = rng.normal(size=(len(labels), 16))
    clients = rng.integers(0, X2_N_CLIENTS, size=len(labels))
    clients = client_partition(labels, "t7") if len(np.unique(clients)) < 2 else clients
    aggregated, recovery = aggregate_client_messages(
        features, labels, clients, X2_N_CLIENTS, X2_AGGREGATION_TOLERANCE
    )
    assert recovery["passed"] is True
    assert recovery["counts_equal"] is True
    assert recovery["max_abs"] <= X2_AGGREGATION_TOLERANCE
    from frontier.cmr_kernel_layer import class_message

    centralized = class_message(features, labels)
    for c in centralized:
        assert aggregated[c]["count"] == centralized[c]["count"]
        assert np.allclose(
            aggregated[c]["vector_sum"], centralized[c]["vector_sum"], rtol=0.0, atol=1e-12
        )


def test_t7_phase1_client_partition_is_frozen_and_complete():
    labels = np.repeat(np.arange(4), 25)
    first = client_partition(labels, "partition_probe")
    second = client_partition(labels, "partition_probe")
    assert np.array_equal(first, second)
    assert first.shape == labels.shape
    assert int(first.min()) >= 0 and int(first.max()) < X2_N_CLIENTS
    assert int(first.min()) == 0

    other = client_partition(labels, "partition_probe_other")
    assert not np.array_equal(first, other)


def test_t7_aggregation_rejects_invalid_partition():
    features = np.ones((4, 3))
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError):
        aggregate_client_messages(features, labels, np.array([0, 1, 2]), X2_N_CLIENTS)
    with pytest.raises(ValueError):
        aggregate_client_messages(features, labels, np.array([0, 1, 2, X2_N_CLIENTS]), X2_N_CLIENTS)


def test_t8_prototype_reconstruction_is_correct():
    rng = np.random.default_rng(31)
    features = rng.normal(size=(30, 8))
    labels = np.array([0] * 10 + [1] * 20)
    from frontier.cmr_kernel_layer import class_message

    message = class_message(features, labels)
    prototypes = reconstruct_prototypes(message)
    assert set(prototypes) == {0, 1}
    assert np.allclose(prototypes[0], features[labels == 0].mean(axis=0), rtol=0.0, atol=1e-12)
    assert np.allclose(prototypes[1], features[labels == 1].mean(axis=0), rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError):
        reconstruct_prototypes({0: {"count": 0, "vector_sum": np.zeros(8)}})


# --- T9: tie rule -----------------------------------------------------------
def test_t9_classifier_tie_rule_uses_smallest_class_id():
    prototypes = {0: np.array([1.0, 0.0]), 1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    predictions = predict_prototype_batch(np.array([[1.0, 0.0]]), prototypes)
    assert predictions.tolist() == [0]

    # A missing class 0 must promote the tie to the next smallest ID.
    predictions = predict_prototype_batch(np.array([[1.0, 0.0]]), {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])})
    assert predictions.tolist() == [1]

    # Distinct scores still win over the tie rule.
    prototypes = {0: np.array([1.0, 0.0]), 1: np.array([2.0, 0.0])}
    assert predict_prototype_batch(np.array([[1.0, 0.0]]), prototypes).tolist() == [1]


def test_t9_tie_rule_is_exact_not_epsilon_based():
    prototypes = {
        3: np.array([1.0, 1.0]),
        5: np.array([1.0, 1.0 + 1e-15]),
    }
    assert predict_prototype_batch(np.array([[1.0, 1.0]]), prototypes).tolist() == [5]
    exact = {3: np.array([1.0, 1.0]), 5: np.array([1.0, 1.0])}
    assert predict_prototype_batch(np.array([[1.0, 1.0]]), exact).tolist() == [3]


# --- T10: balanced accuracy -------------------------------------------------
def test_t10_balanced_accuracy_matches_project_utility():
    from frontier.phase4_eval import balanced_accuracy as reference_bacc

    rng = np.random.default_rng(41)
    for _ in range(5):
        labels = rng.integers(0, 4, size=200)
        predictions = rng.integers(0, 4, size=200)
        mine = bacc_from_predictions(labels, predictions)
        reference = reference_bacc(labels, predictions)
        assert abs(mine - reference) < 1e-12


def test_t10_balanced_accuracy_matches_sklearn_reference():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(43)
    labels = np.repeat(np.arange(3), 50)
    predictions = rng.integers(0, 3, size=len(labels))
    assert abs(
        bacc_from_predictions(labels, predictions)
        - sklearn_metrics.balanced_accuracy_score(labels, predictions)
    ) < 1e-12


# --- T11 / T12: frozen thresholds -------------------------------------------
def _condition(dataset, encoder, bacc_map, recovery_max_abs=0.0):
    recovery = {
        "counts_equal": True,
        "max_abs": recovery_max_abs,
        "tolerance": X2_AGGREGATION_TOLERANCE,
        "passed": recovery_max_abs <= X2_AGGREGATION_TOLERANCE,
    }
    return summarize_x2_condition(dataset, encoder, bacc_map, recovery)


def test_t11_action_relevant_threshold_is_exactly_0_01():
    assert X2_ACTION_RELEVANT_SPREAD_MIN == 0.01
    just_below = _condition(
        "cifar100",
        RESNET_ENCODER_ID,
        {"layer1": 0.5, "layer2": 0.5 - 0.009999, "layer3": 0.5, "layer4": 0.5},
    )
    assert just_below["action_relevant"] is False
    exactly = _condition(
        "cifar100",
        RESNET_ENCODER_ID,
        {"layer1": 0.5, "layer2": 0.49, "layer3": 0.5, "layer4": 0.5},
    )
    assert exactly["layer_spread"] == pytest.approx(0.01, abs=1e-15)
    assert exactly["action_relevant"] is True


def test_t12_unique_best_margin_is_exactly_0_002():
    assert X2_UNIQUE_BEST_MARGIN_MIN == 0.002
    below = _condition(
        "cifar100",
        RESNET_ENCODER_ID,
        {"layer1": 0.500, "layer2": 0.4981, "layer3": 0.40, "layer4": 0.30},
    )
    assert below["best_layer"] == "layer1"
    assert below["unique_best"] is None
    exactly = _condition(
        "cifar100",
        RESNET_ENCODER_ID,
        {"layer1": 0.500, "layer2": 0.498, "layer3": 0.40, "layer4": 0.30},
    )
    assert exactly["unique_best"] == "layer1"
    # No arbitrary tie breaking: an exact tie yields no UNIQUE_BEST.
    tied = _condition(
        "cifar100",
        RESNET_ENCODER_ID,
        {"layer1": 0.50, "layer2": 0.50, "layer3": 0.40, "layer4": 0.30},
    )
    assert tied["unique_best"] is None
    assert tied["best_layer"] == "layer1"
    assert tied["second_best_layer"] == "layer2"


def test_t11_t12_summary_rejects_incomplete_action_family():
    with pytest.raises(ValueError):
        _condition(
            "cifar100",
            RESNET_ENCODER_ID,
            {"layer1": 0.5, "layer2": 0.5, "layer3": 0.5},
        )


def test_x2_gates_require_exactly_eight_conditions():
    conditions = [
        _condition(dataset, encoder, {layer: 0.5 for layer in X2_ENCODER_LAYERS[encoder]})
        for dataset, encoder in X2_CONDITIONS
    ]
    result = evaluate_x2_gates(conditions)
    assert result["all_gates_pass"] is False
    assert result["decision"] == X2_DECISION_FAIL
    assert result["final_cmr_decision"] == "CMR-D"
    assert result["gates"]["X2-G1"]["passed"] is False
    with pytest.raises(ValueError):
        evaluate_x2_gates(conditions[:-1])


def test_x2_gate_thresholds_and_pass_path():
    conditions = []
    for dataset, encoder in X2_CONDITIONS:
        layers = X2_ENCODER_LAYERS[encoder]
        # Give each dataset a different winner with a clear margin.
        offset = X2_DATASETS.index(dataset) % len(layers)
        bacc = {}
        for index, layer in enumerate(layers):
            bacc[layer] = 0.9 if index == offset else 0.5
        conditions.append(_condition(dataset, encoder, bacc))
    result = evaluate_x2_gates(conditions)
    assert result["gates"]["X2-G1"]["action_relevant_count"] == 8
    assert result["gates"]["X2-G1"]["passed"] is True
    assert result["gates"]["X2-G2"]["passed"] is True
    assert result["gates"]["X2-G3"]["passed"] is True
    assert result["gates"]["X2-G4"]["passed"] is True
    assert result["gates"]["X2-G5"]["passed"] is True
    assert result["all_gates_pass"] is True
    assert result["decision"] == X2_DECISION_PASS
    assert result["final_cmr_decision"] is None


def test_x2_gate_g4_and_g5_fail_closed():
    conditions = []
    for dataset, encoder in X2_CONDITIONS:
        layers = X2_ENCODER_LAYERS[encoder]
        bacc = {layer: (0.9 if index == 0 else 0.5) for index, layer in enumerate(layers)}
        conditions.append(_condition(dataset, encoder, bacc, recovery_max_abs=1e-6))
    result = evaluate_x2_gates(conditions)
    assert result["gates"]["X2-G4"]["passed"] is False
    assert result["all_gates_pass"] is False

    conditions = []
    for dataset, encoder in X2_CONDITIONS:
        layers = X2_ENCODER_LAYERS[encoder]
        bacc = {layer: (0.9 if index == 0 else 0.5) for index, layer in enumerate(layers)}
        conditions.append(_condition(dataset, encoder, bacc))
    conditions[0]["finite_pass"] = False
    result = evaluate_x2_gates(conditions)
    assert result["gates"]["X2-G5"]["passed"] is False
    assert result["decision"] == X2_DECISION_FAIL


# --- T13: protocol unchanged ------------------------------------------------
def test_t13_protocol_identity_and_hashes_unchanged():
    assert PROTOCOL_ID == "CMR-V1"
    assert PROTOCOL_VERSION == "1.0.0-FROZEN"
    assert PROTOCOL_STATUS == "FROZEN_BEFORE_CMR_OUTCOME_ACCESS"
    assert sha256_file(PROTOCOL_YAML) == FROZEN_PROTOCOL_YAML_SHA256
    assert sha256_file(PROTOCOL_MD) == FROZEN_PROTOCOL_MD_SHA256
    assert sha256_file(GATE_MATRIX) == FROZEN_GATE_MATRIX_SHA256
    authority = load_x2_protocol_constants(PROTOCOL_YAML)
    assert authority["protocol_status"] == PROTOCOL_STATUS
    assert authority["protocol_version"] == PROTOCOL_VERSION
    assert tuple(authority["datasets"]) == X2_DATASETS
    assert {k: tuple(v) for k, v in authority["encoders"].items()} == X2_ENCODER_LAYERS


def test_t13_protocol_constants_drift_is_detected(tmp_path):
    document = PROTOCOL_YAML.read_text(encoding="utf-8")
    tampered = document.replace("sigma: 1.0", "sigma: 2.0")
    assert tampered != document
    path = tmp_path / "tampered.yaml"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_x2_protocol_constants(path)

    tampered = document.replace("dimension: 256", "dimension: 512")
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_x2_protocol_constants(path)

    tampered = document.replace("action_relevant_spread_min: 0.01", "action_relevant_spread_min: 0.005")
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_x2_protocol_constants(path)


def test_t13_x1_evidence_still_declares_pass():
    gate_summary = json.loads(
        (REPO_ROOT / "evidence/cmr_v1/x1/gate_summary.json").read_text(encoding="utf-8")
    )
    assert gate_summary["all_gates_pass"] is True
    assert gate_summary["decision"] == "X1_PASS_EXACT_KERNEL_PROTOTYPE_LAYER_COLLISION"
    assert gate_summary["protocol_status"] == PROTOCOL_STATUS


# --- T14: no X3-X5 artifacts ------------------------------------------------
# Stage-transition note (recorded, not a protocol change): before the CMR-D
# closeout, ``evidence/cmr_v1/final_gate_summary.json`` was listed here because
# only a full CMR-A/B/C run could produce it. X2 failed its frozen X2-G2 gate and
# the protocol stopped at CMR-D, whose closeout requires exactly that artifact, so
# it is no longer an X3-X5 stage marker. The X3, X4 and X5 prohibitions are
# unchanged and still unconditional.
FORBIDDEN_PATHS = (
    "configs/cmr_v1_wilds_pair_manifest.json",
    "configs/cmr_v1_wilds_unblind_authorization.json",
    "configs/cmr_v1_external_raw_manifest.json",
    "configs/cmr_v1_external_pair_manifest.json",
    "configs/cmr_v1_external_unblind_authorization.json",
    "runs/cmr_v1/x3",
    "runs/cmr_v1/x4",
    "runs/cmr_v1/x5",
    "evidence/cmr_v1/x3",
    "evidence/cmr_v1/x4",
    "evidence/cmr_v1/x5",
)


@pytest.mark.parametrize("relative", FORBIDDEN_PATHS)
def test_t14_no_x3_x4_x5_artifact_exists(relative):
    assert not (REPO_ROOT / relative).exists(), f"forbidden X3-X5 artifact present: {relative}"


def test_t14_runner_has_no_x3_x4_x5_mode():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cmr_runner_t14", REPO_ROOT / "tools/run_cmr_v1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_arg_parser()
    modes = set()
    for action in parser._actions:
        if action.dest == "mode":
            modes = set(action.choices or ())
    assert modes == {"x1", "feature-map", "extract", "x2-dry", "x2"}
    assert modes.isdisjoint({"x3", "x4", "x5"})
    assert hasattr(module, "run_x2_formal_mode")
    assert not hasattr(module, "run_x3_mode")
    assert not hasattr(module, "run_x4_mode")
    assert not hasattr(module, "run_x5_mode")
