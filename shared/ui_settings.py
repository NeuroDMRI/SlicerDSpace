"""User-settings helpers for the SlicerDSpace widget."""

from __future__ import annotations

from typing import Any

SETTINGS_STATE_ATTRIBUTE = 'SlicerDDSurfer.UserDefaultsState'


def mark_new_parameter_node(raw: Any) -> None:
    """Only an empty, newly created/reset MRML node may inherit user defaults."""
    if not raw.GetParameterNames():
        raw.SetAttribute(SETTINGS_STATE_ATTRIBUTE, 'pending')


def consume_user_defaults_request(node: Any) -> bool:
    """Consume the one-time request without overwriting existing/loaded scenes."""
    raw = node.parameterNode
    if raw.GetAttribute(SETTINGS_STATE_ATTRIBUTE) != 'pending':
        return False
    raw.SetAttribute(SETTINGS_STATE_ATTRIBUTE, 'applied')
    # A client may configure/save the node before the GUI first opens.
    return all(node.getValue(name) == node.default(name).value
               for name in getattr(node, 'allParameters', ()))


COMMON_USER_SETTING_SPECS: tuple[tuple[str, type], ...] = (
    ("subjectId", str),
    ("inputDirectory", str),
    ("outputDirectory", str),
    ("orchestrationRunName", str),
    ("orchestrationTractographyBackend", str),
    ("ukfParameters", str),
    ("orchestrationBundleBackend", str),
    ("orchestrationWmqlPreset", str),
    ("orchestrationWmqlSelection", str),
    ("maxParallelModels", int),
    ("orchestrationSelectStreamlines", int),
    ("orchestrationMicrostructureStages", str),
    ("orchestrationIfwEchoTimeMs", str),
    ("connectomeBackend", str),
    ("connectomeCompressMrtrixDenseState", str),
    ("autoInstallPythonDependencies", bool),
    ("batchMode", bool),
    ("batchRoot", str),
    ("batchSize", int),
    ("batchArrangeInputs", str),
    ("experimentScope", str),
    ("experimentRoot", str),
    ("experimentJobs", int),
    ("experimentCliOptions", str),
    ("experimentVariants", str),
    ("targetPreprocessing", bool),
    ("targetSurface", bool),
    ("targetSurfacePostprocess", bool),
    ("targetParcellation", bool),
    ("targetSegmentation", bool),
    ("targetMicrostructure", bool),
    ("targetTractography", bool),
    ("targetBundles", bool),
    ("targetConnectome", bool),
)


def common_user_setting_specs() -> tuple[tuple[str, type], ...]:
    """Return persisted parameter-node attributes and their value types."""

    return COMMON_USER_SETTING_SPECS


def coerce_user_setting_value(value: Any, value_type: type) -> Any:
    """Coerce a Slicer user-settings value to the expected parameter-node type."""

    if value_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if value_type is int:
        return int(value)
    return str(value or "")
