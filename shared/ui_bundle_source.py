"""Batch 2 item 5: the "Tractography source" combo in the settings popup of each bundle algorithm.

The combo is built in code (a combo bound to a parameter-node Choice cannot carry translated captions). Each entry
stores the source id as item data; the captions are built in English or Chinese when the popup opens. The value goes
to the parameter node only when the user clicks Apply (``staged``), like the other popup settings.
"""

from __future__ import annotations

from shared.bundle_sources import CHOICES, NODE_ATTRIBUTES, normalize_bundle_sources

LABEL = ("Tractography source", "纤维追踪来源")
ALL_TICKED_DEFAULT = ("All ticked tractography (default)", "全部已选纤维追踪（默认）")
ALL_TICKED = ("All ticked tractography", "全部已选纤维追踪")
UKF_DEFAULT = ("UKF (default)", "UKF（默认）")
NAMES = {"csd_act": ("CSD/ACT", "CSD/ACT"), "ukf": ("UKF", "UKF"), "dti_det": ("DTI", "DTI")}
ADDS_TRACKING = (" (adds this tracking)", "（将自动加入该追踪）")
WMA_NOT_REFERENCE = ("Not the reference route; UKF is the validated input for WMA.",
                     "非参考路线；UKF 是 WMA 经过验证的输入。")
TOOLTIPS = {
    "wmql": ("Which tractography WMQL uses. The default runs WMQL on every ticked tractography. "
             "A tractography that is not ticked is added automatically.",
             "WMQL 使用哪种纤维追踪结果。默认在每个已选纤维追踪上运行 WMQL。未勾选的纤维追踪会自动加入。"),
    "tractcloud": ("Which tractography TractCloud uses. The default runs TractCloud on every ticked tractography. "
                   "A tractography that is not ticked is added automatically.",
                   "TractCloud 使用哪种纤维追踪结果。默认在每个已选纤维追踪上运行 TractCloud。未勾选的纤维追踪会自动加入。"),
    "wma": ("Which tractography WMA uses. The default is UKF, added automatically when it is not ticked.",
            "WMA 使用哪种纤维追踪结果。默认使用 UKF；未勾选时会自动加入。"),
}


def node_bundle_sources(node) -> dict:
    """``{algorithm: source}`` of the explicit (non-``auto``) values on the parameter node."""

    if node is None:
        return {}
    return normalize_bundle_sources({algorithm: getattr(node, attribute, "auto") or "auto"
                                     for algorithm, attribute in NODE_ATTRIBUTES.items()}, strict=False)


def stored_bundle_sources(sources) -> dict:
    """The map to save: every algorithm with its value, ``auto`` included.

    ``update_user_config`` merges maps key by key, so a missing key would keep an older saved value.
    """

    explicit = normalize_bundle_sources(sources if isinstance(sources, dict) else {}, strict=False)
    return {algorithm: explicit.get(algorithm, "auto") for algorithm in NODE_ATTRIBUTES}


def apply_bundle_sources_to_node(node, sources) -> None:
    """Write a saved map to the three node strings; a missing or unknown value becomes ``auto``."""

    explicit = normalize_bundle_sources(sources if isinstance(sources, dict) else {}, strict=False)
    for algorithm, attribute in NODE_ATTRIBUTES.items():
        setattr(node, attribute, explicit.get(algorithm, "auto"))


def source_entries(algorithm: str, ticked, zh: bool) -> list:
    """``[(source id, caption, tooltip)]`` in the order the combo shows them.

    ``ticked`` is the tractography selection. A single-backend entry whose backend is not ticked gets the
    "(adds this tracking)" suffix: the closure adds that tracking for this algorithm.
    """

    language = int(bool(zh))
    ticked = {str(value).strip().lower() for value in (ticked or ()) if str(value).strip()}

    def named(source):
        caption = NAMES[source][language]
        return caption + (ADDS_TRACKING[language] if source not in ticked else "")

    if algorithm == "wma":
        ukf = UKF_DEFAULT[language] + (ADDS_TRACKING[language] if "ukf" not in ticked else "")
        warning = WMA_NOT_REFERENCE[language]
        return [("auto", ukf, TOOLTIPS["wma"][language]),
                ("csd_act", named("csd_act"), warning),
                ("dti_det", named("dti_det"), warning),
                ("all", ALL_TICKED[language], warning)]
    entries = [("auto", ALL_TICKED_DEFAULT[language], TOOLTIPS[algorithm][language])]
    entries.extend((source, named(source), TOOLTIPS[algorithm][language])
                   for source in CHOICES[algorithm] if source != "auto")
    return entries


def add_staged_source_field(qt, form, node, algorithm: str, zh: bool, ticked, dark=None):
    """Add the combo to ``form`` and return the staged getter of the node string."""

    attribute = NODE_ATTRIBUTES[algorithm]
    current = node_bundle_sources(node).get(algorithm, "auto")
    combo = qt.QComboBox()
    combo.setObjectName("bundleSource_" + algorithm)
    tooltip_role = getattr(qt.Qt, "ToolTipRole", None)
    selected_index = 0
    for index, (source, caption, tooltip) in enumerate(source_entries(algorithm, ticked, zh)):
        combo.addItem(caption, source)
        if tooltip_role is not None and hasattr(combo, "setItemData"):
            combo.setItemData(index, tooltip, tooltip_role)
        if source == current:
            selected_index = index
    combo.setCurrentIndex(selected_index)
    combo.setToolTip(TOOLTIPS[algorithm][int(bool(zh))])
    try:
        from shared.ui_styles import style_combobox_popup

        style_combobox_popup(combo, dark=dark)
    except Exception:
        pass
    form.addRow(LABEL[int(bool(zh))], combo)

    def staged_value() -> str:
        value = combo.itemData(combo.currentIndex)
        value = str(value or "auto")
        return value if value in CHOICES[algorithm] else "auto"

    return attribute, staged_value
