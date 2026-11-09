"""Arrange existing, connected controls into a compact Slicer workflow."""

from __future__ import annotations

from typing import Any


def compact_algorithm_labels(node: Any) -> dict[str, str]:
    """Describe the configured algorithms, not execution or QC status."""

    from shared.ui_run_controller import microstructure_optional_stage_flags
    from shared.user_config import RUN_DEFAULTS
    from shared.algorithm_selection import algorithms, BUNDLES, LABELS

    freewater, noddi, dki = microstructure_optional_stage_flags(
        getattr(node, "orchestrationMicrostructureStages", ""),
        legacy_freewater=bool(getattr(node, "orchestrationRunFreeWater", False)),
        legacy_noddi=bool(getattr(node, "orchestrationRunNoddi", False)),
        legacy_dki=bool(getattr(node, "orchestrationRunDki", False)),
    )
    models = ["DTI"]
    if freewater:
        models.append("FW")
    if noddi:
        models.append("NODDI")
    if dki:
        method = (getattr(node, "orchestrationDkiFitMethod", "") or RUN_DEFAULTS["dki_fit_method"]).upper()
        models.append({"MKCURVE": "MK-Curve", "MKCURVE_MATLAB": "MK-Curve (MATLAB)"}
                      .get(method, f"DKI ({method})"))
    from shared.ui_run_controller import microstructure_ifw_enabled
    if microstructure_ifw_enabled(getattr(node, "orchestrationMicrostructureStages", ""),
                                  legacy_ifw=bool(getattr(node, "orchestrationRunIfw", False))):
        models.append("iFW")
    tractography = (getattr(node, "orchestrationTractographyBackend", "")
                    or RUN_DEFAULTS["tractography_backend"]).strip().lower()
    bundles = (getattr(node, "orchestrationBundleBackend", "")
               or RUN_DEFAULTS["bundle_backend"]).strip().lower()
    return {
        "targetMicrostructureCheckBox": " + ".join(models),
        "targetTractographyCheckBox": {
            "csd_act": "MRtrix3 CSD / ACT", "ukf": "UKF",
            "dti_det": "DTI deterministic", "saf": "CSD + SAF",
        }.get(tractography, " + ".join(LABELS[name] for name in algorithms(tractography))),
        "targetBundlesCheckBox": {
            "wmql": "WMQL", "wma": "WMA", "tractcloud": "TractCloud",
            "all": "WMQL + WMA + TractCloud",
        }.get(bundles, " + ".join(LABELS[name] for name in algorithms(bundles, BUNDLES, "wmql"))),
    }


def refresh_compact_algorithms(owner: Any, language: str | None = None) -> None:
    """Refresh the three product rows after parameter or language changes."""

    from shared.ui_onboarding import onboarding_text

    ui = getattr(owner, "ui", None)
    if ui is None:
        return
    language = language or getattr(owner, "_onboardingLanguage", "en")
    for name, algorithms in compact_algorithm_labels(getattr(owner, "_parameterNode", None)).items():
        control = getattr(ui, name, None)
        if control is not None:
            text = onboarding_text(language, "compact_algorithm_selection").format(algorithms=algorithms)
            summary = getattr(owner, "_algorithmSummaryLabels", {}).get(name)
            if summary is None:
                control.setText(text)
            else:
                control.setText("")
                control.setAccessibleName(text)
            control.setToolTip(onboarding_text(language, "compact_algorithm_tooltip").format(algorithms=algorithms))
            if summary is not None:
                summary.setText(algorithms)
                summary.setToolTip(control.toolTip)
    from shared.ui_algorithm_picker import refresh_picker_language
    refresh_picker_language(owner)


def install_compact_layout(owner: Any) -> None:
    """Move secondary controls, not their state or signals, into Advanced."""

    if getattr(owner, "_advancedSettingsPanel", None) is not None:
        return
    import ctk
    import qt

    ui = owner.ui
    root = owner._uiWidget
    root.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)
    # Content stays at the top through trailing stretches, not setAlignment(AlignTop): an aligned
    # layout keeps its size hint as height, which leaves out wrapped lines (height-for-width), so
    # rows were clipped in narrow docks.
    owner.layout.addStretch(1)
    owner.layout.setSizeConstraint(qt.QLayout.SetMinAndMaxSize)
    layout = root.layout()
    layout.setContentsMargins(6, 4, 6, 6)
    layout.setSpacing(6)
    layout.setSizeConstraint(qt.QLayout.SetMinAndMaxSize)
    ui.headerLayout.setContentsMargins(0, 2, 0, 5)
    ui.headerTextLayout.setContentsMargins(0, 0, 0, 0)
    from shared.ui_branding import install_header_wordmark
    install_header_wordmark(owner)
    ui.subtitleLabel.hide()

    # Reuse the selector so persistence and all translations still work.
    language = owner._onboardingWidgets["language_combo"]
    language.setObjectName("interfaceLanguageComboBox")
    language.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)
    ui.headerLayout.addWidget(language)
    language.show()
    ui.gettingStartedCollapsibleButton.hide()

    advanced = ctk.ctkCollapsibleButton(root)
    advanced.setObjectName("advancedSettingsPanel")
    advanced.collapsed = True
    advanced_layout = qt.QVBoxLayout(advanced)
    advanced_layout.setContentsMargins(8, 8, 8, 8)
    advanced_layout.setSpacing(8)
    advanced_hint = qt.QLabel(advanced)
    advanced_hint.setObjectName("advancedSettingsHintLabel")
    advanced_hint.setWordWrap(True)
    advanced_layout.addWidget(advanced_hint)
    owner._advancedSettingsHintLabel = advanced_hint
    prepare_inputs = owner._prepareInputWidgets["panel"]
    # CTK's old parent keeps a visibility event filter after reparenting.
    prepare_inputs.removeEventFilter(ui.configurationCollapsibleButton)
    ui.configurationLayout.removeWidget(prepare_inputs)
    advanced_layout.addWidget(prepare_inputs)
    prepare_inputs.show()
    prepare_inputs.collapsed = True
    for name in ("parametersCollapsibleButton", "resultsCollapsibleButton"):
        section = getattr(ui, name)
        layout.removeWidget(section)
        advanced_layout.addWidget(section)
        section.collapsed = True

    utilities = qt.QHBoxLayout()
    for name in ("diagnostic_button", "config_button"):
        button = owner._onboardingWidgets[name]
        utilities.addWidget(button)
        button.show()
    advanced_layout.addLayout(utilities)
    layout.insertWidget(layout.indexOf(ui.logCollapsibleButton), advanced)
    owner._advancedSettingsPanel = advanced
    # Dependency preparation is an independent, first-class setup section.
    environment = ui.environmentCollapsibleButton
    layout.removeWidget(environment)
    layout.insertWidget(layout.indexOf(advanced) + 1, environment)
    environment.collapsed = True
    from shared.ui_algorithm_picker import install_algorithm_pickers
    install_algorithm_pickers(owner)

    layout.removeWidget(ui.statusFrame)
    ui.statusLayout.setContentsMargins(0, 0, 0, 0)
    ui.runLayout.addWidget(ui.statusFrame)
    ui.runLayout.setContentsMargins(0, 0, 0, 0)
    ui.runLayout.setSpacing(4)
    ui.logLayout.setContentsMargins(0, 0, 0, 0)
    ui.logLayout.setSpacing(4)
    ui.logPlainTextEdit.setMinimumHeight(80)
    ui.logPlainTextEdit.setMaximumHeight(100)
    ui.logCollapsibleButton.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Maximum)
    owner._prepareInputWidgets["panel"].setStyleSheet("")
    for section in (ui.configurationCollapsibleButton, ui.targetsCollapsibleButton):
        # Preferred, not Maximum: Maximum caps a section at its size hint, which leaves out wrapped
        # form rows; the trailing stretch below keeps sections from growing beyond their content.
        section.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Preferred)
        section.layout().setContentsMargins(8, 6, 8, 6)
        section.layout().setSpacing(4)
    for form in (ui.configurationFormLayout, ui.targetProductsFormLayout):
        form.setVerticalSpacing(2)
        form.setHorizontalSpacing(10)
    ui.targetProductsFormLayout.setVerticalSpacing(4)
    owner._runStateLabel.hide()
    owner._runStatusHintLabel.hide()
    owner._runProgressBar.hide()
    owner._cancelRunButton.hide()
    owner._selectedProductsSummaryLabel.hide()
    layout.addStretch(1)

    def fit_height() -> None:
        # Slicer's scroll area keeps an old expanded height unless resized explicitly.
        try:
            layout.invalidate()
            layout.activate()
            root.resize(root.width, root.minimumSizeHint.height())
        except ValueError:
            return

    def queue_fit(*_args: Any) -> None:
        qt.QTimer.singleShot(0, fit_height)

    for section in (advanced, ui.configurationCollapsibleButton, ui.targetsCollapsibleButton,
                    ui.parametersCollapsibleButton, ui.environmentCollapsibleButton,
                    ui.resultsCollapsibleButton, ui.advancedRunCollapsibleButton,
                    owner._prepareInputWidgets["panel"]):
        section.connect("contentsCollapsed(bool)", queue_fit)
    queue_fit()


def apply_compact_language(owner: Any, language: str) -> None:
    """Keep concise main-screen labels in sync with the language system."""

    if getattr(owner, "_advancedSettingsPanel", None) is None:
        return
    from shared.ui_onboarding import onboarding_text

    owner._advancedSettingsPanel.setText(onboarding_text(language, "compact_advanced"))
    owner._advancedSettingsPanel.setToolTip(onboarding_text(language, "compact_advanced_hint"))
    advanced_hint = getattr(owner, "_advancedSettingsHintLabel", None)
    if advanced_hint is not None:
        advanced_hint.setText(onboarding_text(language, "compact_advanced_hint"))
    owner.ui.headlineLabel.setText("SlicerDSpace")
    owner._onboardingWidgets["language_combo"].setToolTip(onboarding_text(language, "language"))
    prepare_inputs = (getattr(owner, "_prepareInputWidgets", {}) or {}).get("panel")
    if prepare_inputs is not None:
        prepare_inputs.setText(onboarding_text(language, "compact_import"))
    for name, key in {
        "configurationCollapsibleButton": "section_inputs",
        "targetsCollapsibleButton": "section_targets",
        "parametersCollapsibleButton": "compact_parameters",
        "environmentCollapsibleButton": "compact_environment",
        "resultsCollapsibleButton": "compact_results",
        "runSelectedButton": "compact_run",
        "logHintLabel": "compact_log",
        "targetPreprocessingCheckBox": "compact_preprocessing",
        "targetSurfaceCheckBox": "compact_surface",
        "targetSurfacePostprocessCheckBox": "compact_postprocess",
        "targetParcellationCheckBox": "compact_parcellation",
        "targetSegmentationCheckBox": "compact_segmentation",
        "targetMicrostructureCheckBox": "compact_microstructure",
        "targetTractographyCheckBox": "compact_tractography",
        "targetBundlesCheckBox": "compact_bundles",
        "targetConnectomeCheckBox": "compact_connectome",
    }.items():
        getattr(owner.ui, name).setText(onboarding_text(language, key))
    refresh_compact_algorithms(owner, language)
    from shared.ui_advanced_layout import apply_advanced_language
    apply_advanced_language(owner)
