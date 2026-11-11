"""A small category selector replaces deeply nested parameter accordions."""

CATEGORIES = (("processing", "Preprocessing & mask", "预处理与掩膜"),
              ("models", "Model parameters", "模型参数"),
              ("execution", "Run & performance", "运行与性能"),
              ("tools", "Regions & input import", "脑区与输入导入"),
              ("experiments", "Experiments & batch", "实验与批处理"))

BATCH_CONTROLS = ("batchModeLabel", "batchModeCheckBox", "batchRootLabel", "batchRootPath",
                  "batchSizeLabel", "batchSizeSpinBox", "batchArrangeInputsLabel", "batchArrangeInputsComboBox")


def enforce_single_subject_gui(node):
    """Migrate legacy GUI state; cohort scheduling belongs to the batch CLI."""
    if node is not None:
        if getattr(node, "batchMode", False):
            node.batchMode = False
        if getattr(node, "batchSize", 1) != 1:
            node.batchSize = 1


def hide_batch_controls(owner):
    for name in BATCH_CONTROLS:
        widget = getattr(owner.ui, name, None)
        if widget is not None:
            widget.hide()


def _move(owner, name, form):
    widget = getattr(owner.ui, name)
    for source in (owner.ui.parametersFormLayout, owner.ui.advancedRunFormLayout, owner.ui.environmentFormLayout):
        if source.indexOf(widget) < 0:
            continue
        label = source.labelForField(widget)
        for item in (label, widget):
            if item is not None:
                item.removeEventFilter(owner.ui.parametersCollapsibleButton)
                item.removeEventFilter(owner.ui.advancedRunCollapsibleButton)
                source.takeAt(source.indexOf(item))
        if label:
            form.addRow(label, widget)
        else:
            form.addRow(widget)
        widget.show()
        return


def _unpack_brain_mask_rows(owner, form):
    ui = owner.ui
    options = getattr(ui, "brainMaskOptions", None)
    options_layout = getattr(ui, "brainMaskOptionsLayout", None)
    params_form = getattr(ui, "parametersFormLayout", None)
    if options is None or params_form is None:
        _move(owner, "brainMaskOptions", form)
        return
    backend_label = getattr(ui, "brainMaskBackendLabel", None)
    for item in (backend_label, options):
        if item is not None:
            item.removeEventFilter(ui.parametersCollapsibleButton)
            if params_form.indexOf(item) >= 0:
                params_form.takeAt(params_form.indexOf(item))
    pairs = (
        ("brainMaskBackendLabel", "brainMaskBackendComboBox"),
        ("brainMaskDeviceLabel", "brainMaskDeviceComboBox"),
        ("pnlMaskRouteLabel", "pnlMaskRouteComboBox"),
    )
    for label_name, field_name in pairs:
        lbl = getattr(ui, label_name, None)
        fld = getattr(ui, field_name, None)
        if fld is None:
            continue
        for w in (lbl, fld):
            if w is not None:
                w.removeEventFilter(ui.parametersCollapsibleButton)
                if options_layout is not None and options_layout.indexOf(w) >= 0:
                    options_layout.removeWidget(w)
        if lbl is not None:
            lbl.show()
            fld.show()
            form.addRow(lbl, fld)
        else:
            fld.show()
            form.addRow(fld)
    options.hide()


def install_advanced_categories(owner):
    import qt
    root = owner._advancedSettingsPanel
    layout = root.layout()
    selector = qt.QComboBox()
    selector.objectName = "advancedCategoryComboBox"
    stack = qt.QStackedWidget()
    stack.objectName = "advancedCategoryStack"
    stack.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Maximum)
    stack.setMinimumWidth(0)
    owner._advancedCategorySelector, owner._advancedCategoryStack = selector, stack
    owner._advancedCategoryForms = {}
    for key, _, _ in CATEGORIES:
        page = qt.QWidget()
        page.objectName = "advancedPage_" + key
        page.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Maximum)
        page_layout = qt.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 4, 0, 0)
        page_layout.setSpacing(6)
        page_layout.setAlignment(qt.Qt.AlignTop)
        form = qt.QFormLayout()
        form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(qt.QFormLayout.WrapLongRows)
        form.setVerticalSpacing(6)
        form.setHorizontalSpacing(10)
        page_layout.addLayout(form)
        owner._advancedCategoryForms[key] = form
        stack.addWidget(page)
    layout.insertWidget(1, selector)
    layout.insertWidget(2, stack)
    _unpack_brain_mask_rows(owner, owner._advancedCategoryForms["processing"])
    rows = {
        "models": ("orchestrationDkiFitMethodComboBox", "orchestrationIfwEchoTimeLineEdit", "orchestrationIfwT2TissueLineEdit", "orchestrationIfwT2CsfLineEdit", "wmqlQueryPresetComboBox", "wmqlQuerySelectionLineEdit", "wmaAtlasComboBox", "wmaInputTractographyPath", "orchestrationSelectStreamlinesSpinBox"),
        "execution": ("orchestrationRunNameLineEdit", "parallelModelsSpinBox"),
    }
    for key, names in rows.items():
        for name in names:
            _move(owner, name, owner._advancedCategoryForms[key])
    owner.ui.parametersFormLayout.takeAt(owner.ui.parametersFormLayout.indexOf(owner.ui.topupCollapsibleGroupBox))
    stack.widget(0).layout().addWidget(owner.ui.topupCollapsibleGroupBox)
    for key, panel in (("tools", owner._prepareInputWidgets["panel"]), ("tools", owner._regionWidgets["panel"])):
        layout.removeWidget(panel)
        stack.widget([item[0] for item in CATEGORIES].index(key)).layout().addWidget(panel)
        panel.collapsed = True
    owner.ui.parametersCollapsibleButton.hide()
    layout.removeWidget(owner.ui.parametersCollapsibleButton)
    if hasattr(owner, "_ukfGroup"):
        owner.ui.parametersLayout.removeWidget(owner._ukfGroup)
        stack.widget(1).layout().addWidget(owner._ukfGroup)
    for name in ("orchestrationTractographyBackendLabel", "orchestrationTractographyBackendComboBox",
                 "orchestrationBundleBackendLabel", "orchestrationBundleBackendComboBox",
                 "orchestrationMicrostructureStagesLabel", "orchestrationMicrostructureStageOptions"):
        getattr(owner.ui, name).hide()
    # Scene/result controls are a separate workflow, not an algorithm parameter.
    root_layout = owner._uiWidget.layout()
    layout.removeWidget(owner.ui.resultsCollapsibleButton)
    root_layout.insertWidget(root_layout.indexOf(owner.ui.logCollapsibleButton), owner.ui.resultsCollapsibleButton)
    owner.ui.resultsCollapsibleButton.collapsed = True
    # It sat in the collapsed Advanced panel, which hid it explicitly; moving it does not show it again.
    owner.ui.resultsCollapsibleButton.show()
    selector.currentIndexChanged.connect(stack.setCurrentIndex)
    stack.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    def _sync_stack_height():
        page = stack.currentWidget()
        if page:
            for child in page.findChildren(qt.QWidget):
                child_layout = child.layout()
                if child_layout is not None:
                    child_layout.invalidate()
                    child_layout.activate()
            page.layout().invalidate()
            page.layout().activate()
            eff_width = page.width if page.width > 120 else max(120, owner._uiWidget.width - 28)
            hfw = page.layout().totalHeightForWidth(eff_width) if eff_width > 0 and page.layout().hasHeightForWidth() else 0
            measured = hfw if hfw > 0 else max(page.sizeHint.height(), page.minimumSizeHint.height())
            target_stack_height = max(44, measured + 6)
            if stack.height != target_stack_height or stack.maximumHeight != target_stack_height:
                stack.setMinimumHeight(target_stack_height)
                stack.setMaximumHeight(target_stack_height)

    owner._syncAdvancedStackHeight = _sync_stack_height

    def fit(*_):
        _sync_stack_height()
        if getattr(root, "collapsed", False):
            return
        compact_fit = getattr(owner, "_fitCompactHeight", None)
        if callable(compact_fit):
            compact_fit()
        queue_fit = getattr(owner, "_queueCompactFit", None)
        if callable(queue_fit):
            queue_fit()

    selector.currentIndexChanged.connect(fit)
    owner._regionWidgets['panel'].connect('contentsCollapsed(bool)', fit)
    owner._regionWidgets['options'].connect('contentsCollapsed(bool)', fit)
    owner.ui.topupCollapsibleGroupBox.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    owner.ui.topupCollapsibleGroupBox.setMinimumWidth(0)
    owner.ui.topupCollapsibleGroupBox.toggled.connect(fit)
    if hasattr(owner, '_ukfGroup'):
        owner._ukfGroup.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        owner._ukfGroup.setMinimumWidth(0)
        owner._ukfGroup.toggled.connect(fit)
    owner._fitAdvancedPage = fit
    from shared.ui_experiments import install
    install(owner, stack.widget(4))
    from shared.ui_compact import organize_environment_section
    organize_environment_section(owner)
    qt.QTimer.singleShot(0, fit)
    apply_advanced_language(owner)
    hide_batch_controls(owner)


def apply_advanced_language(owner):
    selector = getattr(owner, "_advancedCategorySelector", None)
    if selector is None:
        return
    zh = getattr(owner, "_onboardingLanguage", "en").startswith("zh")
    if hasattr(owner.ui, "wmaAtlasLabel"):
        owner.ui.wmaAtlasLabel.setText("WMA 图谱" if zh else "WMA atlas")
        owner.ui.wmaAtlasComboBox.setToolTip(
            "org：成人参考图谱；east-west：东西方成人；naba：新生儿与成人。在环境面板按需下载所选图谱。" if zh else
            "org: adult reference; east-west: Eastern/Western adults; naba: neonates/adults. Prepare only the selected atlas in the Environment panel.")
    previous = selector.currentIndex
    blocked = selector.blockSignals(True)
    selector.clear()
    for key, en, chinese in CATEGORIES:
        selector.addItem(chinese if zh else en, key)
    selector.setCurrentIndex(max(0, previous))
    selector.blockSignals(blocked)
    from shared.ui_experiments import apply_language
    apply_language(owner)


def refresh_advanced_categories(owner):
    hide_batch_controls(owner)
    if not hasattr(owner, "_advancedCategorySelector"):
        return
    # The old algorithm controls remain bound for compatibility, but never
    # compete with the three main-screen pickers.
    for name in ("orchestrationTractographyBackendLabel", "orchestrationTractographyBackendComboBox",
                 "orchestrationBundleBackendLabel", "orchestrationBundleBackendComboBox",
                 "orchestrationMicrostructureStagesLabel", "orchestrationMicrostructureStageOptions"):
        widget = getattr(owner.ui, name, None)
        if widget is not None and widget.visible:
            widget.hide()
    from shared.ui_experiments import sync
    sync(owner)
    if not getattr(owner._advancedSettingsPanel, "collapsed", True):
        owner._fitAdvancedPage()
