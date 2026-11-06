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
        page_layout.setContentsMargins(0, 6, 0, 0)
        page_layout.setSpacing(8)
        page_layout.setAlignment(qt.Qt.AlignTop)
        form = qt.QFormLayout()
        form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(qt.QFormLayout.WrapLongRows)
        form.setVerticalSpacing(7)
        page_layout.addLayout(form)
        owner._advancedCategoryForms[key] = form
        stack.addWidget(page)
    layout.insertWidget(1, selector)
    layout.insertWidget(2, stack)
    rows = {
        "processing": ("preprocessingBackendComboBox", "brainMaskOptions"),
        "models": ("orchestrationDkiFitMethodComboBox", "orchestrationIfwEchoTimeLineEdit", "orchestrationIfwT2TissueLineEdit", "orchestrationIfwT2CsfLineEdit", "wmqlQueryPresetComboBox", "wmqlQuerySelectionLineEdit", "wmaAtlasComboBox", "wmaInputTractographyPath", "orchestrationSelectStreamlinesSpinBox"),
        "execution": ("orchestrationRunNameLineEdit", "unifiedTorchDeviceComboBox", "parallelModelsSpinBox"),
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
    def fit(*_):
        page = stack.currentWidget()
        if page:
            page.layout().activate()
            stack.setMaximumHeight(max(60, page.sizeHint.height() + 12))
        owner._uiWidget.adjustSize()
    selector.currentIndexChanged.connect(lambda *_: qt.QTimer.singleShot(0, fit))
    for panel in (root, owner._prepareInputWidgets['panel'], owner._regionWidgets['panel']):
        panel.connect('contentsCollapsed(bool)', lambda *_: qt.QTimer.singleShot(0, fit))
    owner.ui.topupCollapsibleGroupBox.toggled.connect(lambda *_: qt.QTimer.singleShot(0, fit))
    if hasattr(owner, '_ukfGroup'):
        owner._ukfGroup.toggled.connect(lambda *_: qt.QTimer.singleShot(0, fit))
    owner._fitAdvancedPage = fit
    from shared.ui_experiments import install
    install(owner, stack.widget(4))
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
        getattr(owner.ui, name).hide()
    owner._fitAdvancedPage()
    from shared.ui_experiments import sync
    sync(owner)
