"""Verify the real Qt header, without altering the active user's Slicer."""

import json
import os
import traceback
from pathlib import Path

import qt
import slicer


def run():
    try:
        slicer.util.selectModule('SlicerDSpace')
        widget = slicer.modules.slicerdspace.widgetRepresentation().self()
        logo = widget._headerWordmark
        assert not logo.pixmap.isNull()
        assert logo.width == 176 and logo.height == 32
        assert widget.ui.logoLabel.isHidden()
        assert widget.ui.headerTextWidget.isHidden()
        assert not logo.isHidden()
        assert widget._onboardingWidgets['language_combo'].isVisible()
        image = logo.pixmap.toImage()
        dark_opaque = sum(image.pixelColor(x, y).alpha() > 200 and image.pixelColor(x, y).lightness() < 60
                          for y in range(0, image.height(), 4) for x in range(0, image.width(), 4))
        assert dark_opaque > 20, 'Light GUI must use the dark wordmark, including on transparent header frames'
        directory = Path(os.environ['SLICERDSPACE_BRANDING_EVIDENCE'])
        directory.mkdir(parents=True, exist_ok=True)
        for language in ('en', 'zh'):
            from shared.ui_compact import apply_compact_language
            apply_compact_language(widget, language)
            slicer.app.processEvents()
            assert widget.ui.logoLabel.isHidden() and widget.ui.headerTextWidget.isHidden()
        # Capture the actual Qt header, not a re-typeset approximation.
        widget.ui.headerFrame.grab().save(str(directory / 'header.png'))
        slicer.util.mainWindow().grab().save(str(directory / 'slicer.png'))
        (directory / 'receipt.json').write_text(json.dumps(dict(
            valid_vector=True, size=[logo.width, logo.height],
            legacy_logo_hidden=True, legacy_heading_hidden=True,
            language_selector_visible=True, language_switch_preserves_branding=True,
            svg_path=widget.resourcePath('Icons/SlicerDSpace-wordmark.svg'),
        ), indent=2) + '\n')
        print('SLICER_HEADER_WORDMARK_PASS', flush=True)
        slicer.app.exit(0)
    except Exception:
        traceback.print_exc()
        slicer.app.exit(1)


qt.QTimer.singleShot(4000, run)
