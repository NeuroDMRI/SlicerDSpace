"""Compact, resolution-independent module branding."""

from pathlib import Path


def install_header_wordmark(owner, *, dark=None):
    import qt

    existing = getattr(owner, '_headerWordmark', None)
    path = Path(owner.resourcePath('Icons/SlicerDSpace-wordmark.svg'))
    artwork = path.read_bytes()
    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode(getattr(owner, '_uiWidget', None)) if dark is None else bool(dark)
    if use_dark:
        artwork = (
            artwork.replace(b'fill="#000000"', b'fill="#f5f3ee"')
            .replace(b'fill="#787f84"', b'fill="#95abb8"')
        )
    else:
        artwork = artwork.replace(b'fill="#000000"', b'fill="#18222c"')
    wordmark = existing if existing is not None else qt.QLabel(owner.ui.headerFrame)
    wordmark.setObjectName('slicerDSpaceHeaderWordmark')
    # Qt's SVG image plugin is available even when PythonQt does not wrap QtSvg.
    pixmap = qt.QPixmap()
    if not pixmap.loadFromData(qt.QByteArray(artwork), 'SVG'):
        raise ValueError('Cannot decode the packaged SlicerDSpace wordmark')
    pixmap = pixmap.scaled(704, 128, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)
    pixmap.setDevicePixelRatio(4.0)
    wordmark.setPixmap(pixmap)
    wordmark.setFixedSize(176, 32)
    wordmark.setAccessibleName('SlicerDSpace')
    wordmark.setToolTip('SlicerDSpace')
    owner.ui.logoLabel.hide()
    owner.ui.headerTextWidget.hide()
    if existing is None:
        owner.ui.headerLayout.insertWidget(0, wordmark)
        owner.ui.headerLayout.insertStretch(1, 1)
    owner._headerWordmark = wordmark
    return wordmark
