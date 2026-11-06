"""Remember unfinished preparation, never commands or authorization to execute."""

import json
import hashlib
from pathlib import Path

KIND = "slicerddsurfer_pending_setup"


def pending_path():
    from shared.user_config import default_user_config_path

    return default_user_config_path().parent / "setup-history" / "pending.json"


def pending_setup():
    try:
        path = pending_path()
        if path.is_symlink():
            return None
        record = json.loads(path.read_text())
        token = record.get("request", "")
        if record.get("kind") != KIND or not token or token in {'.', '..'} or Path(token).name != token:
            return None
        return record
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def accepted_licences():
    """Licences accepted earlier ({id: {"id", "version", "time"}}); a continuation after a restart asks nothing for them (B7)."""
    from shared.user_config import load_user_config

    try:
        return load_user_config()["accepted_licences"]
    except (OSError, ValueError):
        return {}


def remember_setup(request):
    from shared.managed_runtimes import write_json

    if not request or request in {'.', '..'} or Path(request).name != request:
        raise ValueError("Invalid setup request identifier")
    path = pending_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"kind": KIND, "request": request, "status": "needs_preparation",
                      "policy": "Recheck current selections and ask confirmation; never resume scientific analysis automatically"})


def complete_setup(request):
    record = pending_setup()
    if record and record["request"] == request:
        pending_path().unlink(missing_ok=True)


def remember_functional_failure(request, signature):
    from shared.managed_runtimes import write_json
    remember_setup(request)
    record = pending_setup()
    record.update(status='functional_failed', signature=_signature_hash(signature))
    write_json(pending_path(), record)


def _signature_hash(signature):
    return hashlib.sha256(json.dumps(signature, default=str, sort_keys=True).encode()).hexdigest()


def functional_failure_matches(signature):
    record = pending_setup()
    return bool(record and record.get('status') == 'functional_failed' and
                record.get('signature') == _signature_hash(signature))


def refresh_setup_button(owner):
    from shared.ui_onboarding import onboarding_text

    button = getattr(owner, "_managedRuntimeSetupButton", None)
    if button is None:
        return
    key = "managed_runtime_resume" if pending_setup() else "managed_runtime_setup"
    bindings = getattr(owner, "_uiLanguageBindings", [])
    owner._uiLanguageBindings = [(widget, prop, key if widget is button and prop == "text" else text)
                                 for widget, prop, text in bindings]
    button.setText(onboarding_text(getattr(owner, "_onboardingLanguage", "en"), key))


def offer_pending_preparation(owner):
    """Recheck after restart; installation still requires the normal confirmation."""
    record = pending_setup()
    if not record or getattr(owner, '_setupResumeOffered', None) == record['request']:
        return False
    if getattr(owner, '_pipelineActive', False) or getattr(owner, '_selectedSetupActive', False):
        return False
    owner._setupResumeOffered = record['request']
    import qt
    from shared.ui_environment_check import request_prepare_environment

    def resume():
        if not getattr(owner, '_setupResumeEnabled', True):
            owner._setupResumeOffered = None
            return
        if getattr(owner, '_pipelineActive', False) or getattr(owner, '_setupJob', None) or getattr(owner, '_selectedSetupActive', False):
            owner._setupResumeOffered = None
            return
        current = pending_setup()
        if current and current['request'] == record['request']:
            request_prepare_environment(owner)

    qt.QTimer.singleShot(0, resume)
    return True
