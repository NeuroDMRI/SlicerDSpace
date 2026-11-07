"""pytest entry: install the per-user data isolation (B2) before any test module is imported."""

import user_data_isolation

user_data_isolation.install()
