from __future__ import annotations

import sys
from typing import Any

import numpy as np


class XGBClassifierWithLabels:
    """Compatibility shell for Stable-FCI's saved notebook-defined wrapper.

    The saved object was pickled as __main__.XGBClassifierWithLabels. This class
    is registered under that name before joblib.load(). The fitted XGBoost model
    and its original class order are retained; no estimator is fitted here.
    """

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._model.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        class_values = np.asarray(self.classes_)
        return class_values[np.argmax(probabilities, axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


def register_stable_fci_wrapper_for_unpickling() -> None:
    """Make the expected __main__ symbol available to the pickle loader."""
    main_module = sys.modules.get("__main__")
    if main_module is None:
        raise RuntimeError("Could not access __main__ while registering the Stable-FCI XGBoost wrapper.")
    setattr(main_module, "XGBClassifierWithLabels", XGBClassifierWithLabels)
