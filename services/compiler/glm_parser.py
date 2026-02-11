"""
Statsmodels GLM Parser

Parser for statsmodels GeneralizedLinearModel exports.
Supports all GLM families: Gaussian, Binomial, Poisson, Gamma, Tweedie.
"""

import json
import logging
from typing import Dict, List, Optional

from .ir import (
    ModelIR, ModelFamily, LinkFunction, Aggregation,
    LinearCoefficients, PreprocessingStep,
)
from .parser import BaseParser

logger = logging.getLogger(__name__)

# Map statsmodels family names to link functions
FAMILY_LINK_MAP = {
    "gaussian": LinkFunction.IDENTITY,
    "binomial": LinkFunction.LOGIT,
    "poisson": LinkFunction.LOG,
    "gamma": LinkFunction.LOG,
    "tweedie": LinkFunction.LOG,
    "inverse_gaussian": LinkFunction.RECIPROCAL,
}


class StatsmodelsGLMParser(BaseParser):
    """Parser for statsmodels GLM exported via model_export.py."""

    def parse(self, content: bytes) -> ModelIR:
        data = self._validate_content(content)

        if data.get("model_type") != "glm":
            raise ValueError(
                f"Expected model_type 'glm', got '{data.get('model_type')}'"
            )

        # Extract coefficients
        params = data["params"]
        param_names = data.get("param_names", [])

        # Separate intercept from feature coefficients
        if param_names and param_names[0] in ("const", "Intercept", "intercept"):
            intercept_val = float(params[0])
            weights = [float(p) for p in params[1:]]
            feature_names = param_names[1:] if param_names else None
        elif "intercept" in data:
            intercept_val = float(data["intercept"])
            weights = [float(p) for p in params]
            feature_names = param_names if param_names else None
        else:
            intercept_val = 0.0
            weights = [float(p) for p in params]
            feature_names = param_names if param_names else None

        num_features = len(weights)

        # Determine family and link function
        family = data.get("family", "gaussian").lower()
        link_override = data.get("link")
        if link_override:
            link_fn = LinkFunction(link_override)
        else:
            link_fn = FAMILY_LINK_MAP.get(family, LinkFunction.IDENTITY)

        # Build preprocessing
        preprocessing = []
        if "preprocessing" in data:
            for step in data["preprocessing"]:
                preprocessing.append(PreprocessingStep(
                    step_type=step["type"],
                    params=step.get("params", {}),
                    feature_indices=step.get("feature_indices"),
                ))

        logger.info(
            f"Parsed GLM: family={family}, link={link_fn.value}, "
            f"{num_features} features, intercept={intercept_val:.4f}"
        )

        return ModelIR(
            model_type="linear_glm",
            trees=[],
            num_features=num_features,
            base_score=0.0,
            model_family=ModelFamily.LINEAR,
            coefficients=LinearCoefficients(weights=weights, intercept=intercept_val),
            link_function=link_fn,
            glm_family=family,
            aggregation=Aggregation.NONE,
            feature_names=feature_names,
            preprocessing=preprocessing,
            metadata={
                "family": family,
                "aic": data.get("aic"),
                "bic": data.get("bic"),
                "deviance": data.get("deviance"),
            },
        )
