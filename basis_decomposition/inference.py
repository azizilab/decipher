from enum import Enum

import pyro


class InferenceMode(Enum):
    POINT_ESTIMATE = 0
    GAUSSIAN = 1
    GAUSSIAN_BETA_ONLY = 2
    GAUSSIAN_BASIS_ONLY = 3


def get_inference_guide(model, inference_mode):
    # define the variational inference method
    # 1) All with uncertainty estimation (gaussian)
    if inference_mode == InferenceMode.GAUSSIAN:
        guide = pyro.infer.autoguide.AutoNormal(model)

    # 2) beta only have uncertainty
    elif inference_mode == InferenceMode.GAUSSIAN_BETA_ONLY:
        guide = pyro.infer.autoguide.AutoGuideList(model)
        guide.append(
            pyro.infer.autoguide.AutoDelta(
                pyro.poutine.block(model, hide=["beta"]),  # init_scale=1.
            )
        )
        guide.append(
            pyro.infer.autoguide.AutoNormal(pyro.poutine.block(model, expose=["beta"]), init_scale=0.2)
        )

    # 3) basis only have uncertainty
    elif inference_mode == InferenceMode.GAUSSIAN_BASIS_ONLY:
        guide = pyro.infer.autoguide.AutoGuideList(model)
        guide.append(
            pyro.infer.autoguide.AutoNormal(
                pyro.poutine.block(model, hide=["beta"]),
            )
        )
        guide.append(pyro.infer.autoguide.AutoDelta(pyro.poutine.block(model, expose=["beta"])))

    # 4) no uncertainty
    elif inference_mode == InferenceMode.POINT_ESTIMATE:
        guide = pyro.infer.autoguide.AutoDelta(model)

    return guide
