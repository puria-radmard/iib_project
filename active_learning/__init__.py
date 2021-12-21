from . import acquisition
from . import agent
from . import batch_querying
from . import beam_search
from . import selector
from . import util_classes
from . import dataset_classes
from . import semisupervision
from . import annotation_classes


def disable_tqdm():
    for mdl in [
        acquisition,
        agent,
        batch_querying,
        beam_search,
        selector,
        util_classes,
        dataset_classes,
        semisupervision,
        annotation_classes
    ]:
        mdl.TQDM_MODE = False
