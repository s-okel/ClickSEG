import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from isegm.data.datasets import *
from isegm.model.losses import *
from isegm.data.transforms import *
from isegm.model.metrics import AdaptiveIoU, DiceScore
from isegm.data.points_sampler import MultiPointSampler
from isegm.utils.log import logger
from isegm.model import initializer

from isegm.model.is_hrnet_model import HRNetModel
from isegm.model.is_deeplab_model import DeeplabModel
from isegm.model.is_segformer_model import SegFormerModel
from isegm.model.is_strong_baseline import BaselineModel