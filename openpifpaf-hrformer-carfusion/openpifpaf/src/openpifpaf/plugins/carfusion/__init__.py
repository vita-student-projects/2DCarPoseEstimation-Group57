import openpifpaf

from .carfusion_kp import CarfusionKp
from .dataset import CocoDataset


def register():
    openpifpaf.DATAMODULES['carfusionkp'] = CarfusionKp