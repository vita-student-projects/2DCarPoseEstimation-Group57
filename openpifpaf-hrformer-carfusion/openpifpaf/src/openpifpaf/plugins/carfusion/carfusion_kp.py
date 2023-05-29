import argparse

import torch

import openpifpaf

from .dataset import CocoDataset
from .constants import (
    CAR_CATEGORIES,
    CARFUSION_KEYPOINTS,
    CARFUSION_SKELETON,
    CARFUSION_SIGMAS,
    CARFUSION_SCORE_WEIGHTS,
    CAR_POSE,
    CAR_POSE_RIGHT,
    CAR_POSE_FRONT,
    CAR_POSE_LEFT,
    CAR_POSE_REAR,
    HFLIP,
)

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass

import numpy as np

from openpifpaf.datasets import DataModule
from openpifpaf import encoder, headmeta, metric, transforms
from openpifpaf.datasets import collate_images_anns_meta, collate_images_targets_meta
from openpifpaf.plugins.coco import CocoDataset as CocoLoader

from .metrics import MeanPixelError


class CarfusionKp(openpifpaf.datasets.DataModule, openpifpaf.Configurable):
    # cli configurable
    train_annotations = 'openpifpaf/dataCarFusion/annotations/car_keypoints_train.json'
    val_annotations = 'openpifpaf/dataCarFusion/annotations/car_keypoints_test.json'
    eval_annotations = val_annotations
    train_image_dir = 'openpifpaf/dataCarFusion/train/'
    val_image_dir = 'openpifpaf/dataCarFusion/test/'
    eval_image_dir = val_image_dir

    square_edge = 513
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    bmin = 1

    eval_annotation_filter = True
    eval_long_edge = 0
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    skeleton = CARFUSION_SKELETON

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cif = openpifpaf.headmeta.Cif('cif', 'carfusionkp',
                                      keypoints=CARFUSION_KEYPOINTS,
                                      sigmas=CARFUSION_SIGMAS,
                                      pose=CAR_POSE,
                                      draw_skeleton=self.skeleton,
                                      score_weights=CARFUSION_SCORE_WEIGHTS)
        caf = openpifpaf.headmeta.Caf('caf', 'carfusionkp',
                                      keypoints=CARFUSION_KEYPOINTS,
                                      sigmas=CARFUSION_SIGMAS,
                                      pose=CAR_POSE,
                                      skeleton=self.skeleton)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module CocoKp')

        group.add_argument('--carfusionkp-train-annotations', default=cls.train_annotations,
                           help='train annotations')
        group.add_argument('--carfusionkp-val-annotations', default=cls.val_annotations,
                           help='val annotations')
        group.add_argument('--carfusionkp-train-image-dir', default=cls.train_image_dir,
                           help='train image dir')
        group.add_argument('--carfusionkp-val-image-dir', default=cls.val_image_dir,
                           help='val image dir')

        group.add_argument('--carfusionkp-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--carfusionkp-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--carfusionkp-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--carfusionkp-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--carfusionkp-no-augmentation',
                           dest='carfusionkp_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--carfusionkp-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--carfusionkp-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--carfusionkp-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--carfusionkp-bmin',
                           default=cls.bmin, type=float,
                           help='bmin')

        # evaluation
        #eval_set_group = group.add_mutually_exclusive_group()
        #eval_set_group.add_argument('--carfusionkp-eval-test2017', default=False, action='store_true')
        #eval_set_group.add_argument('--carfusionkp-eval-testdev2017', default=False, action='store_true')

        assert cls.eval_annotation_filter
        group.add_argument('--carfusion-no-eval-annotation-filter',
                           dest='carfusion_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--carfusion-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--carfusion-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--carfusion-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # carfusionkp specific
        cls.train_annotations = args.carfusionkp_train_annotations
        cls.val_annotations = args.carfusionkp_val_annotations
        cls.train_image_dir = args.carfusionkp_train_image_dir
        cls.val_image_dir = args.carfusionkp_val_image_dir

        cls.square_edge = args.carfusionkp_square_edge
        cls.extended_scale = args.carfusionkp_extended_scale
        cls.orientation_invariant = args.carfusionkp_orientation_invariant
        cls.blur = args.carfusionkp_blur
        cls.augmentation = args.carfusionkp_augmentation
        cls.rescale_images = args.carfusionkp_rescale_images
        cls.upsample_stride = args.carfusionkp_upsample
        cls.min_kp_anns = args.carfusionkp_min_kp_anns
        cls.bmin = args.carfusionkp_bmin

        # evaluation
        cls.eval_annotation_filter = args.carfusion_eval_annotation_filter
        #if args.cocokp_eval_test2017:
        #    cls.eval_image_dir = cls._test2017_image_dir
        #    cls.eval_annotations = cls._test2017_annotations
        #    cls.annotation_filter = False
        #if args.cocokp_eval_testdev2017:
        #    cls.eval_image_dir = cls._test2017_image_dir
        #    cls.eval_annotations = cls._testdev2017_annotations
        #    cls.annotation_filter = False
        cls.eval_long_edge = args.carfusion_eval_long_edge
        cls.eval_orientation_invariant = args.carfusion_eval_orientation_invariant
        cls.eval_extended_scale = args.carfusion_eval_extended_scale

        if (args.cocokp_eval_test2017 or args.cocokp_eval_testdev2017) \
                and not args.write_predictions and not args.debug:
            raise Exception('have to use --write-predictions for this dataset')

    def _preprocess(self):
        encoders = [openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
                    openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin)]

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.25 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.33 * self.rescale_images,
                             1.33 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(CARFUSION_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateBy90(),
                 openpifpaf.transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.2],
            ),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = openpifpaf.transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                None,
                openpifpaf.transforms.RotateBy90(fixed_angle=90),
                openpifpaf.transforms.RotateBy90(fixed_angle=180),
                openpifpaf.transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToKpAnnotations(
                    CAR_CATEGORIES,
                    keypoints_by_category={1: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[1].skeleton},
                ),
                openpifpaf.transforms.ToCrowdAnnotations(CAR_CATEGORIES),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = CocoDataset(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[1] if self.eval_annotation_filter else [],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [openpifpaf.metric.Coco(
            pycocotools.coco.COCO(self.eval_annotations),
            max_per_image=20,
            category_ids=[1],
            iou_type='keypoints',
            keypoint_oks_sigmas=CARFUSION_SIGMAS
        ), MeanPixelError()]
