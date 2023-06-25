# Copyright (c) OpenMMLab. All rights reserved.
from .props.anchor_generator import AnchorGenerator
from .props.DeltaXYWHBBoxCoder import DeltaXYWHBBoxCoder
from .models import SmoothL1Loss,CrossEntropyLoss,BoundedIoULoss,FocalLoss
from typing import List, Optional, Tuple
import numpy as np
import torch.nn as nn
from torch import Tensor
from functools import partial


from mmcv.ops import DeformConv2d, MaskedConv2d
from mmengine.model import BaseModule
from .anchor_head_template import AnchorHeadTemplate
from .MMregistries import MODELS,TASK_UTILS


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class FeatureAdaption(BaseModule):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deform conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size. Defaults to 3.
        deform_groups (int): Deformable conv group size. Defaults to 4.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or \
            list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        deform_groups: int = 4,
        init_cfg= dict(
            type='Normal',
            layer='Conv2d',
            std=0.1,
            override=dict(type='Normal', name='conv_adaption', std=0.01))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            2, deform_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, shape: Tensor) -> Tensor:
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x

class GuidedAnchorHead(AnchorHeadTemplate):
    
    def __init__(self,model_cfg,input_channels,num_class,class_names,grid_size,point_cloud_range,predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        print(f'******************************* Called GUIDED****************!')
        approx_anchor_generator,square_anchor_generator,anchor_coder,bbox_coder,loss_loc,loss_cls,loss_shape,loss_bbox,reg_decoded_bbox=self.__initparameters()    
        self.in_channels = input_channels
        self.num_classes = num_class
        self.feat_channels = 256
        self.deform_groups = 4
        self.loc_filter_thr = .01

        # build approx_anchor_generator and square_anchor_generator
        # assert (approx_anchor_generator['octave_base_scale'] ==
        #         square_anchor_generator['scales'][0])
        # assert (approx_anchor_generator['strides'] ==
        #         square_anchor_generator['strides'])
        self.approx_anchor_generator = AnchorGenerator(approx_anchor_generator)
        self.square_anchor_generator = AnchorGenerator(square_anchor_generator)
        self.approxs_per_octave = self.approx_anchor_generator.num_base_priors[0]

        self.reg_decoded_bbox = reg_decoded_bbox

        # one anchor per location
        self.num_base_priors = self.square_anchor_generator.num_base_priors[0]

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.loc_focal_loss = loss_loc['type'] in ['FocalLoss']
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        # build bbox_coder
        self.anchor_coder = DeltaXYWHBBoxCoder(anchor_coder)
        self.bbox_coder = DeltaXYWHBBoxCoder(bbox_coder)

        # build losses
        self.loss_loc = FocalLoss(loss_loc)
        self.loss_shape = BoundedIoULoss(loss_shape)
        self.loss_cls = CrossEntropyLoss(loss_cls)
        self.loss_bbox = SmoothL1Loss(loss_bbox)

        self.train_cfg = None
        self.test_cfg = None

        if self.train_cfg:
            pass

        self._init_layers()
    def __initparameters(self):
        approx_anchor_generator = dict(
            type='AnchorGenerator',
            octave_base_scale=8,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64])
        square_anchor_generator = dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[8],
            strides=[4, 8, 16, 32, 64])
        anchor_coder = dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0])
        
        bbox_coder = dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0])
        loss_loc=dict(type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)
        loss_cls = dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
        
        loss_shape= dict(
            type='BoundedIoULoss', beta=0.2, loss_weight=1.0)
        
        loss_bbox= dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        reg_decoded_bbox=False

        return approx_anchor_generator,square_anchor_generator,anchor_coder,bbox_coder,loss_loc,loss_cls,loss_shape,loss_bbox,reg_decoded_bbox

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.conv_loc = nn.Conv2d(self.in_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.in_channels, self.num_base_priors * 2,
                                    1)
        self.feature_adaption = FeatureAdaption(
            self.in_channels,
            self.feat_channels,
            kernel_size=3,
            deform_groups=self.deform_groups)
        print(self.num_anchors_per_location*self.num_class)
        self.conv_cls = MaskedConv2d(
            self.feat_channels, self.num_anchors_per_location * self.num_class,
            1)
        self.conv_reg = MaskedConv2d(self.feat_channels,
                                     self.num_base_priors * 4, 1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor]:
        """Forward feature of a single scale level."""
        loc_pred = self.conv_loc(x)
        shape_pred = self.conv_shape(x)
        x = self.feature_adaption(x, shape_pred)
        # masked conv is only used during inference for speed-up
        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.conv_cls(x, mask)
        bbox_pred = self.conv_reg(x, mask)
        return cls_score, bbox_pred, shape_pred, loc_pred

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network."""
        tensors=x['spatial_features_2d']
        cls_score,bbox_pred,shape_pred,loc_pred=self.forward_single(tensors)
        
        cls_score = cls_score.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        print(cls_score.shape)
        print(bbox_pred.shape)
        print(shape_pred.shape)
        print(loc_pred.shape)

        x['cls_preds_normalized'] = False

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=x['batch_size'],
                cls_preds=cls_score, box_preds=bbox_pred, dir_cls_preds=None
            )
        
        x['batch_cls_preds']= batch_cls_preds
        x['batch_box_preds']= batch_box_preds
        return x






        return self.forward_single(x)
