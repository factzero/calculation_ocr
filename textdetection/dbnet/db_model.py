# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone_resnet import deformable_resnet18
from db_det_segout import SegDetector


class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.backbone = deformable_resnet18()
        self.decoder = SegDetector(adaptive=True, in_channels=[64, 128, 256, 512], k=50)

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return nn.DataParallel(model)


class SegDetectorModel(nn.Module):
    def __init__(self, device, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        self.model = BasicModel('')
        self.device = device

    def forward(self, batch, training=True):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()
        pred = self.model(data, training=self.training)

        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred


if __name__ == "__main__":
    model_path = "./checkpoints/model_epoch_648_minibatch_651000"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SegDetectorModel(device=device)
    print('loading pretrained model from %s' % model_path)
    states = torch.load(model_path, map_location=device)
    model.load_state_dict(states, strict=False)