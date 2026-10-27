import torch
import torch.nn as nn
from .backbones.vit_pytorch import vit_base_in, vit_ics_lup

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.cfg = cfg.clone()
        self.modalities = tuple(cfg.DATASETS.MODALITIES)

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base=factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                        scsr_cfg=cfg.SCSR_V41, modalities=self.modalities)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained model......from {}'.format(model_path))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
    def forward(self, x=None, label=None, camids=None, mode=0, modal_ids=None):
        if mode==0:
            if self.base.scsr_v41 is not None:
                if not isinstance(x, (tuple, list)) or len(x) != len(self.modalities):
                    raise ValueError('PKM groups must follow DATASETS.MODALITIES without reordering')
                sizes = [images.shape[0] for images in x]
                if len(set(sizes)) != 1 or (label is not None and len(label) != sizes[0]):
                    raise ValueError('PKM requires equally sized, identity-aligned modality groups')
                expected = torch.cat([torch.full((size,), index + 1, dtype=torch.long, device=x[0].device)
                                      for index, size in enumerate(sizes)])
                if modal_ids is not None and not torch.equal(modal_ids, expected):
                    raise ValueError('Training sensor IDs contradict PKM modality order')
                modal_ids = expected
            x = torch.cat(list(x), dim=0)
            global_feat = self.base(x, modal_label=modal_ids)
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return cls_score, global_feat, feat
        else:
            global_feat = self.base(x, modal_label=modal_ids)
            feat = self.bottleneck(global_feat)
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat


    def load_param(self, trained_path):
        if self.base.scsr_v41 is not None:
            from utils.scsr_v41_checkpoint import load_v41
            load_v41(self, trained_path, include_classifier=False)
            return
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        if self.base.scsr_v41 is not None:
            from utils.scsr_v41_checkpoint import load_v41
            load_v41(self, model_path, include_classifier=True)
            return
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_in': vit_base_in,
    'vit_ics_lup': vit_ics_lup,
}

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model
