import torch
import torch.nn as nn
import os
import yaml
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
        self.modalities = tuple(cfg.DATASETS.MODALITIES)
        self.scsr_enabled = cfg.SCSR.ENABLED
        self.experiment_cfg = cfg.clone()
        target_id = None
        if self.scsr_enabled:
            if len(set(self.modalities)) != len(self.modalities):
                raise ValueError('DATASETS.MODALITIES must contain unique names')
            if cfg.SCSR.TARGET_MODALITY not in self.modalities:
                raise ValueError('SCSR.TARGET_MODALITY must appear in DATASETS.MODALITIES')
            target_id = self.modalities.index(cfg.SCSR.TARGET_MODALITY) + 1

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
                                                        scsr_cfg=cfg.SCSR, scsr_target_id=target_id,
                                                        num_modalities=len(self.modalities))

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
            if self.scsr_enabled:
                if not isinstance(x, (list, tuple)) or len(x) != len(self.modalities):
                    raise ValueError('Training requires one image batch per configured modality')
                if len({batch.shape[0] for batch in x}) != 1:
                    raise ValueError('PKM modality batches must have equal sizes for UAD losses')
                if modal_ids is not None:
                    raise ValueError('Training modal_ids are derived from the PKM modality order')
                modal_ids = torch.cat([
                    torch.full((batch.shape[0],), i, dtype=torch.long, device=batch.device)
                    for i, batch in enumerate(x, start=1)])
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
        self._check_checkpoint_config(trained_path)
        try:
            param_dict = torch.load(trained_path, map_location='cpu', weights_only=True)
        except TypeError:
            param_dict = torch.load(trained_path, map_location='cpu')
        param_dict = {k.replace('module.', ''): v for k, v in param_dict.items()}
        self._check_scsr_weights(param_dict)
        if self.scsr_enabled:
            state = {k: v for k, v in param_dict.items() if 'classifier' not in k}
            result = self.load_state_dict(state, strict=False)
            if set(result.missing_keys) != {'classifier.weight'} or result.unexpected_keys:
                raise ValueError('Incomplete or incompatible SCSR checkpoint: {}'.format(result))
            print('Loading SCSR model from {}'.format(trained_path))
            return
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        self._check_checkpoint_config(model_path)
        try:
            param_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        except TypeError:
            param_dict = torch.load(model_path, map_location='cpu')
        self._check_scsr_weights(param_dict)
        if self.scsr_enabled:
            self.load_state_dict(param_dict, strict=True)
            return
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def _check_scsr_weights(self, state):
        found = {k for k in state if k.startswith('base.scsr.')}
        expected = {k for k in self.state_dict() if k.startswith('base.scsr.')}
        if found != expected:
            raise ValueError('SCSR checkpoint/architecture mismatch; use the matching experiment config')

    def _check_checkpoint_config(self, path):
        if not self.scsr_enabled:
            return
        sidecar = str(path) + '.yml'
        if not os.path.isfile(sidecar):
            raise ValueError('SCSR checkpoint requires its saved configuration: ' + sidecar)
        with open(sidecar, encoding='utf-8') as stream:
            saved = yaml.safe_load(stream)
        current = yaml.safe_load(self.experiment_cfg.dump())
        checks = [('SCSR', k) for k in current['SCSR'] if k != 'LR_MULTIPLIER']
        checks += [('DATASETS', 'MODALITIES'), ('MODEL', 'TRANSFORMER_TYPE'),
                   ('MODEL', 'STRIDE_SIZE'), ('INPUT', 'SIZE_TRAIN'),
                   ('INPUT', 'SIZE_TEST'), ('MODEL', 'SIE_CAMERA'), ('MODEL', 'SIE_VIEW'),
                   ('MODEL', 'SIE_COE')]
        for section, key in checks:
            if saved.get(section, {}).get(key) != current[section][key]:
                raise ValueError('Checkpoint config mismatch: {}.{}'.format(section, key))


__factory_T_type = {
    'vit_base_in': vit_base_in,
    'vit_ics_lup': vit_ics_lup,
}

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model
