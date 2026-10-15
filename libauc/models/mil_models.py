import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

    
class FFNN(torch.nn.Module):
    r"""
    The basic 3-layer-MLP in multiple instance learning experiments utilized from [1]_.
    
    Args:
        input_dim (int, required): input data dimension.
        hidden_sizes (list[int], required): number of nurons for the hidden layers.
        num_class (int, required): number of class for model prediction, default: 1.
        last_activation (str, optional): the activation function for the output layer.
        
    Example:
        >>> model = FFNN_stoc_att(num_classes=1, dims=DIMS)

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning, pp. xxxxx-xxxxx. PMLR, 2023.
           https://arxiv.org/abs/2305.08040
    """
    def __init__(self, input_dim=29, hidden_sizes=(16,), last_activation=None, num_classes=1):
        super(FFNN, self).__init__()
        self.inputs = torch.nn.Linear(input_dim, hidden_sizes[0])
        self.last_activation = last_activation
        layers = []
        for i in range(len(hidden_sizes)-1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1])) 
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.classifer = torch.nn.Linear(hidden_sizes[-1], num_classes)
    def forward(self, x):
        x = torch.tanh(self.inputs(x))
        x = self.layers(x)
        if self.last_activation is None:
            return self.classifer(x)
        elif self.last_activation == 'sigmoid':
            return torch.sigmoid(self.classifer(x))


class FFNN_stoc_att(nn.Module):
    r"""
    The basic 3-layer-MLP with an extra attention module that generates importance weights for combining the instance-level hidden features for each bag under multiple instance learning setting [1]_.

    Args:
        input_dim (int, required): input data dimension.
        hidden_sizes (list[int], required): number of nurons for the hidden layers.
        num_class (int, required): number of class for model prediction, default: 1.

    Example:
        >>> model = FFNN_stoc_att(num_classes=1, dims=DIMS)

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning, pp. xxxxx-xxxxx. PMLR, 2023.
           https://arxiv.org/abs/2305.08040
    """
    def __init__(self, input_dim=29, hidden_sizes=(16,), num_classes=1):
        super(FFNN_stoc_att, self).__init__()
        self.inputs = torch.nn.Linear(input_dim, hidden_sizes[0])
        layers = []
        for i in range(len(hidden_sizes)-1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1])) 
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.classifer = torch.nn.Linear(hidden_sizes[-1], num_classes)
        self.attention = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[-1], 1)
        )

        self.apply(_weights_init)
        print('model initialized')


    def forward(self, x):
        x = torch.tanh(self.inputs(x))
        x = self.layers(x)
        weights = self.attention(x)
        weights = torch.exp(weights)
        out = self.classifer(x)
        return out, weights



def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight) 

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = activation_func(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = activation_func(out)
        return out

    

class ResNet_stoc_att(nn.Module):
    r"""
    The ResNet [2,3,4]_ with an extra attention module that generates importance weights for combining the instance-level hidden features for each bag under multiple instance learning setting [1]_.

    Args:
        block (torch.nn.module, required): block module for ResNet.
        num_blocks (list[int], required): number of nurons for the hidden layer. 
        inchannels (int, required): the number of channels for the input image, default: 3.
        num_classes (int, required): the model prediction class number, default: 1.

    Example:
        >>> model = ResNet_stoc_att(block=BasicBlock, num_blocks=[3,3,3], inchannels=3, num_classes=1)

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning, pp. xxxxx-xxxxx. PMLR, 2023. https://arxiv.org/abs/2305.08040
           
        .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun "Deep Residual Learning for Image Recognition." arXiv:1512.03385
           
        .. [3] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
           
        .. [4] https://github.com/akamaster/pytorch_resnet_cifar10/tree/master
    """
    def __init__(self, block, num_blocks, inchannels=3, num_classes=1):
        super(ResNet_stoc_att, self).__init__()
        self.in_planes = 16
        self.inchannels = inchannels

        self.conv1 = nn.Conv2d(inchannels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.attention = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)
        
        self.bnlast = nn.BatchNorm1d(1)

    def init_weights(self):
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1,self.inchannels,x.shape[2],x.shape[3])
        out = activation_func(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.avg_pool2d(out, (out.size()[2],out.size()[3]))
        out = out.view(out.size()[0], -1)
        weights = self.attention(out)
        weights = torch.exp(weights)
        out = self.linear(out)
        return out, weights



def ResNet20_stoc_att(activations='relu', **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    return ResNet_stoc_att(BasicBlock, [3, 3, 3], **kwargs)


class FoundationEncoder(nn.Module):
    """Return foundation-model features and optionally freeze the encoder."""

    def __init__(self, model, frozen=False):
        super().__init__()
        self.model = model
        self.frozen = frozen
        if frozen:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, images):
        if self.frozen:
            with torch.no_grad():
                return self.model(images)
        return self.model(images)


def build_mil_encoder(
    name,
    pretrained=False,
    pretrained_path=None,
    config_path=None,
    cache_dir=None,
    tuning_mode="full_finetune",
    image_size=224,
    modality="MRI",
    body_part="N/A",
):
    """Build an image encoder that returns one feature vector per slice."""
    if name.startswith("resnet"):
        from . import resnet

        encoder = getattr(resnet, name)(
            pretrained=pretrained,
            last_activation=None,
            num_classes=1,
        )
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        return encoder, feature_dim

    if name.startswith("coatnet"):
        import timm

        name = "coatnet_0_rw_224" if name == "coatnet" else name
        encoder = timm.create_model(name, pretrained=pretrained, num_classes=0)
        return encoder, encoder.num_features

    if tuning_mode not in {"linear_probe", "full_finetune"}:
        raise ValueError("tuning_mode must be 'linear_probe' or 'full_finetune'")

    if name == "biovfm":
        
        if pretrained_path is None:
            if cache_dir is None:
                raise ValueError("BioVFM requires pretrained_path or cache_dir")
            pretrained_path = os.path.join(
                cache_dir, "BIOVFM", "pretrained_models", "biovfm_checkpoint.pth"
            )

        try:
            from .biovfm import build_biovfm
        except ImportError as exc:
            raise ImportError("The 'biovfm' encoder needs libauc/models/biovfm.py and builder_vfm.py, "
                              "which are not part of this distribution.") from exc

        encoder = build_biovfm(pretrained_path)
        frozen = tuning_mode == "linear_probe"
        return FoundationEncoder(encoder, frozen=frozen), encoder.num_features

    if name == "radeo":
       
        if not pretrained or pretrained_path is None:
            raise ValueError("RADEO requires pretrained=true and pretrained_path")

        try:
            from .radeo import load_radeo
        except ImportError as exc:
            raise ImportError("The 'radeo' encoder needs libauc/models/radeo.py, builder_vfm.py and tamevfm/, "
                              "which are not part of this distribution.") from exc

        encoder, feature_dim = load_radeo(
            pretrained_path,
            config_path=config_path,
            modality=modality,
            body_part=body_part,
        )
        frozen = tuning_mode == "linear_probe"
        return FoundationEncoder(encoder, frozen=frozen), feature_dim

    timm_name = None
    if name == "imagenet_vit":
        timm_name = "vit_base_patch16_224"
    if name == "omnirad":
        timm_name = "hf_hub:Snarcy/OmniRad-base"

    if timm_name is not None:
        if tuning_mode not in {"linear_probe", "full_finetune"}:
            raise ValueError("tuning_mode must be 'linear_probe' or 'full_finetune'")

        import timm

        kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "img_size": image_size,
        }
        cache_dir = cache_dir or os.environ.get("BASELINES_CACHE_DIR")
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        encoder = timm.create_model(timm_name, **kwargs)

        feature_dim = encoder.num_features
        frozen = tuning_mode == "linear_probe"
        return FoundationEncoder(encoder, frozen=frozen), feature_dim

    raise ValueError(f"Unknown MIL encoder: {name}")


class MILClassifier(nn.Module):
    """Apply an image encoder to a study and pool its slice features."""

    def __init__(
        self,
        encoder,
        feature_dim,
        num_classes=1,
        pool="mean",
        attention_dim=128,
        encoder_chunk_size=64,
        separate_classifier_heads=False,
        query_heads=1,
        query_rmsnorm=True,
        query_mode=None,
        num_slots=0,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        # Optional learned embedding of each window's series slot (e.g. sagittal
        # fluid vs coronal T1), added to its features before pooling. Zero init:
        # the model starts out identical to one without it.
        self.slot_embedding = None
        if num_slots:
            self.slot_embedding = nn.Embedding(int(num_slots), feature_dim)
            nn.init.zeros_(self.slot_embedding.weight)
        self.pool = pool
        self.encoder_chunk_size = int(encoder_chunk_size)
        self.query_heads = int(query_heads)
        
        if pool == "query_attention" and query_mode is None:
            raise ValueError("query_mode is required for query_attention")
        self.query_mode = query_mode

        class_attention = pool in {"class_attention", "class_gated_attention"}

        # We get class specific pooled vectors 
        class_query = pool == "query_attention" and query_mode in {
            "single_query_per_class",
            "shared_multi_query_class_pools",
            "class_multi_query_class_pools",
        }

        self.separate_classifier_heads = (
            separate_classifier_heads or class_attention or class_query
        )

        if self.separate_classifier_heads:
            self.classifiers = nn.ModuleList([
                nn.Linear(feature_dim, 1) for _ in range(num_classes)
            ])
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)

        if pool in {"attention", "class_attention"}:
            output_dim = num_classes if class_attention else 1
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, output_dim),
            )
        elif pool in {"gated_attention", "class_gated_attention"}:
            output_dim = num_classes if class_attention else 1
            self.attention_v = nn.Linear(feature_dim, attention_dim)
            self.attention_u = nn.Linear(feature_dim, attention_dim)
            self.attention = nn.Linear(attention_dim, output_dim)
        elif pool == "query_attention":
            valid_modes = {
                "single_query_per_class",
                "shared_multi_query_single_pool",
                "shared_multi_query_class_pools",
                "class_multi_query_class_pools",
            }


            if self.query_mode == "single_query_per_class":
                query_shape = (num_classes, feature_dim)

            elif self.query_mode == "class_multi_query_class_pools":
                query_shape = (num_classes, self.query_heads, feature_dim)

            else:
                query_shape = (self.query_heads, feature_dim)

            self.queries = nn.Parameter(torch.randn(query_shape))

            self.query_norm = (
                nn.RMSNorm(feature_dim) if query_rmsnorm else nn.Identity()
            )

            # Packed value projections for all attention heads.
            # Equivalent to num_heads separate Linear(D, D/num_heads) layers
            # whose outputs are concatenated into one D-dimensional tensor.
            self.value_projection = nn.Linear(feature_dim, feature_dim, bias=False)

            if self.query_mode in {
                "shared_multi_query_class_pools",
                "class_multi_query_class_pools",
            }:
                self.output_projections = nn.ModuleList([
                    nn.Linear(feature_dim, feature_dim, bias=False)
                    for _ in range(num_classes)
                ])
            elif self.query_mode == "shared_multi_query_single_pool":
                self.output_projection = nn.Linear(
                    feature_dim, feature_dim, bias=False
                )
        elif pool not in {"mean", "max"}:
            raise ValueError(
                "pool must be 'mean', 'max', 'attention', 'gated_attention', "
                "'class_attention', 'class_gated_attention', "
                "or 'query_attention'"
            )

    def forward(self, studies, return_attention=False, return_midam=False, slot_ids=None):
        batch_size, num_slices, channels, height, width = studies.shape
        images = studies.reshape(-1, channels, height, width)
        # forward pass the encoder in chunks to avoid OOM
        features = torch.cat([
            self.encoder(images[start:start + self.encoder_chunk_size])
            for start in range(0, len(images), self.encoder_chunk_size)
        ])
        features = features.reshape(batch_size, num_slices, self.feature_dim)
        if self.slot_embedding is not None and slot_ids is not None:
            features = features + self.slot_embedding(slot_ids.to(features.device))
        if self.pool == "mean":
            study_features = features.mean(dim=1)
        elif self.pool == "max":
            study_features = features.amax(dim=1)
        elif self.pool == "attention":
            weights = torch.softmax(self.attention(features), dim=1)
            study_features = (features * weights).sum(dim=1)
        elif self.pool == "gated_attention":
            attention = torch.tanh(self.attention_v(features))
            gate = torch.sigmoid(self.attention_u(features))
            weights = torch.softmax(self.attention(attention * gate), dim=1)
            study_features = (features * weights).sum(dim=1)
        elif self.pool == "query_attention":
            normalized = self.query_norm(features)
            values = self.value_projection(normalized)

            if self.query_mode == "single_query_per_class":
                scores = torch.matmul(normalized, self.queries.transpose(0, 1))
                weights = torch.softmax(scores / self.feature_dim ** 0.5, dim=1)
                class_features = (values.unsqueeze(2) * weights.unsqueeze(-1)).sum(dim=1)
                logits = []
                for class_id, classifier in enumerate(self.classifiers):
                    logits.append(classifier(class_features[:, class_id]))
                logits = torch.cat(logits, dim=1)
                if return_attention:
                    return logits, weights
                return logits

            head_dim = self.feature_dim // self.query_heads
            values = values.reshape(
                batch_size, num_slices, self.query_heads, head_dim
            )

            if self.query_mode == "class_multi_query_class_pools":
                queries = self.queries.reshape(-1, self.feature_dim)
                scores = torch.matmul(normalized, queries.transpose(0, 1))
                scores = scores.reshape(batch_size, num_slices, -1, self.query_heads)
                weights = torch.softmax(scores / head_dim ** 0.5, dim=1)
                class_features = (values.unsqueeze(2) * weights.unsqueeze(-1)).sum(dim=1)
                class_features = class_features.reshape(batch_size, -1, self.feature_dim)
                logits = []
                for class_id, classifier in enumerate(self.classifiers):
                    projected = self.output_projections[class_id](class_features[:, class_id])
                    logits.append(classifier(projected))
                logits = torch.cat(logits, dim=1)
                if return_attention:
                    return logits, weights
                return logits

            scores = torch.matmul(normalized, self.queries.transpose(0, 1))
            weights = torch.softmax(scores / head_dim ** 0.5, dim=1)
            study_features = (values * weights.unsqueeze(-1)).sum(dim=1)
            study_features = study_features.reshape(batch_size, self.feature_dim)

            if self.query_mode == "shared_multi_query_class_pools":
                logits = []
                for class_id, classifier in enumerate(self.classifiers):
                    projected = self.output_projections[class_id](study_features)
                    logits.append(classifier(projected))
                logits = torch.cat(logits, dim=1)
                if return_attention:
                    return logits, weights
                return logits

            if self.query_mode == "shared_multi_query_single_pool":
                study_features = self.output_projection(study_features)
        else:
            if self.pool == "class_attention":
                scores = self.attention(features)
            else:
                attention = torch.tanh(self.attention_v(features))
                gate = torch.sigmoid(self.attention_u(features))
                scores = self.attention(attention * gate)

            if return_midam:
                if self.pool != "class_attention":
                    raise ValueError("MIDAM currently supports class_attention only")
                instance_logits = torch.cat([classifier(features) for classifier in self.classifiers], dim=2)
                unnormalized_weights = torch.exp(scores)
                sn = (instance_logits * unnormalized_weights).mean(dim=1)
                sd = unnormalized_weights.mean(dim=1)
                return sn, sd

            weights = torch.softmax(scores, dim=1)
            logits = []
            for class_id, classifier in enumerate(self.classifiers):
                class_weights = weights[:, :, class_id].unsqueeze(-1)
                class_features = (features * class_weights).sum(dim=1)
                logits.append(classifier(class_features))
            logits = torch.cat(logits, dim=1)
            if return_attention:
                return logits, weights
            return logits

        if self.separate_classifier_heads:
            logits = torch.cat([
                classifier(study_features) for classifier in self.classifiers
            ], dim=1)
        else:
            logits = self.classifier(study_features)
        if return_attention and self.pool in {
            "attention", "gated_attention", "query_attention"
        }:
            return logits, weights
        return logits
