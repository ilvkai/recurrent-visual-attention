import torchvision
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],output_size=7,sampling_ratio=2)
# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler)
model.eval()
x = torch.rand(1,3,800,800)
predictions = model(x)
print(predictions.shape)