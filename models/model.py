import timm
import torch.nn as nn
from commons.config import ARCHITECTURE
from huggingface_hub import PyTorchModelHubMixin

class ConvNextV2(nn.Module, PyTorchModelHubMixin, 
    repo_url="your-repo-url",
    pipeline_tag="text-to-image",
    license="mit",):
    def __init__(self, n_out):
        super(ConvNextV2, self).__init__()
        # Define model
        self.convnext = timm.create_model(
            ARCHITECTURE, pretrained=True, num_classes=n_out
        )

        for param in self.convnext.parameters():
            param.requires_grad = False

        for param in self.convnext.stages[3].parameters():
            param.requires_grad = True
        for param in self.convnext.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.convnext(x)