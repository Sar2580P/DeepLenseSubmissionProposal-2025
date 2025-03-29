import torch
import torch.nn as nn
import torchvision.models as models

class ImagenetModels(nn.Module):
    AVAILABLE_MODELS = [name for name in dir(models) if callable(getattr(models, name)) and not name.startswith("_")]
    

    def __init__(self, model_name, should_finetune, num_classes, lr):
        super(ImagenetModels, self).__init__()

        self.model_name = model_name
        self.should_finetune = should_finetune
        self.num_classes = num_classes

        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model name '{self.model_name}'. Available models: {self.AVAILABLE_MODELS}")

        # Load model and modify output layer
        self.model = self.get_model()
        self.layer_lr = self.get_layer_lr(lr)  
        
    def get_layer_lr(self, lr):
        def get_params(modules):
            params = []
            for module in modules:
                params.extend(list(module.parameters()))
            return params

        if self.model_name=="resnet18":
            children = list(self.model.children())
            base = children[:-3]
            classifier = children[-3:]
            base_lr = [{'params' : get_params(base), 'lr' : lr/40}]
            classifier_lr = [{'params' : get_params(classifier), 'lr' : lr}]
            return base_lr + classifier_lr
        else :
            raise ValueError(f"Model '{self.model_name}' not supported for layer-wise learning rate adjustment")

    def get_model(self)->nn.Module:
        # Load the model from torchvision
        try:
            base_model = getattr(models, self.model_name)(pretrained=self.should_finetune)
        except AttributeError:
            raise ValueError(f"Model '{self.model_name}' not found in torchvision.models")

        # Get number of features from the last layer
        if hasattr(base_model, 'fc'):  # For models like ResNet
            in_features = base_model.fc.in_features
            base_model.fc = nn.Linear(in_features, self.num_classes)
        elif hasattr(base_model, 'classifier'):  # For models like EfficientNet
            in_features = base_model.classifier[-1].in_features
            base_model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError("Unsupported model architecture for modification")

        return base_model

    def forward(self, x):
        return self.model(x)


if __name__=="__main__":
    # Example usage
    model = ImagenetModels(model_name='resnet18', should_finetune=True, num_classes=3)
    print(model.AVAILABLE_MODELS)
    inp = torch.randn(1, 3, 150, 150)
    children = list(model.model.children())
    print(len(children))
    for ch in children:
        print(ch)
        print("\n\n")
        print("_"*50)
    # out = model(inp)
    # print(out.shape)
