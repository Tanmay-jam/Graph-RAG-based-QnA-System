# model.py
# Module for loading and saving the MobileNetV3 leaf-disease classifier weights

import os
import argparse
import torch
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights


def build_and_save_model(state_dir: str, output_dir: str, device: torch.device):
    """
    Load a MobileNetV3 base model, modify classifier for 6 classes,
    load weights from state_dir, and save the complete model to output_dir.
    """
    # Initialize base model
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    # Modify classifier for 6 classes
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    model.classifier[-2] = torch.nn.Dropout(p=0.3, inplace=True)
    num_classes = 6
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_feat, num_classes)
    model.classifier[-1].apply(init_weights)

    # Load pre-trained weights
    state_path = os.path.join(state_dir, 'acc_model.pth')
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Save the entire model for inference
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'mobilenetv3_leaf_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', type=str, required=True,
                        help='Directory containing acc_model.pth')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save the full model weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    build_and_save_model(args.state_dir, args.output_dir, device)