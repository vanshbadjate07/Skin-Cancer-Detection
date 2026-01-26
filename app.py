import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from torchvision.models import efficientnet_b0

st.set_page_config(layout="centered")
st.title("Skin Cancer Detector with GradCAM")

device = "cpu"

# Load model
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 3)
model.load_state_dict(torch.load("skin_model.pth", map_location=device))
model.eval()
model.to(device)

labels = ["AK", "BCC", "SK"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# GradCAM helper
class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        if target_layer is None:
            target_layer = None
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
            if target_layer is None:
                raise ValueError("No Conv2d layer found in model")
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.clone().to(next(self.model.parameters()).device)
        input_tensor.requires_grad_(True)

        outputs = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(outputs.argmax(dim=1).item())

        score = outputs[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=False)

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = grads.mean(dim=(1,2))

        cam = torch.zeros(acts.shape[1:], dtype=acts.dtype, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        cam_np = cam.detach().cpu().numpy()
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
        conf = float(probs[class_idx])

        return cam_np, class_idx, conf

def apply_heatmap_on_image(img_pil, cam_np, alpha=0.5):
    W_img, H_img = img_pil.size
    cam_resized = cv2.resize((cam_np * 255).astype(np.uint8), (W_img, H_img))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img = np.array(img_pil.convert("RGB"))
    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

    return Image.fromarray(overlay)

uploaded = st.file_uploader("Upload skin image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_tensor)
        prob = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
        pred_idx = int(prob.argmax())
        conf = float(prob[pred_idx])

    gradcam = GradCAM(model)
    cam_np, class_idx, conf2 = gradcam(img_tensor)
    heatmap_img = apply_heatmap_on_image(img, cam_np)

    # ---------- NEW UI LAYOUT ----------
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Preview", width=320)

    with col2:
        st.image(heatmap_img, caption="Model Attention Heatmap", width=320)

    st.markdown(
        f"<h3 style='text-align:center;'>Prediction: {labels[pred_idx]} — {conf*100:.1f} percent</h3>",
        unsafe_allow_html=True
    )
    # ---------- END UI LAYOUT ----------
