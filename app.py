
# 🔽 Install required packages before running (if not already installed)
!pip install torch torchvision medmnist gradio gtts opencv-python matplotlib pillow  grad-cam --quiet

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import gradio as gr
from gtts import gTTS
import os
import cv2
import matplotlib.pyplot as plt

from medmnist import PneumoniaMNIST
from medmnist import INFO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 🔽 CNN model
class PheumoNet(nn.Module):
    def __init__(self):
        super(PheumoNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 🔽 Load PneumoniaMNIST dataset and train
def train_model():
    info = INFO['pneumoniamnist']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = PneumoniaMNIST(split='train', transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 🔽 Visualize 10 samples from training set
    transform = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize([0.5], [0.5])
      ])

    train_dataset = PneumoniaMNIST(split='train', transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    classes = ['Normal', 'Pneumonia']

    # Get one batch
    images, labels = next(iter(train_loader))

    # Show 10 images in 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f"Label: {classes[labels[i].item()]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


    model = PheumoNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🔁 Training started...")
    model.train()
    for epoch in range(5):  # Train for 5 epochs
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "pheumonet.pth")
    print("✅ Model trained and saved as pheumonet.pth")
    return model

# 🔽 Load or train model
model = PheumoNet()
if os.path.exists("pheumonet.pth"):
    model.load_state_dict(torch.load("pheumonet.pth", map_location=torch.device("cpu")))
else:
    model = train_model()
model.eval()

# 🔽 Convert prediction to audio
def speak_prediction(text, lang_code, filename):
    tts = gTTS(text=text, lang=lang_code)
    tts.save(filename)
    return filename

COLORMAPS = {
    "Jet": cv2.COLORMAP_JET,
    "Hot": cv2.COLORMAP_HOT,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Cool": cv2.COLORMAP_COOL,
}

# 🔽 Prediction with GradCAM and voice output
def predict(image, cmap_name):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    original_image = image.copy()
    image = image.convert("L")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        label = ['Normal', 'Pneumonia'][pred]

    target_layer = model.features[3]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred)])[0]

    img_np = input_tensor.squeeze().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    rgb_img = np.repeat(img_np[..., np.newaxis], 3, axis=-1)

    colormap = COLORMAPS.get(cmap_name, cv2.COLORMAP_JET)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, colormap=colormap)
    cam_pil = Image.fromarray(cam_image)

    if label == "Pneumonia":
        en_audio = speak_prediction("Pneumonia detected.Seek medical attention immediately.", "en", "en.mp3")
        ha_audio = speak_prediction("An gano cutar Pneumonia.Da fatan za a nemi kulawan likita.", "ha", "ha.mp3")
        pidgin_audio = speak_prediction("Dem don see Pneumonia. Abeg go hospital sharp sharp", "en", "pidgin.mp3")
    else:
        en_audio = speak_prediction("No pneumonia detected. You are safe.", "en", "en.mp3")
        ha_audio = speak_prediction("Ba a gano cutar Pneumonia ba. Kana lafiya.", "ha", "ha.mp3")
        pidgin_audio = speak_prediction("Dem no see Pneumonia. You dey alright", "en", "pidgin.mp3")

    return label, original_image, cam_pil, en_audio, ha_audio, pidgin_audio

# 🔽 Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Upload Chest X-ray", image_mode='L', type='pil'),
        gr.Radio(list(COLORMAPS.keys()), label="Choose GradCAM Heatmap Color", value="Inferno")
    ],
    outputs=[
        gr.Text(label="Prediction"),
        gr.Image(label="Original Image"),
        gr.Image(label="Grad-CAM Heatmap"),
        gr.Audio(label="English Voice"),
        gr.Audio(label="Hausa Voice"),
        gr.Audio(label="Pidgin Voice")
    ],
    title="🩺 PheumoScan AI",
    description="Upload a chest X-ray (PneumoniaMNIST-style). Choose heatmap color. Get AI diagnosis with Grad-CAM and hear results in English, Hausa, and Pidgin."
)

demo.launch()
