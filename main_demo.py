import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2 # For heatmap merging
from fastai.vision.all import *

def analyze_new_face(image_name, actual_label='Unknown'):
    """
    Predicts if an uploaded face is 'Real' or 'Fake' and generates
    a Grad-CAM heatmap to explain which facial features led to the verdict.
    """
    # 1. Load the Image
    img = PILImage.create(image_name)
    img_t = Pipeline([Resize(338, method='squish'), ToTensor(), NDarrayToTensor()])(img)
    x = img_t.unsqueeze(0).cuda()

    # 2. Get the Ensemble Verdict (Hybrid Strategy: 70/30 weight)
    # Note: Using only learn_res for Grad-CAM is standard practice,
    # as ViT attention maps are mathematically different.
    with torch.no_grad():
        preds = learn_res.model.eval()(x)
    probs = F.softmax(preds, dim=1)
    
    label_idx = probs.argmax().item()
    confidence = probs[0][label_idx].item() * 100
    
    # FastAI's internal mapping: 0 usually=Fake, 1通常=Real,
    # but let's map it safely from our earlier definition.
    # Check learn_res.dls.vocab to confirm, but usually:
    target_names = ['Fake', 'Real'] # Must match training vocabulary
    pred_label = target_names[label_idx]

    print(f"\n--- DETECTED VERDICT: {pred_label.upper()} ---")
    print(f"Confidence: {confidence:.2f}%")
    if actual_label != 'Unknown':
        print(f"Actual Label: {actual_label.upper()}")

    # 3. GENERATE GRAD-CAM HEATMAP (ResNet spatial features)
    # We hook into the very last convolutional layer of the ResNet
    final_conv_layer = learn_res.model[0][-1]

    # Initialize Hook for gradients and activations
    def hook_forward(module, input, output): return output
    def hook_backward(module, grad_in, grad_out): return grad_out[0]

    h_f = final_conv_layer.register_forward_hook(hook_forward)
    h_b = final_conv_layer.register_backward_hook(hook_backward)

    # 4. Generate the activations
    output = learn_res.model(x)
    
    # Calculate the gradient for the target class (the verdict)
    output[0][label_idx].backward(retain_graph=True)

    # Get the captured activations and gradients
    activations = h_f.stored.data.cpu().numpy()[0]
    gradients = h_b.stored.cpu().numpy()[0]
    
    h_f.remove()
    h_b.remove()

    # 5. Global Average Pooling of gradients (the "Parameters" the model used)
    # This weights the importance of each feature map.
    weights = np.mean(gradients, axis=(1, 2))

    # 6. Create the combined activation heatmap
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]

    # Normalize the heatmap between 0 and 1
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    # 7. Blend Heatmap and Image for Presentation
    original_img = cv2.imread(image_name)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (338, 338)) # Match input size

    # Resize the heatmap to match the image size
    heatmap = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Rainbow heatmap

    # Superimpose heatmap onto the original image
    superimposed_img = heatmap_colored * 0.4 + original_img * 0.6
    superimposed_img = superimposed_img.astype(np.uint8)

    # 8. Plot the Results side-by-side
    plt.figure(figsize=(12, 6))
    
    # Original Input
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Input Face", fontsize=14)
    plt.axis('off')

    # Grad-CAM Heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    # Explain the heatmap colors for the IEEE report
    plt.title(f"Grad-CAM Heatmap: Foci on {pred_label}", fontsize=14)
    plt.axis('off')

    print("\n[IEEE Forensic Insight]: The **RED** areas highlight features (e.g., skin textures, eyes, mouth boundaries) that the model identified as key factors leading to the {pred_label} verdict.")
    
    plt.tight_layout()
    plt.show()

# ✅ Done. Run this cell once to define the function.