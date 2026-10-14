from fastai.vision.all import *
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pathlib
import platform

# 1. WINDOWS PATH COMPATIBILITY
plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

def run_evaluation():
    # 2. DEFINE LOCAL DATA PATH
    # Ensure this folder contains your 'Real' and 'Fake' subfolders
    data_path = Path(r'D:\mini project code\experimental\scrach code\My_Deepfake_Project\Data set\real_and_fake_face') 

    # 3. LOAD THE MODELS
    print("Loading Models...")
    try:
        learn_res = load_learner(r'D:\mini project code\experimental\scrach code\My_Deepfake_Project\Models\resnet_50.pkl')
        learn_swin = load_learner(r'D:\mini project code\experimental\scrach code\My_Deepfake_Project\Models\vision_transformer.pkl')
        print("Models loaded successfully on Windows!")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 4. ATTACH LOCAL DATA
    # The models need a 'Dataloader' to know what images to evaluate locally
    dls = ImageDataLoaders.from_folder(data_path, valid_pct=0.2, item_tfms=Resize(224))
    learn_res.dls = dls
    learn_swin.dls = dls

    # 5. EXTRACT PREDICTIONS
    print("Extracting predictions from ResNet and Swin...")
    # num_workers=0 is safer for debugging on Windows to avoid process errors
    preds_res, targs = learn_res.get_preds(dl=dls.valid)
    preds_swin, _ = learn_swin.get_preds(dl=dls.valid)

    # 6. HYBRID WEIGHTED FUSION (0.55 ResNet + 0.45 Swin)
    print("Calculating Hybrid Fusion...")
    hybrid_probs = (preds_res[:, 1] * 0.55) + (preds_swin[:, 1] * 0.45)
    hybrid_preds = (hybrid_probs > 0.5).int()

    # 7. CALCULATE AND DISPLAY METRICS
    acc = accuracy_score(targs, hybrid_preds)
    pre = precision_score(targs, hybrid_preds)
    rec = recall_score(targs, hybrid_preds)

    print(f"\n" + "="*30)
    print(f"   FINAL HYBRID REPORT")
    print(f"="*30)
    print(f"Hybrid Accuracy:  {acc*100:.2f}%")
    print(f"Hybrid Precision: {pre*100:.2f}%")
    print(f"Hybrid Recall:    {rec*100:.2f}%")
    print(f"="*30)

# 8. THE "MAIN GUARD" (Crucial for Windows)
if __name__ == '__main__':
    run_evaluation()