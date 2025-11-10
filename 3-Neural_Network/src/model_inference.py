import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse
from model import UNet


def predict_image(model, device, img_path, threshold=0.59):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_t)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    mask = (pred > threshold).astype(np.uint8) * 255
    return mask


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    print(f"🚀 Rodando inferência em {device}")
    print(f"📂 Modelo: {args.modelpath}")
    print(f"📂 Input: {args.rgb}")
    print(f"💾 Output: {args.output}")

    model.load_state_dict(torch.load(args.modelpath, map_location=device))
    model.eval()

    os.makedirs(args.output, exist_ok=True)

    for img_name in tqdm(os.listdir(args.rgb), desc="Inferindo imagens de teste"):
        if not img_name.lower().endswith(('.jpg', '.png', '.tif')):
            continue

        img_path = os.path.join(args.rgb, img_name)
        mask = predict_image(model, device, img_path)

        save_path = os.path.join(args.output, img_name)
        cv2.imwrite(save_path, mask)

    print(f"✅ Inferência concluída! Máscaras salvas em: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência com U-Net treinada")
    parser.add_argument("--modelpath",default="runs/unet_best.pth", required=True, help="Caminho do modelo treinado (.pth)")
    parser.add_argument("--rgb", required=True, help="Diretório com imagens para inferência")
    parser.add_argument("--output", required=True, help="Diretório de saída para salvar as máscaras")

    args = parser.parse_args()
    infer(args)
