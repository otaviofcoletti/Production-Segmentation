import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import PlantSegmentationDataset
from model import UNet


def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Treinando em: {device}")
    print(f"📂 Imagens: {args.rgb}")
    print(f"📂 Máscaras: {args.groundtruth}")
    print(f"💾 Saída: {args.modelpath}")

    # Transformações
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomCrop(512, 512),
        ToTensorV2(),
    ])

    # Dataset e DataLoaders
    dataset = PlantSegmentationDataset(args.rgb, args.groundtruth, transform=None)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Modelo
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validação
        model.eval()
        val_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_iou += iou_score(preds, masks).item()

        avg_iou = val_iou / len(val_loader)
        print(f"📘 Epoch {epoch}: Train Loss={avg_loss:.4f} | Val IoU={avg_iou:.4f}")

        if avg_iou > best_iou:
            best_iou = avg_iou
            os.makedirs(args.modelpath, exist_ok=True)
            model_path = os.path.join(args.modelpath, "unet_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Novo melhor modelo salvo: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de Segmentação de Plantação (U-Net)")
    parser.add_argument("--rgb", required=True, help="Diretório com imagens de treino")
    parser.add_argument("--groundtruth", required=True, help="Diretório com máscaras correspondentes")
    parser.add_argument("--modelpath", default="runs", help="Diretório de saída para salvar o modelo")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas de treinamento")
    parser.add_argument("--batch-size", type=int, default=4, help="Tamanho do batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Taxa de aprendizado")

    args = parser.parse_args()
    train(args)
