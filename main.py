import argparse
import os
import torch

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

from crossvit import CrossViT
from dataset import Dataset_siwoo
from utils import load_checkpoint, save_checkpoint

def calculate_class_iou(outputs, targets, num_classes):
    class_tp = [0] * num_classes
    class_fp = [0] * num_classes
    class_fn = [0] * num_classes

    for cls in range(num_classes):
        tp = ((outputs == cls) & (targets == cls)).sum().item()
        class_tp[cls] += tp

        fp = ((outputs == cls) & (targets != cls)).sum().item()
        class_fp[cls] += fp

        fn = ((outputs != cls) & (targets == cls)).sum().item()
        class_fn[cls] += fn

    class_iou = []
    for cls in range(num_classes):
        union = class_tp[cls] + class_fp[cls] + class_fn[cls]
        if union == 0:
            iou = 0
        else:
            iou = class_tp[cls] / union
        class_iou.append(iou)

    return class_iou

def main(args):


    # create model
    model = CrossViT(224, 3, 1000)
    if args.pretrained != 'None':
        model = load_checkpoint(model, args.pretrained)

    model.to(args.device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    # init param & create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    # create dataset & dataloader
    train_dataset = Dataset_siwoo(args, 'train')
    val_dataset = Dataset_siwoo(args, 'val')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset)


    best_val_acc = 0.0
    best_model_path = os.path.join(CKPT_PATH, f'best_Mean_IOU_{best_val_acc:.4f}_model.pth')

    for epoch in range(args.num_epoch):
        # Training loop
        model.train()  # 모델을 학습 모드로 설정
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_dataloader):

            data, target = data.to(device), target.to(device)

            output = model(data)  # 순방향 전파

            loss = criterion(output, target)  # 손실 계산

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # 배치 손실 누적

        # Training 손실 출력
        train_loss /= len(train_dataloader)
        print(f"Epoch: {epoch + 1}/{args.num_epoch}, Train Loss: {train_loss:.4f}")

        # Validation loop
        model.eval()  # 모델을 평가 모드로 설정
        val_loss = 0.0
        acc_scores = []  # Validation accuracy

        for batch_idx, (data, target) in enumerate(val_dataloader):
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output = model(data)  # 순방향 전파
                loss = criterion(output, target)  # 손실 계산

                # Calculate acc score
                #??????????


                # Plot images with predictions
                for i in range(data.shape[0]):
                    plt.figure(figsize=(10, 5))

                    input_image = data[i].detach().cpu().permute(1, 2, 0)
                    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

                    plt.subplot(1, 3, 1)
                    plt.imshow(input_image)
                    plt.title("Image")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(target[i].detach().cpu().squeeze())
                    plt.title("Ground truth")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(output[i].argmax(dim=0).detach().cpu().numpy())
                    plt.title("Prediction")
                    plt.axis("off")
                    plt.tight_layout()

                    plt.savefig(os.path.join(LOG_PATH, f"epoch_{epoch}_batch_{batch_idx}_sample_{i}.png"))
                    plt.close()

            val_loss += loss.item()

        # Validation 손실 및 성능 출력
        val_loss /= len(valid_loader.dataset)
        mean_acc = np.mean(acc_scores)
        print(f"Epoch: {epoch + 1}/{args.num_epoch}, Validation Loss: {val_loss:.4f}, Mean acc: {mean_acc:.4f}")


        if mean_acc > best_val_acc:
            best_val_iou = mean_iou
            best_model_path = os.path.join(args.ckpt_path, f'best_Mean_IOU_{best_val_iou:.4f}_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1}")



def test(args):
    # create model
    model = CrossViT(224, 3, 1000)
    model.to(args.device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)


    # create dataset & dataloader
    test_dataset = Dataset_siwoo(args, 'test')
    test_dataloader = DataLoader(test_dataset)

    # Test loop
    model.eval()  # 모델을 평가 모드로 설정
    test_loss = 0.0
    test_iou_scores = []  # Test IOU scores
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():  # 그래디언트 계산 비활성화
            data = data.float()  # 입력 데이터의 데이터 타입을 float로 변경

            output = model(data)  # 순방향 전파

            loss = criterion(output, target)  # 손실 계산

            # Calculate IOU score
            iou_score = calculate_class_iou(output.argmax(dim=1), target, num_classes=4)
            test_iou_scores.append(iou_score)

            test_loss += loss.item()  # 배치 손실 누적

    # Test 손실 및 성능 출력
    test_loss /= len(test_loader.dataset)
    mean_test_iou = np.mean(test_iou_scores)
    test_iou_list.append(mean_test_iou)
    print(f"Epoch: {epoch + 1}/{args.num_epoch}, Test Loss: {test_loss:.4f}, Mean Test IOU: {mean_test_iou:.4f}")

    print(f"Mean Test IOU after {args.num_epoch} epochs: {mean_test_iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrossViT helped by Siwoo")

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument("--data_dir", type=str, default="/home/",help="Path to train dataset directory")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_epoch", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--apply_transform", action="store_true", help="Apply data transformation")
    parser.add_argument("--pretrained", type=str, default="None", help="Path to test dataset directory")
    #https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_224.pth

    parser.add_argument("--save_dir", type=str, default="/home/", help="Path to test dataset directory")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args.device = device


    args.ckpt_path = os.path.join(args.save_dir, 'checkpoints')
    args.result_path = os.path.join(save_dir, 'results')

    os.makedirs(args.ckpt_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    if args.train==True:
        main(args)
    if args.test==True:
        test(args, device)
