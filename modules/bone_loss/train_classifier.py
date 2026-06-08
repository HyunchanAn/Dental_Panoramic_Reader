import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def train_model():
    data_dir = 'data/classifier'
    model_save_path = 'models/pano_classifier.pt'
    
    if not os.path.exists(data_dir):
        print("데이터 디렉토리가 없습니다. utils/prepare_classifier_data.py 를 먼저 실행하세요.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 변환 (Data Augmentation & Normalization)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # ImageFolder는 폴더명을 클래스로 사용합니다. ('non_pano': 0, 'pano': 1)
    # 폴더 구조: data/classifier/non_pano, data/classifier/pano
    image_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
    
    # 클래스 인덱스 확인
    class_names = image_dataset.classes
    print(f"Classes: {class_names} -> Class to idx: {image_dataset.class_to_idx}")
    
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 가벼운 MobileNetV3-Small 모델 로드 (Pre-trained)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # 마지막 분류기(Classifier) 교체 (2개 클래스로)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print("Training started...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(image_dataset)
        epoch_acc = running_corrects.double() / len(image_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    # 모델 가중치 저장
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete! Model saved to {model_save_path}")

if __name__ == '__main__':
    train_model()
