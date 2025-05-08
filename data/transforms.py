import torchvision.transforms as transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomRotation(degrees=360),
        transforms.ToTensor(),
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
    ])