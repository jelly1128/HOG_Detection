import torchvision.transforms as transforms

def get_train_transforms():
    return transforms.Compose([
        # 五度までの回転及びせん断，2.5%までの平行移動
        transforms.RandomAffine(degrees=5, translate=(0.025, 0.025), shear=5),
        transforms.ToTensor(),
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
    ])