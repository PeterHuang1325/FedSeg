import albumentations as A
from pytorch_lightning import seed_everything
from torchvision import transforms

seed_everything(1337)

#transformation: brain
train_tfm = A.Compose([
    A.Resize(width=256, height=256, p=1.0),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
])

eval_tfm = A.Compose([
    A.Resize(width=256, height=256, p=1.0)
])


#transformation: prostate
train_pro_tfm = A.Compose([
    A.Resize(width=384, height=384, p=1.0),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
])

eval_pro_tfm = A.Compose([
    A.Resize(width=384, height=384, p=1.0)
])


#transformation: mnist
train_dig_tfm = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

eval_dig_tfm = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


