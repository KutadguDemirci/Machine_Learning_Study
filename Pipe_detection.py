import os
from PIL import Image
import torch
from torch import nn
import torchvision
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from pycocotools.coco import COCO

# ✅ Device setup
device = torch.device("cpu")
print(f"Using device: {device}")

# ✅ Custom Dataset
class COCODetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms if transforms else Compose([Resize((512, 512)), ToTensor()])

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        img = self.transforms(img)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                continue
            xmax = min(x + w, img_w)
            ymax = min(y + h, img_h)
            if xmax <= x or ymax <= y:
                continue
            boxes.append([x, y, xmax, ymax])
            labels.append(1)  # Assuming "pipe" is the only class

        if not boxes:
            print(f"⚠️ No valid boxes for image ID {img_id} — skipping")
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        return img, target

    def __len__(self):
        return len(self.ids)

# ✅ Collate
def collate_fn(batch):
    batch = list(filter(None, batch))  # remove skipped samples
    if len(batch) == 0:
        return None, None
    return tuple(zip(*batch))

# ✅ Datasets and loaders
train_dataset = COCODetectionDataset(
    root="pipe_data/train_img",
    annotation="pipe_data/train_annot/_annotations.coco.json"
)

valid_dataset = COCODetectionDataset(
    root="pipe_data/valid_img",
    annotation="pipe_data/valid_annot/_annotations.coco.json"
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# ✅ Model init
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes=2)  # 1 class + background
model.to(device)

# ✅ Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# ✅ Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0.0

    for step, (images, targets) in enumerate(train_loader):
        if images is None or targets is None:
            print(f"⚠️ Skipping empty batch at step {step}")
            continue

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"❌ Non-finite loss detected at step {step}: {losses.item()}")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = losses.item()
            epoch_loss += loss_value

            print(f"  Step {step}: loss={loss_value:.4f}")

        except Exception as e:
            print(f"❌ Error during training step {step}: {e}")
            continue

    lr_scheduler.step()
    print(f"✅ Epoch {epoch+1} completed — Total Loss: {epoch_loss:.4f}")

# ✅ Evaluation loop
model.eval()
print("\n🔍 Evaluation Results:")
with torch.no_grad():
    for batch_i, (images, targets) in enumerate(valid_loader):
        if images is None or targets is None:
            continue

        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            scores = output['scores'].cpu()
            boxes = output['boxes'].cpu()
            count = sum(score > 0.5 for score in scores)
            print(f"Image {batch_i * len(images) + i}: Detected {count} pipes")
