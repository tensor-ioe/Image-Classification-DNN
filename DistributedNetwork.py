import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from TransformDefin import Lighting, clamp_to_range  

class Trainer:
    def __init__(self, args):
        # 1) Distributed setup
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        self.rank   = rank
        self.device = torch.device("cuda", local_rank)

        # 2) Load your saved PCA‐lighting transform
        #    Make sure the file "pca_lighting_transform.pth" is accessible here.
        train_transforms = torch.load("pca_lighting_transform.pth", map_location="cpu")

        # 3) Build your Dataset & DataLoader
        #    Example using an ImageFolder of JPEGs -- replace with your dataset
        dataset = datasets.ImageFolder(
            root=args.train_dir,
            transform=train_transforms
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader  = DataLoader(dataset,
                             batch_size=args.batch_size,
                             sampler=sampler,
                             num_workers=4,
                             pin_memory=True)

        # 4) Model, optimizer, loss
        self.model     = self.build_model().to(self.device)
        self.model     = DDP(self.model, device_ids=[local_rank])
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

        self.loader = loader
        self.sampler = sampler
        self.epochs  = args.epochs

    def build_model(self):
        return nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32,1)
        )

    def train(self):
        for epoch in range(self.epochs):
            self.sampler.set_epoch(epoch)
            total_loss = 0.0
            for imgs, targets in self.loader:
                imgs    = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                preds = self.model(imgs)
                loss  = self.criterion(preds, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if self.rank == 0:
                avg = total_loss / len(self.loader)
                print(f"Epoch {epoch+1}/{self.epochs} → Loss: {avg:.4f}")

        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",  type=str,   required=True,
                        help="Folder of training images, organized by class")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    Trainer(args).train()

if __name__ == "__main__":
    main()
