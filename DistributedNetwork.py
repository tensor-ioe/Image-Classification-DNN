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
from CNN_Architecture import alexNet, save_model, load_model


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
        train_dataset = datasets.ImageFolder(
            root=args.train_dir,
            transform=train_transforms
        )
        trainSampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        trainLoader  = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             sampler=trainSampler,
                             num_workers=4,
                             pin_memory=True)
        valid_dataset = datasets.ImageFolder(
            root=args.valid_dir,
            transform=train_transforms
        )
        validSampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
        validLoader  = DataLoader(valid_dataset,
                             batch_size=args.batch_size,
                             sampler=validSampler,
                             num_workers=4,
                             pin_memory=True)

        # 4) Model, optimizer, loss
        self.model     = alexNet().to(self.device)
        self.model     = DDP(self.model, device_ids=[local_rank])
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.trainLoader = trainLoader
        self.trainSampler = trainSampler
        self.validLoader = validLoader
        self.validSampler = validSampler
        self.epochs  = args.epochs


    def validation(self):
        self.model.eval()
        local_loss_sum = 0
        local_sample_count = 0
        local_correct=0
        with torch.no_grad():
            for images, labels in self.validLoader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                # per-sample loss, sum over this batch
                losses = self.criterion(outputs, labels)       # [batch_size]
                local_loss_sum    += losses.sum().item()
                local_sample_count += labels.size(0)

                # count correct predictions in this batch
                _, preds = outputs.topk(1, dim=1)
                local_correct += (preds.view(-1) == labels).sum().item()

            # turn to tensors for all_reduce
            loss_tensor   = torch.tensor(local_loss_sum,    device=self.device)
            count_tensor  = torch.tensor(local_sample_count, device=self.device)
            correct_tensor= torch.tensor(local_correct,          device=self.device)

            # sum them across all ranks
            dist.all_reduce(loss_tensor,  op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor,   op=dist.ReduceOp.SUM)

            avg_loss = loss_tensor.item() / count_tensor.item()
            avg_acc  = correct_tensor.item() / count_tensor.item()

        return avg_loss, avg_acc
    
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            self.trainSampler.set_epoch(epoch)

            local_loss_sum    = 0.0
            local_sample_count = 0

            for imgs, targets in self.trainLoader:
                imgs    = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                preds = self.model(imgs)
                losses  = self.criterion(preds, targets)
                loss=losses.mean()  # average loss over this batch
                loss.backward()
                self.optimizer.step()

                local_loss_sum    += losses.sum().item()      # sum of batch’s per-sample losses
                local_sample_count += targets.size(0)

            local_avg_loss = local_loss_sum / local_sample_count
            #loss per example in the epoch

            if self.rank == 0:
                self.model.eval()
                valid_loss, valid_accuracy = self.validation()
                print(f"Epoch {epoch}/ {self.epochs}\tTraining Loss : {local_avg_loss}\tValidation Loss : {valid_loss}\tValidation Accuracy : {valid_accuracy}")
                self.model.train()
                if epoch % 2 == 0 or epoch == (self.epochs-1):
                    save_model(self.model.module, f"Checkpoint{epoch}.pth")

        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",  type=str,   required=True,
                        help="Folder of training images, organized by class")
    parser.add_argument("--valid_dir",  type=str,   required=True, default="valid",
                        help="Folder of Validating images, organized by class")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    args = parser.parse_args()

    Trainer(args).train()

if __name__ == "__main__":
    main()
