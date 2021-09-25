import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from dataset import create_dataloaders
from wavelet_ae import WaveletAutoEncoder
from loss import CompressionLoss
from metrics import psnr, msssim


class Trainer:
    def __init__(self, data_dir, log_dir=None) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = WaveletAutoEncoder(self.device)
        self.model = self.model.to(device=self.device)
        
        self.loss_func = CompressionLoss()
        self.optimizer = optim.Adam(self.model.parameters(), 1e-4)
        self.epochs = 100
        self.batch_size = {"train": 8, "valid": 1}

        self.data_loaders = create_dataloaders(data_dir, self.batch_size)
        self.writer = SummaryWriter(log_dir)

    def validate(self, epoch):
        with torch.no_grad():
            self.model.eval()
            num_batches = len(self.data_loaders['valid'])
            
            averages = {
                "psnr": 0,
                "msssim": 0,
                "full loss": 0,
                "recon loss": 0,
                "entropy loss": 0
            }

            for batch_idx, images in enumerate(self.data_loaders['valid']):
                print("Validating on batch %d/%d" % (batch_idx + 1, num_batches))

                images = images.to(self.device)
                recon_images, symbols = self.model(images)

                full_loss, recon_loss_val, entropy_loss_val = self.loss_func(recon_images, images, symbols)
                
                images = (images + 1.) * 0.5
                recon_images = (recon_images + 1.) * 0.5

                averages['psnr'] += psnr(images, recon_images).item()
                averages["msssim"] += msssim(images, recon_images).item()
                averages["full loss"] += full_loss.item()
                averages["recon loss"] += recon_loss_val
                averages["entropy loss"] += entropy_loss_val
            
            for key in averages:
                averages[key] /= num_batches

            print("Validation scores:", averages)
            self.writer.add_scalars("Loss/valid", {"full loss": averages["full loss"], "recon loss": averages["recon loss"],
                                                   "entropy loss": averages["entropy loss"]}, epoch)
            self.writer.add_scalar("PSNR/valid", averages['psnr'], epoch)
            self.writer.add_scalar("MSSSIM/valid", averages['msssim'], epoch)
            self.writer.add_images("Images/GT", images, epoch)
            self.writer.add_images("Images/reconstructed", recon_images, epoch)

    def train(self):
        
        for epoch in range(self.epochs):
            print("Training epoch %d/%d" % (epoch + 1, self.epochs))
            self.model.train()
            num_batches = len(self.data_loaders['train'])

            for batch_idx, patches in enumerate(self.data_loaders["train"]):
                self.optimizer.zero_grad()

                patches = patches.to(self.device)
                recon_patch, symbols = self.model(patches)

                full_loss, recon_loss_val, entropy_loss_val = self.loss_func(recon_patch, patches, symbols)
                # full_loss += wavelet_loss
                full_loss.backward()
                self.optimizer.step()

                full_loss_val = full_loss.item()
                if batch_idx % 20 == 0:
                    print(f"Batch {batch_idx + 1}/{num_batches}: Full loss: {full_loss_val}, Recon loss: {recon_loss_val}, Entropy loss: {entropy_loss_val}")#, Wavelet loss: {wavelet_loss.item()}")

                self.writer.add_scalars("Loss/train", {"full loss": full_loss_val, "recon loss": recon_loss_val,
                                                       "entropy loss": entropy_loss_val}, batch_idx)
            self.validate(epoch)
        
        self.writer.flush()
        return self.model


if __name__ == "__main__":
    data_dir = "/home/orweiser/university/Digital-Processing-of-Single-and-Multi-Dimensional-Signals/data"
    trainer = Trainer(data_dir)
    trainer.train()
