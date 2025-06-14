import torch
from torchvision import transforms
from PIL import Image


# 3) Lighting (PCA jitter) transform
class Lighting(object):
    """
    PCA-based color augmentation.
    α_i ~ N(0, alphastd)
    shift = eigvecs @ (α * eigvals)
    """
    def __init__(self, alphastd, eigvals, eigvecs):
        self.alphastd = alphastd
        # register as tensors for fast matmul
        self.eigvals = torch.from_numpy(eigvals).float()
        self.eigvecs = torch.from_numpy(eigvecs).float()

    def __call__(self, img):
        """
        img: Tensor shape (3,H,W), values in [0,1]
        """
        if self.alphastd == 0:
            return img
        # sample alpha
        alpha = img.new_empty(3).normal_(0, self.alphastd) #This simply creates the tensor of (3,) sampling a random number from normal distribution with mean 0 and stddev alphastd
        #We use img.new_empty to ensure the tensor is on the same device(gpu) as img


        # compute shift: (3×3) @ (3,) → (3,)
        shift = self.eigvecs @ (alpha * self.eigvals)
        # reshape to (3,1,1) and add
        return img + shift.view(3,1,1)



def clamp_to_range(tensor, min_val=0.0, max_val=1.0):
    """
    Clamp tensor values to a specified range.
    """
    return torch.clamp(tensor, min=min_val, max=max_val)



if __name__ == "__main__":
    # Load PCA values from file
    pca_values = torch.load("pca_values.pth",weights_only=False)
    mean = pca_values['mean']
    std = pca_values['std']
    eigvals = pca_values['eigvals']
    eigvecs = pca_values['eigvecs']

    # Define your training transforms
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),                               # gives [0,1]
        Lighting(alphastd=0.01, eigvals=eigvals, eigvecs=eigvecs),
        transforms.Lambda(clamp_to_range),
        transforms.Normalize(mean=mean.tolist(),
                                std= std.tolist())  # Convert numpy arrays to lists for normalization,
        ])

    #Saving the transformer after PCA calculations
    torch.save(train_transforms, "pca_lighting_transform.pth")
    print("Saved full transform to pca_lighting_transform.pth")
