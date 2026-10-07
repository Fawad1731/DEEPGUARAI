import os
from PIL import Image
import random

def create_dummy_dataset():
    base_dir = "./dataset"
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print("Generating dummy dataset...")

    # Create 50 real and 50 fake dummy images
    for i in range(50):
        # Real images: slightly bluish
        real_img = Image.new('RGB', (224, 224), color = (random.randint(50, 100), random.randint(50, 100), random.randint(150, 255)))
        real_img.save(os.path.join(real_dir, f'real_{i}.jpg'))
        
        # Fake images: slightly reddish
        fake_img = Image.new('RGB', (224, 224), color = (random.randint(150, 255), random.randint(50, 100), random.randint(50, 100)))
        fake_img.save(os.path.join(fake_dir, f'fake_{i}.jpg'))

    print("Successfully created dummy dataset with 100 images in './dataset'!")

if __name__ == "__main__":
    create_dummy_dataset()
