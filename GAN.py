import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import progressbar

# Mapping from numerical labels to class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

OUTPUT_DIR = './data/CIFAR-10'
MODEL_DIR = './models'
IMAGE_SAVE_DIR = './generated_images'
BATCH_SIZE = 100
LR = 0.001
NUM_EPOCHS = 1
NUM_TEST_SAMPLES = 32

# Ensure the directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)


def load_data():
    compose = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    return datasets.CIFAR10(root=OUTPUT_DIR, train=False, transform=compose, download=True)


data = load_data()
data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
NUM_BATCHES = len(data_loader)

print("No. of Batches =", NUM_BATCHES)


def noise(size, num_classes):
    # Generate random noise
    n = Variable(torch.randn(size, 100)).to(device)
    # Create random class labels
    labels = np.random.randint(0, num_classes, size)
    one_hot_labels = np.zeros((size, num_classes))
    one_hot_labels[np.arange(size), labels] = 1
    one_hot_labels = torch.tensor(one_hot_labels, dtype=torch.float32).to(device)
    # Concatenate noise and one-hot labels
    noise_with_labels = torch.cat((n, one_hot_labels), dim=1)
    return noise_with_labels, labels


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)


def real_data_target(size):
    data = Variable(torch.ones(size, 1)).to(device)
    return data


def fake_data_target(size):
    data = Variable(torch.zeros(size, 1)).to(device)
    return data


def train_discriminator(optimizer, real_data, fake_data):
    optimizer.zero_grad()
    prediction_real = discriminator(real_data)
    error_real = criterion(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()
    prediction_fake = discriminator(fake_data)
    error_fake = criterion(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = criterion(prediction, real_data_target(prediction.size(0)))
    error.backward()
    optimizer.step()
    return error


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024), nn.LeakyReLU(0.2, inplace=True))
        self.output = nn.Sequential(nn.Linear(1024 * 4 * 4, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 1024 * 4 * 4)
        x = self.output(x)
        return x


class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes):
        super(Generator, self).__init__()
        self.linear = nn.Linear(noise_dim + num_classes, 1024 * 4 * 4)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False))
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return self.output(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

noise_dim = 100
num_classes = len(class_names)
generator = Generator(noise_dim, num_classes).to(device)
discriminator = Discriminator().to(device)
generator.apply(init_weights)
discriminator.apply(init_weights)

d_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
criterion = nn.BCELoss()

test_noise, test_labels = noise(NUM_TEST_SAMPLES, num_classes)


def generate_and_save_images(generator, test_noise, test_labels, epoch, save_dir, class_names,
                             num_images=NUM_TEST_SAMPLES):
    with torch.no_grad():
        test_images = generator(test_noise).cpu()

    rows = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(rows, rows, figsize=(10, 10))
    fig.suptitle(f"Generated Images at Epoch {epoch}", fontsize=16)

    for i in range(num_images):
        image_tensor = test_images[i]
        class_name = class_names[test_labels[i]]  # Use the provided class labels
        np_image = image_tensor.numpy().transpose((1, 2, 0))

        # Rescale image to [0, 1] for proper display
        np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())

        # Plot the image with the class label on top
        ax = axes[i // rows, i % rows]
        ax.imshow(np_image)
        ax.set_title(f"{class_name}")
        ax.axis('off')

        # Save the image with the class name in the file name
        image_name = f'epoch_{epoch}_image_{i}_class_{class_name}.png'
        image_path = os.path.join(save_dir, image_name)
        plt.imsave(image_path, np_image)
        print(f'Saved image at {image_path}')

    # Remove empty subplots
    for j in range(i + 1, rows * rows):
        fig.delaxes(axes[j // rows, j % rows])

    # Save the combined image grid
    combined_image_path = os.path.join(save_dir, f'epoch_{epoch}_combined.png')
    plt.savefig(combined_image_path)
    plt.show()
    plt.close()
    print(f'Saved combined image grid at {combined_image_path}')


def save_models(generator, discriminator):
    torch.save(generator.state_dict(), os.path.join(MODEL_DIR, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, 'discriminator.pth'))


def load_models(generator, discriminator):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'generator.pth'), map_location=device))
    discriminator.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'discriminator.pth'), map_location=device))

    generator.to(device)
    discriminator.to(device)


# Load models for generator and discriminator
if os.path.exists(os.path.join(MODEL_DIR, 'generator.pth')) and os.path.exists(
        os.path.join(MODEL_DIR, 'discriminator.pth')):
    load_models(generator, discriminator)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch #{epoch} in progress...")
    progress_bar = progressbar.ProgressBar()
    d_running_loss = 0
    g_running_loss = 0

    for n_batch, (real_batch, _) in enumerate(progress_bar(data_loader)):
        real_data = Variable(real_batch).to(device)
        fake_noise, _ = noise(real_data.size(0), num_classes)
        fake_data = generator(fake_noise).detach()
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)
        fake_noise, _ = noise(real_batch.size(0), num_classes)
        fake_data = generator(fake_noise)
        g_error = train_generator(g_optimizer, fake_data)
        d_running_loss += d_error.item()
        g_running_loss += g_error.item()

    print(f"Loss (Discriminator): {d_running_loss}")
    print(f"Loss (Generator): {g_running_loss}")

    generate_and_save_images(generator, test_noise, test_labels, epoch, IMAGE_SAVE_DIR, class_names)

    # Save the models after each epoch
    save_models(generator, discriminator)
