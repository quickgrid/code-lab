"""This implementation does not work yet.

References
    - https://github.com/fangpin/siamese-pytorch
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    - https://discuss.pytorch.org/t/dataloader-for-a-siamese-model-with-concatdataset/66085
"""
import os
from typing import Dict, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from fire import Fire
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class SiameseModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(SiameseModel, self).__init__()
        self.base_network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=256 * 2 * 2, out_features=4096),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.Softmax(),
        )

    def forward_single_image(self, x: Tensor) -> Tensor:
        x = self.base_network(x)
        x = self.flatten(x)
        x = self.embedding_layer(x)
        return x

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.forward_single_image(x1)
        x2 = self.forward_single_image(x2)
        distance_vector = torch.abs(x1 - x2)
        out = self.output(distance_vector)
        return out


class SiameseDataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms = None) -> None:
        self.root_dir = root_dir
        self.transform = transform

        self.class_name_to_images_list = dict()
        self.class_name_to_idx_map = dict()
        self.class_idx_to_name_map = dict()
        self.class_name_file_count_list = dict()

        class_names = os.listdir(root_dir)
        self.class_names_len = len(class_names)

        for idx, class_name in enumerate(class_names):
            self.class_name_to_idx_map[class_name] = idx
            self.class_idx_to_name_map[idx] = class_name
            self.class_name_to_images_list[class_name] = os.listdir(os.path.join(root_dir, class_name))
            self.class_name_file_count_list[class_name] = len(
                self.class_name_to_images_list[class_name]
            )

    def __len__(self) -> int:
        return self.class_names_len

    def __getitem__(self, idx: int) -> Dict[str, Union[Image.Image, int, bool]]:
        """Alternates selection between similar class and different class images.

        Args:
            idx: A sub folder index.

        Returns:
            Dictionary of two PIL.Image and two integer label.
        """
        current_class_name = self.class_idx_to_name_map[idx]
        current_dir_file_count = self.class_name_file_count_list[current_class_name]

        # Alternate between images from same class.
        if idx % 2 == 1:
            random_image_idx1 = np.random.randint(low=0, high=current_dir_file_count)
            random_image_idx2 = np.random.randint(low=0, high=current_dir_file_count)

            image1_path = os.path.join(
                self.root_dir,
                current_class_name,
                self.class_name_to_images_list[current_class_name][random_image_idx1]
            )
            image2_path = os.path.join(
                self.root_dir,
                current_class_name,
                self.class_name_to_images_list[current_class_name][random_image_idx2]
            )

            label1 = idx
            label2 = idx

            # assert label1 != label2, 'Something went wrong with file selection.'

            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)
            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)

            sample = {
                'image1': image1,
                'image2': image2,
                'label1': label1,
                'label2': label2,
                'same_type': True,
            }

        else:
            random_image_idx2 = np.random.randint(low=0, high=self.class_names_len)
            # Think of better logic to handle this.
            while idx == random_image_idx2:
                random_image_idx2 = np.random.randint(low=0, high=self.class_names_len)
            different_class_name = self.class_idx_to_name_map[random_image_idx2]

            random_image_idx1 = np.random.randint(low=0, high=current_dir_file_count)

            image1_path = os.path.join(
                self.root_dir,
                current_class_name,
                self.class_name_to_images_list[current_class_name][random_image_idx1]
            )
            image2_path = os.path.join(
                self.root_dir,
                different_class_name,
                self.class_name_to_images_list[different_class_name][random_image_idx2]
            )

            label1 = idx
            label2 = random_image_idx2

            # assert label1 == label2, 'Something went wrong with file selection.'

            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)
            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)

            sample = {
                'image1': image1,
                'image2': image2,
                'label1': label1,
                'label2': label2,
                'same_type': True,
            }

        return sample


def contrastive_loss_fn(margin: float):
    def contrastive_loss(y_true, y_pred) -> Tensor:
        y_pred = y_pred.detach().cpu()
        y_true = y_true.detach().cpu()
        print(y_true.shape, y_pred.shape)
        square_pred = np.square(y_pred)
        margin_square = np.square(np.maximum(margin - y_pred, 0))
        return torch.from_numpy(
            np.mean(y_true * square_pred + (1 - y_true) * margin_square)
        )
    return contrastive_loss


class Trainer:
    def __init__(
            self,
            dataset_path: str,
            batch_size: int = 6,
            num_workers: int = 4,
            learning_rate: float = 0.0001,
            num_epochs: int = 10,
    ):
        super(Trainer, self).__init__()
        self.num_epochs = num_epochs

        data_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = SiameseDataset(
            root_dir=dataset_path,
            transform=data_transforms
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.num_classes = len(os.listdir(dataset_path))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseModel(num_classes=self.num_classes).to(self.device)
        # self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        self.loss_fn = contrastive_loss_fn(margin=0.05)
        self.optimizer_fn = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self) -> None:
        for epoch in range(self.num_epochs):
            with tqdm(self.train_loader) as tqdm_epoch:
                for batch_idx, data in enumerate(tqdm_epoch):
                    # print(batch_idx, data)
                    image1 = data['image1']
                    image2 = data['image2']
                    label1 = data['label1']
                    label2 = data['label2']

                    image1 = image1.to(self.device)
                    image2 = image2.to(self.device)

                    print(image1.shape)
                    print(image2.shape)

                    # onehot_label1 = F.one_hot(label1)
                    # onehot_label2 = F.one_hot(label2)

                    output = self.model.forward(image1, image2)
                    print(output)

                    predicted_index = torch.argmax(output, dim=0)
                    predicted_index_onehot = F.one_hot(predicted_index)

                    label_merging_tensor = torch.zeros_like(data['same_type'], dtype=torch.long)
                    print(predicted_index)
                    print(label_merging_tensor)
                    print(data['same_type'])
                    # print(onehot_label1)
                    # print(onehot_label2)
                    print('%' * 60)
                    for i in range(len(data['same_type'])):
                        if data['same_type'][i]:
                            label_merging_tensor[i] = label1[i]
                        else:
                            label_merging_tensor[i] = label2[i]


                    print(label_merging_tensor)
                    print(predicted_index_onehot)
                    label_merging_tensor = F.one_hot(label_merging_tensor, num_classes=self.num_classes)
                    loss = self.loss_fn(y_true=label_merging_tensor, y_pred=predicted_index_onehot)

                    self.optimizer_fn.zero_grad()
                    loss.backward()
                    self.optimizer_fn.step()

                    fig, ax = plt.subplots(1, 2)
                    ax[0].imshow(np.transpose(image1[0].detach().cpu().numpy(), (1, 2, 0)))
                    ax[1].imshow(np.transpose(image2[0].detach().cpu().numpy(), (1, 2, 0)))
                    plt.show()


if __name__ == '__main__':
    # siamese_network = SiameseModel()
    # print(siamese_network)
    # print(list(siamese_network.parameters()))

    # Fire(Trainer)

    trainer = Trainer(
        dataset_path='../dataset/mnist_sample'
    )
    trainer.train()
