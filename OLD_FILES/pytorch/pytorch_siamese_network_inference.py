import torch
from PIL import Image
from torchvision.transforms import transforms
import torch.nn as nn

from siamese_pytorch import SiameseModel
from siamese_pytorch import calculate_distances


def infer():
    embedding_dim = 128
    model = SiameseModel(embedding_dim=embedding_dim)
    model.load_state_dict(
        torch.load('checkpoints/checkpoint_350.pt')['model_state_dict']
    )
    model.eval()

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    img1 = Image.open('../dataset/mnist_sample/1/img_12.jpg')
    img2 = Image.open('../dataset/mnist_sample/1/img_527.jpg')
    img3 = Image.open('../dataset/mnist_sample/5/img_144.jpg')

    img1 = data_transforms(img1).unsqueeze(dim=0)
    img2 = data_transforms(img2).unsqueeze(dim=0)
    img3 = data_transforms(img3).unsqueeze(dim=0)

    embedding_1 = model.forward_single_minibatch(img1)
    embedding_2 = model.forward_single_minibatch(img2)
    embedding_3 = model.forward_single_minibatch(img3)

    # print(embedding_1.shape)
    # print(embedding_2.shape)
    # print(embedding_3.shape)

    with torch.no_grad():
        ap_distance, an_distance = calculate_distances(
            anchor_embedding=embedding_1,
            positive_embedding=embedding_2,
            negative_embedding=embedding_3,
        )
        print(ap_distance, an_distance)

        cosine_similarity = nn.CosineSimilarity()
        print(f'Similar Score: {cosine_similarity(embedding_1, embedding_2)}')
        print(f'Different Score: {cosine_similarity(embedding_1, embedding_3)}')


if __name__ == '__main__':
    infer()
