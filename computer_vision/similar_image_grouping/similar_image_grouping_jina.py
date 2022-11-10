import os

from jina import DocumentArray, Document
import torchvision


def preproc(d: Document):
    return (
        d.load_uri_to_image_blob()  # load
        .set_image_blob_shape(shape=(224, 224))
        .set_image_blob_normalization()  # normalize color
        .set_image_blob_channel_axis(-1, 0)
    )  # switch color axis


if __name__ == '__main__':
    base_path = r'C:\Users\computer\pictures'
    search_extension = '*.jpg'
    query_image = '1.jpg'

    docs = DocumentArray.from_files(os.path.join(base_path, search_extension)).apply(preproc)

    model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
    docs.embed(model, device='cuda')  # embed via GPU to speedup

    # q = (
    #     Document(uri=os.path.join(base_path, query_image))  # build query image & preprocess
    #     .load_uri_to_image_blob()
    #     .set_image_blob_shape(shape=(512, 512))
    #     .set_image_blob_normalization()
    #     .set_image_blob_channel_axis(-1, 0)
    # )
    #

    q = preproc(Document(uri=os.path.join(base_path, query_image)))
    q.embed(model)  # embed
    q.match(docs)  # find top-20 nearest neighbours, done!

    for m in q.matches:
        m.set_image_blob_channel_axis(0, -1).set_image_blob_inv_normalization()
    q.matches.plot_image_sprites()
