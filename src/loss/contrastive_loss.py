
import torch.nn.functional as TF

def contrastive_loss(encoded_x, encoded_x_same, encoded_x_diff):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html
    dist_same = 1 - TF.cosine_similarity(encoded_x, encoded_x_same)
    dist_diff = 1 - TF.cosine_similarity(encoded_x, encoded_x_diff)

    # ???
    # dist_same = TF.mse_loss(encoded_x, encoded_x_same)
    # dist_diff = 1 - TF.mse_loss(encoded_x, encoded_x_diff)
    # sum = dist_same - dist_diff

    sum = dist_same + dist_diff
    return sum


def contrastive_classification_loss(encoded_x, encoded_x_same, encoded_x_diff, output_x, target, classification_loss):
    loss = classification_loss(output_x, target)
    loss += contrastive_loss(encoded_x, encoded_x_same, encoded_x_diff)
    return loss


# TODO: modify
def contrastive_reconstruction_loss(encoded_x, encoded_x_same, encoded_x_diff, output_x, target, classification_loss):
    loss = classification_loss(output_x, target)
    loss += contrastive_loss(encoded_x, encoded_x_same, encoded_x_diff)
    return loss