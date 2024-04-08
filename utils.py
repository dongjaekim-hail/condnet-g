import huggingface_hub


def download_imagenet(dir2save):
    # Download the dataset imagenet 1k
    dataset = huggingface_hub.dataset("imagenet")

    # Print the dataset
    print(dataset)

    # return true if it downloaded successfully or already exists
    return True


if __name__ == '__main__':
    download_imagenet()
