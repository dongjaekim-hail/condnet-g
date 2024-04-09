import huggingface_hub
from datasets import load_dataset

def download_imagenet(dir2save):
    # Download the dataset imagenet 1k
    dataset = load_dataset('imagenet-1k', data_dir=dir2save)

    # Print the dataset
    print(dataset)

    # return true if it downloaded successfully or already exists
    return True


if __name__ == '__main__':
    dir2save = 'D:/imagenet-1k'
    dataset = load_dataset('imagenet-1k',num_proc=8)
    dataset.save_to_disk(dir2save, num_proc=24)
    # dataset_train = load_dataset('imagenet-1k', data_dir=dir2save, split='train', num_proc=8)
    print('')