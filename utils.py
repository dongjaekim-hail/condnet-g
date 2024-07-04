# download imagenet
import huggingface_hub
from datasets import load_dataset

def download_imagenet(dir2save):
    # Download the dataset imagenet 1k
    dataset = load_dataset('imagenet-1k', data_dir=dir2save, trust_remote_code=True)

    # Print the dataset
    print(dataset)

    # return true if it downloaded successfully or already exists
    return True


if __name__ == '__main__':
    dir2save = 'D:/imagenet-1k'
    # dir2save = '/Users/dongjaekim/Documents/imagenet'
    dataset = load_dataset('imagenet-1k',num_proc=8, trust_remote_code=True)
    dataset.save_to_disk(dir2save, num_proc=12)
    # dataset_train = load_dataset('imagenet-1k', data_dir=dir2save, split='train', num_proc=8)
    print('')