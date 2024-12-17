from dataset import Flickr8k, MSVD

def test_load_flickr8k_dataset(json_file, root):
    """
    Test if the user successfully loads the flickr8k dataset.

    Parameters:
    - json_file: str, the path to the JSON file (either 'flickr8k.json' for flickr8k-expert dataset or 'crowdflower_flickr8k.json' for flickr8k-cf dataset)
    - root: str, the root directory where the dataset is stored

    Returns:
    - None, but prints out the sample details
    """
    dataset = Flickr8k(json_file=json_file, root=root, load_images=False)
    sample_idx = 0  # Change this index to get different samples
    sample = dataset[sample_idx]
    print("Image Path:", sample[0])
    print("Candidate Caption:", sample[1])
    print("References:", sample[2])
    print("Human Score:", sample[3])
    
def test_load_msvd_dataset(json_file, root):
    """
    Test if the user successfully loads the MSVD dataset.

    Parameters:
    - json_file: str, the path to the JSON file
    - root: str, the root directory where the dataset is stored

    Returns:
    - None, but prints out the sample details
    """
    dataset = MSVD(json_file=json_file, root=root, load_videos=False)
    sample_idx = 0  # Change this index to get different samples
    sample = dataset[sample_idx]
    print("Video Path:", sample[0])
    print("Candidate Caption:", sample[1])
    print("References:", sample[2])
    print("Human Score:", sample[3])

if __name__ == "__main__":
    # Example usage for Flickr8k
    json_file_expert = 'flickr8k.json'
    json_file_cf = 'crowdflower_flickr8k.json'
    root_flickr8k = 'data/flickr8k/'

    print("Testing flickr8k-expert dataset:")
    test_load_flickr8k_dataset(json_file_expert, root_flickr8k)

    print("\nTesting flickr8k-cf dataset:")
    test_load_flickr8k_dataset(json_file_cf, root_flickr8k)

    # Example usage for MSVD
    json_file_msvd = 'MSVD-Eval.json'
    root_msvd = 'data/YouTubeClips/'

    print("\nTesting MSVD dataset:")
    test_load_msvd_dataset(json_file_msvd, root_msvd)