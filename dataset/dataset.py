
from dataset.cremad_dataset import cremad_dataset

from dataset.nvGesture_dataset import nvGesture_dataset

def create_dataset(dataset, config=None):
    config = {}
    if dataset == 'CREMAD':
        config["train_file"] = '/data/wxx/MML_dataset/CREMA/annotations/train.csv'
        config["test_file"] = '/data/wxx/MML_dataset/CREMA/annotations/test.csv'
        config["audio_root"] = '/data/wxx/MML_dataset/CREMA/AudioWAV'
        config["visual_root"] = '/data/wxx/MML_dataset/CREMA/Image-01-FPS'
        train_dataset = cremad_dataset(config, mode='train')
        test_dataset = cremad_dataset(config, mode='test')
        return train_dataset, test_dataset
   
    elif dataset == 'NVGesture':
        config['train_file'] = '/data/wxx/MML_dataset/nvGesture/nvgesture_train_correct_v2.lst'
        config['test_file'] = '/data/wxx/MML_dataset/nvGesture/nvgesture_test_correct_v2.lst'
        config['video_root'] = '/data/wxx/MML_dataset/nvGesture/Video_data'
        train_dataset = nvGesture_dataset(config, mode='train')
        test_dataset = nvGesture_dataset(config, mode='test')
        return train_dataset, test_dataset
        
