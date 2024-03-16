from abc import ABC, abstractmethod
from torch.utils.data import Dataset,DataLoader
import torch
class KTDatasetBase(ABC):
    @abstractmethod
    def get_sequences(self):
        pass

    @abstractmethod
    def get_users(self):
        pass

    @abstractmethod
    def get_skill_name(self,sid):
        pass


    @abstractmethod
    def get_features(self):
        pass

    @abstractmethod
    def get_q_matrix(self):
        pass


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming data[idx] is a tuple containing a, b, c and additional features d, e, f,...
        sample = self.data[idx]
        a, b, c = sample[:3]
        additional_features = sample[3:]  # Additional features

        # Convert everything into PyTorch tensors
        a = torch.tensor(a)
        b = torch.tensor(b)
        c = torch.tensor(c)

        # Convert additional features into a dictionary
        additional_features_dict = {f'feature_{i}': torch.tensor(feature) for i, feature in
                                    enumerate(additional_features, start=4)}

        return a, b, c, additional_features_dict

# Example usage
# Assuming data is a list of tuples where each tuple contains a, b, c and additional features d, e, f,...
data = [(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)]  # Example data
dataset = CustomDataset(data)

# Example of how to use DataLoader to iterate over the dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

#Iterate over batches
for batch in dataloader:
    a, b, c, additional_features_dict = batch
    # Access additional features using keys
    feature_d = additional_features_dict['feature_4']
    feature_e = additional_features_dict['feature_5']
    feature_f = additional_features_dict['feature_6']
    # Your training/validation loop here
    print(a,b,c,feature_d,feature_e,feature_f)





