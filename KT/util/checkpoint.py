import os

import torch

import copy

class CheckpointManager:
    def __init__(self, save_dir='checkpoints'):
        self.save_dir = save_dir

    @staticmethod
    def match(file_name,parameters):
        filename_parts = os.path.splitext(file_name)[0].split('-')
        filename_hyperparameters = {filename_parts[i]: filename_parts[i + 1] for i in range(0, len(filename_parts), 2)}
        return filename_hyperparameters == parameters
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, model_name, dataset, hyperparameters, extra_info=None,
                        save_dir='checkpoints'):
        """
        Save a checkpoint of the model and optimizer.

        Args:
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            epoch (int): The current training epoch.
            model_name (str): A name or identifier for the model.
            dataset (str): A name or identifier for the dataset.
            hyperparameters (dict): A dictionary containing hyperparameter settings.
            extra_info (str, optional): Additional information to include in the checkpoint name.
            save_dir (str, optional): Directory to save the checkpoints. Default is 'checkpoints'.
        """
        # Construct a string representation of hyperparameters
        for k,v in hyperparameters.items():
            try:
                number = float(v)
                formatted_content = "{:.0f}".format(number)

            except ValueError:
                formatted_content = v
            hyperparameters[k] = formatted_content
        hyperparam_str = "-".join([f"{key}-{value}" for key, value in hyperparameters.items()])

        checkpoint_dir = os.path.join(save_dir, model_name, dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_name = hyperparam_str
        if extra_info:
            checkpoint_name += f'-{extra_info}'
        checkpoint_name += '.pth'

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        print(checkpoint_path)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparameters': hyperparameters,
        }

        torch.save(checkpoint, checkpoint_path)

    @staticmethod
    def load_checkpoint(model, optimizer, checkpoint_path):
        """
        Load a checkpoint into the model and optimizer.

        Args:
            model (torch.nn.Module): The model to load the checkpoint into.
            optimizer (torch.optim.Optimizer): The optimizer to load the checkpoint into.
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            Tuple containing (model, optimizer, epoch, hyperparameters)
        """
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        hyperparameters = checkpoint['hyperparameters']

        return model, optimizer, epoch, hyperparameters

    @staticmethod
    def load_checkpoint_by_hyperparameters(model, optimizer, directory, model_name, dataset, hyperparameters):
        """
        Load a checkpoint from a directory based on hyperparameter settings.

        Args:
            model (torch.nn.Module): The model to load the checkpoint into.
            optimizer (torch.optim.Optimizer): The optimizer to load the checkpoint into.
            directory (str): The directory containing checkpoint files.
            hyperparameters (dict): A dictionary containing hyperparameter settings.

        Returns:
            Tuple containing (model, optimizer, epoch) if a matching checkpoint is found,
            or ("Failed to load", None, None) if no matching checkpoint is found.
        """
        # Construct a string representation of hyperparameters to match checkpoint filenames
        for k,v in hyperparameters.items():
            try:
                number = float(v)
                formatted_content = "{:.0f}".format(number)

            except ValueError:
                formatted_content = v
            hyperparameters[k] = formatted_content
        hyperparam_str = "-".join([f"{key}-{value}" for key, value in hyperparameters.items()])

        directory = os.path.join(directory, model_name, dataset)
        # List checkpoint files in the directory
        checkpoint_files = os.listdir(directory)

        # Look for a checkpoint file matching the hyperparameters

        matching_checkpoint = None
        for checkpoint_file in checkpoint_files:

            if hyperparam_str == os.path.splitext(checkpoint_file)[0]:
                matching_checkpoint = checkpoint_file
                break

        if matching_checkpoint:
            checkpoint_path = os.path.join(directory, matching_checkpoint)
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))

            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            epoch = checkpoint['epoch']
            assert epoch >= 0
            return model, optimizer, epoch
        else:
            print("Failed to load")
            return "Failed to load", None, None

if __name__ == '__main__':
    from KT.models.SAKT import SAKT

    checkpoint_manager = CheckpointManager()
    model = SAKT(q_num=100, seq_len=200, embed_dim=50, heads=1,
                 dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    hyperparameters_to_load = model.get_hyperparameters()

    CheckpointManager.save_checkpoint(
        model=model,
        model_name='SAKT-Demo',
        dataset='KT-Dataset-Demo',
        optimizer=optimizer,
        epoch=10,
        hyperparameters=model.get_hyperparameters(),
        save_dir='../../Checkpoints'
    )
    model = SAKT(q_num=1001, seq_len=200, embed_dim=50, heads=1,
                 dropout=0.2)
    loaded_model, loaded_optimizer, loaded_epoch = CheckpointManager.load_checkpoint_by_hyperparameters(
        model=model,
        optimizer=optimizer,
        model_name='SAKT-Demo',
        dataset='KT-Dataset-Demo',
        directory='../../Checkpoints',
        hyperparameters=model.get_hyperparameters()
    )

    if loaded_model == "Failed to load":
        print("Checkpoint not found.")
    else:
        print("Checkpoint loaded successfully.")
