import os

import torch


class CheckpointManager:
    def __init__(self, save_dir='checkpoints'):
        self.save_dir = save_dir

    @staticmethod
    def save_checkpoint(model, optimizer, epoch, model_name, dataset_name, hyperparameters, extra_info=None,
                        save_dir='checkpoints'):
        """
        Save a checkpoint of the model and optimizer.

        Args:
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            epoch (int): The current training epoch.
            model_name (str): A name or identifier for the model.
            dataset_name (str): A name or identifier for the dataset.
            hyperparameters (dict): A dictionary containing hyperparameter settings.
            extra_info (str, optional): Additional information to include in the checkpoint name.
            save_dir (str, optional): Directory to save the checkpoints. Default is 'checkpoints'.
        """
        # Construct a string representation of hyperparameters
        hyperparam_str = "_".join([f"{key}_{value}" for key, value in hyperparameters.items()])

        checkpoint_dir = os.path.join(save_dir, model_name, dataset_name, hyperparam_str)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_name = f'epoch{epoch}'
        if extra_info:
            checkpoint_name += f'_{extra_info}'
        checkpoint_name += '.pth'

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

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
    def load_checkpoint_by_hyperparameters(model, optimizer, directory, hyperparameters):
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
        hyperparam_str = "_".join([f"{key}_{value}" for key, value in hyperparameters.items()])

        # List checkpoint files in the directory
        checkpoint_files = os.listdir(directory)

        # Look for a checkpoint file matching the hyperparameters
        matching_checkpoint = None
        for checkpoint_file in checkpoint_files:
            if hyperparam_str in checkpoint_file:
                matching_checkpoint = checkpoint_file
                break

        if matching_checkpoint:
            checkpoint_path = os.path.join(directory, matching_checkpoint)
            checkpoint = torch.load(checkpoint_path)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            epoch = checkpoint['epoch']

            return model, optimizer, epoch
        else:
            return "Failed to load", None, None

if __name__=='__main__':
    checkpoint_manager = CheckpointManager()
    model = YourModelClass()  # Replace with your model class
    optimizer = YourOptimizerClass()  # Replace with your optimizer class
    hyperparameters_to_load = {'A': 1, 'B': 2, 'C': 3}

    loaded_model, loaded_optimizer, loaded_epoch = checkpoint_manager.load_checkpoint(model, optimizer, 'checkpoints',
                                                                                      hyperparameters_to_load)

    if loaded_model == "Failed to load":
        print("Checkpoint not found.")
    else:
        print("Checkpoint loaded successfully.")
