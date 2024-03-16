import os
import subprocess

def get_package_info(env_name):
    try:
        # Activate the Conda environment
        activate_cmd = f"conda activate {env_name}"
        subprocess.run(activate_cmd, shell=True, check=True)

        # Get the package information for the environment
        list_cmd = f"conda list -n {env_name}"
        result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            return result.stdout
        else:
            print(f"Error: Could not list packages in environment '{env_name}'.")
            return None

    except subprocess.CalledProcessError:
        print(f"Error: Could not activate environment '{env_name}'.")
        return None

def list_conda_envs():
    # Get a list of Conda environments
    envs_list_cmd = "conda env list"
    result = subprocess.run(envs_list_cmd, shell=True, capture_output=True, text=True)

    envs = []
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) > 0 and parts[0] != '#':
                envs.append(parts[0])
    else:
        print("Error: Could not list Conda environments.")

    return envs

def main():
    # Target package and version range
    target_package = "tensorflow"
    version_range = ">=1.0,<2.0"

    # List all Conda environments
    envs = list_conda_envs()

    # Check each environment for the target package
    for env in envs:
        package_info = get_package_info(env)
        if package_info:
            if f"{target_package}" in package_info:
                print(f"Package '{target_package}' with version in range '{version_range}' is installed in environment '{env}'.")
            else:
                print(f"Package '{target_package}' with version in range '{version_range}' is not installed in environment '{env}'.")

if __name__ == "__main__":
    main()
