# %% [markdown]
"""
Colab Team Starter Script

Use this in Google Colab by copying cells into a notebook,
or run sections manually in VS Code/Jupyter.
"""

# %%
# 1) Install project dependencies (run in Colab notebook)
# !pip install -q -r requirements.txt

# %%
# 2) Mount Google Drive (run in Colab)
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# 3) Clone or update repository (run in Colab)
# !git clone https://github.com/<org-or-user>/<repo-name>.git
# %cd <repo-name>
# !git pull

# %%
# 4) Configure shared dataset path
DATA_CSV = '/content/drive/MyDrive/<team-shared-folder>/Combined Data.csv'

# %%
# 5) Run smoke checks
import os
import subprocess


def run_cmd(cmd):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")


def ensure_data_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Update DATA_CSV to your shared Drive location."
        )


if __name__ == '__main__':
    ensure_data_exists(DATA_CSV)

    # Optional: export path so scripts can consume it if later needed.
    os.environ['DATA_CSV'] = DATA_CSV

    # Run current project checks
    run_cmd('python smoke_check.py')
    run_cmd('python mini_training_3epoch_check.py')

    print('\nColab starter flow completed successfully.')
