# Colab Team Setup (Shared Workflow)

This guide lets multiple teammates run the same code reliably in Google Colab.

## 1. One-time team setup

1. Create a GitHub repository and upload this project.
2. Add all teammates as GitHub collaborators.
3. Create one shared Google Drive folder for data and outputs.
4. Put the dataset in shared Drive as: `Combined Data.csv`.

## 2. Colab setup for each teammate

In a fresh Colab notebook, run:

```python
!pip install -q -r requirements.txt
```

Then mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Clone repo (first time):

```python
!git clone https://github.com/<org-or-user>/<repo-name>.git
%cd <repo-name>
```

Pull latest changes (later sessions):

```python
%cd /content/<repo-name>
!git pull
```

## 3. Shared data path

Use a single team path variable in notebooks/scripts:

```python
DATA_CSV = '/content/drive/MyDrive/<team-shared-folder>/Combined Data.csv'
```

## 4. Quick correctness check in Colab

```python
!python smoke_check.py
```

## 5. Short training sanity run

```python
!python mini_training_3epoch_check.py
```

## 6. Team workflow rules (recommended)

1. Create a feature branch for each task: `feature/<name>-<task>`.
2. Keep commits small and focused.
3. Open pull requests into `main`.
4. Review before merge.
5. Pull latest `main` before starting a new task.

## 7. Notes

- If Colab disconnects, rerun setup cells.
- Keep large model files and outputs in Drive, not GitHub.
- `requirements.txt` and `.gitignore` are included in this project.
