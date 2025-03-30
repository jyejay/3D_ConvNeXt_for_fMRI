# Running Instructions

### Prerequisites
#### Data Directory Structure
```
# ROOT_DIR = "your_data_path/" 
└── EMOTION/
|    └── subject_id/              # e.g., 100206
|        ├── fear/                # Label: 0
|        │   └── LR/
|        │       ├── 0001.npy
|        │       ├── 0002.npy
|        │       └── ...
|        └── neut/                # Label: 1
|            └── LR/
|                ├── 0001.npy
|                ├── 0002.npy
|                └── ...
└── GAMBLING/
|    └── subject_id/             
|        ├── loss/                # Label: 0
|        └── win/                 # Label: 1
└── LANGUAGE/
|    └── subject_id/             
|        ├── math/                # Label: 0
|        └── story/               # Label: 1
└── MOTOR/
|    └── subject_id/             
|        ├── lf/                  # Label: 0
|        ├── lh/                  # Label: 1
|        ├── rf/                  # Label: 2
|        ├── rh/                  # Label: 3
|        └── t/                   # Label: 4
└── RELATIONAL/
|    └── subject_id/             
|        ├── match/               # Label: 0
|        └── relation/            # Label: 1
└── SOCIAL/
|    └── subject_id/             
|        ├── mental/              # Label: 0
|        └── rnd/                 # Label: 1
└── WM/
    └── subject_id/             
        ├── 0bk_body/            # Label: 0
        ├── 0bk_faces/           # Label: 1
        ├── 0bk_places/          # Label: 2
        ├── 0bk_tools/           # Label: 3
        ├── 2bk_body/            # Label: 4
        ├── 2bk_faces/           # Label: 5
        ├── 2bk_places/          # Label: 6
        └── 2bk_tools/           # Label: 7

```
#### Data Description
- Each `.npy` file contains a 3D brain image
- Labels are determined by the directory name (e.g., 'fear': 0, 'neut': 1)
- Data is organized by subject IDs and task types

#### Data Loading Process
 - **Dataset Class (`HCPDataset`):**
   - Loads 3D brain images from `.npy` files
   - Automatically assigns labels based on directory structure
   - Returns data as torch tensors with shape `[1, D, H, W]`

Before running the code, you need to modify the data path in the configuration file:

```python
# In data/task_configs.py
ROOT_DIR = "your_data_path"  # Change this to your data directory
```

```
python main.py --task EMOTION --batch_size 32 --learning_rate 1e-4 --num_epochs 100 --device cuda:1 --model_dims 64 128 256 512 --drop_path_rate 0.1
```


### Data arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | HCP task | - |

### Model arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_depths` | Model layer depths | `[3, 3, 9, 3]` |
| `--model_dims` | Model layer dimensions | `[96, 192, 384, 768]` |
| `--drop_path_rate` | Drop path rate | `0.` |

### Training arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--batch_size` | Batch size for training | `32` |
| `--learning_rate` | Learning rate | `1e-4` |
| `--num_epochs` | Number of epochs | `100` |
| `--num_workers` | Number of workers | `8` |
| `--device` | Device for training | `cuda` |

### Other arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--seed` | Random seed | `1234` |
| `--checkpoint_dir` | Checkpoint directory | `checkpoints` |
| `--project_name` | Project name | `hcp_classification` |
