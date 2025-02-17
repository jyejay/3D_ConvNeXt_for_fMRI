# Running Instructions

### Prerequisites
'Before running the code, you need to modify the data path in the configuration file:
```python
# In data/task_configs.py
ROOT_DIR = "your_data_path"  # Change this to your data directory
```

```
python main.py --task WM --batch_size 32 --learning_rate 1e-4 --num_epochs 100 --device cuda:1 --model_dims 64 128 256 512
```


### Data arguments

'--task'
### Model arguments

'--model_depths' # default=[3, 3, 9, 3]

'--model_dims' # default=[96, 192, 384, 768]

'--drop_path_rate' # default=0.


### Training arguments

'--batch_size' # default=32

'--learning_rate' # default=1e-4

'--num_epochs' # default=100

'--num_workers' # default=8

'--device' # default='cuda'

### Other arguments

'--seed' # default=1234

'--checkpoint_dir' # default='checkpoints'

'--project_name' # default='hcp_classification'


