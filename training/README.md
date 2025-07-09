# YOLO Classification Training Environment

Dedicated training environment for YOLO classification model using data from `data_wrong` directory.

## 🏗️ Structure

```
training/
├── requirements.txt          # Training-specific dependencies
├── config.yaml              # Training configuration parameters  
├── train.py                 # Main training script
├── utils/                   # Training utilities
│   ├── __init__.py
│   ├── data_split.py        # Dataset splitting
│   ├── augmentation.py      # Data augmentation
│   └── trainer.py           # YOLO training
└── README.md               # This guide
```

## 🚀 Setup

### 1. Create Virtual Environment

```bash
# Windows
python -m venv training_env
training_env\Scripts\activate

# Linux/Mac  
python -m venv training_env
source training_env/bin/activate
```

### 2. Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

## ⚙️ Configuration

Edit `config.yaml` to adjust training parameters:

- **Data paths**: Source data and output directories
- **Split ratios**: Train/val/test split ratios 
- **Augmentation**: Data augmentation methods
- **Training parameters**: Epochs, batch size, device, etc.

## 🎯 Usage

### Run Complete Pipeline:

```bash
python train.py
```

### Run Individual Steps:

```bash
# Data splitting only
python train.py --no-augment --no-train

# Augmentation only (requires split data)
python train.py --no-split --no-train  

# Training only (requires prepared data)
python train.py --no-split --no-augment
```

### Additional Options:

```bash
# Use different config file
python train.py --config my_config.yaml

# Force re-split data (remove existing results)
python train.py --force-split

# Show help
python train.py --help
```

## 📊 Input Data

Script is configured to use data from `../data_wrong/` with structure:

```
data_wrong/
├── empty/       # Class 1: Empty plates
├── kakigori/    # Class 2: Kakigori  
└── not_empty/   # Class 3: Plates with food
```

## 📈 Output

- **Split data**: Dataset splits saved in `dish_classification_split/`
- **Trained model**: Model saved in `runs/classify/train*/`
- **Logs**: Training logs saved in `training.log`

## 🔧 Customization

### Add New Augmentation:

1. Add new method to `utils/augmentation.py`
2. Update `augmentation_methods` dictionary
3. Add to config.yaml

### Change Model:

Edit `training.model_name` in config.yaml:
- `yolov8n-cls.pt` (nano - fastest)
- `yolov8s-cls.pt` (small)  
- `yolov8m-cls.pt` (medium)
- `yolov8l-cls.pt` (large)
- `yolov8x-cls.pt` (extra large - most accurate)

## 🐛 Troubleshooting

### GPU Not Detected:
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU instead
# In config.yaml: device: 'cpu'
```

### Memory Error:
- Reduce `batch_size` in config.yaml
- Use smaller model (yolov8n instead of yolov8l)

### Import Error:
```bash
# Ensure environment is activated
pip install -r requirements.txt
```

## 📝 Logs

Training generates:
- Console logs with progress bars
- File logs (`training.log`) if enabled
- YOLO training logs in `runs/`

## 🎯 Complete Example

```bash
# 1. Activate environment
cd training
source training_env/bin/activate  # Linux/Mac
# or training_env\Scripts\activate  # Windows

# 2. Run training
python train.py

# 3. Results will include:
# - Data split in dish_classification_split/
# - Trained model in runs/classify/train*/
# - Logs in training.log
```

## 🔄 Quick Setup

Use the automated setup script:

```bash
python setup.py
```

This will:
- Create virtual environment
- Install all dependencies
- Validate installation
- Create activation scripts

Then simply run:
```bash
./activate.sh    # Linux/Mac
# or activate.bat # Windows
python train.py
``` 