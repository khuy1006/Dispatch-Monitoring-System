# EatLab - Food Detection & Classification System

AI-powered system for food detection and classification using YOLO models and computer vision.

## 🎯 Overview

EatLab is a comprehensive food analysis system that can:
- **Detect** food objects in images/videos
- **Classify** food types (empty plates, kakigori, filled plates)
- **Monitor** food consumption patterns
- **Visualize** analysis results

## 🏗️ Architecture

```
main/
├── 🚀 app.py                    # Main application
├── 📊 components/               # Core modules
│   ├── classifier.py           # Food classification
│   ├── detector.py             # Object detection
│   └── visualizer.py           # Result visualization
├── ⚙️ config.py                # System configuration
├── 🤖 models/                  # Trained AI models 
│   ├── classification.pt       # Classification model
│   └── detection.pt           # Detection model
├── 📁 data/                    # Data storage
├── 🐳 docker-compose.yml       # Container orchestration
├── 📈 grafana/                 # Monitoring dashboards
└── 🎓 training/                # Model training environment
```

## 🚀 Quick Start

### Using Docker (Recommended)
```bash
# Start the system
docker-compose up -d

# Access the application
http://localhost:5000
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🎓 Model Training

For training new models or retraining existing ones:

```bash
cd training/
python setup.py      # Setup training environment
python train.py      # Train models
```

📚 **See [training/README.md](training/README.md) for detailed training instructions.**

## 📊 Features

- **Real-time Processing**: Live video analysis
- **Batch Processing**: Process multiple images/videos
- **Multi-class Classification**: Empty, kakigori, filled plates
- **Monitoring Dashboard**: Grafana integration
- **REST API**: HTTP endpoints for integration
- **Docker Support**: Containerized deployment

## 🔧 Configuration

Main configuration in `config.py`:
- Model paths and parameters
- Processing settings
- API endpoints
- Database connections

## 🐛 Troubleshooting

- **GPU Issues**: Check CUDA installation
- **Memory Issues**: Reduce batch size in config
- **Model Loading**: Ensure models exist in `models/`
- **Docker Issues**: Check `docker-compose logs`

## 🤝 Contributing

1. Train models using `training/` environment
2. Test changes locally
3. Update documentation
4. Submit pull request

---

**📖 Documentation**: See individual component READMEs for detailed information.  
**🎓 Training**: See [training/README.md](training/README.md) for model training.  
**📥 Download Model**: [Google Drive](https://drive.google.com/drive/folders/1s4JPTj3K5nvwP1XI6yHi5PfyWsqVBBfr)