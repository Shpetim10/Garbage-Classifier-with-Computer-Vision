# ♻️ Smart Garbage Classifier

An intelligent web application that uses deep learning to classify waste items and provide recycling guidance.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **Multi-Image Upload**: Process multiple images at once
- **Camera Support**: Take photos directly in the app
- **Smart Classification**: AI-powered garbage classification with 12 categories
- **Confidence Scoring**: Shows prediction confidence levels
- **Out-of-Scope Detection**: Identifies items outside training data
- **Recyclability Guide**: Comprehensive recycling instructions for each item
- **User Feedback**: Collect feedback to improve the model
- **Batch Processing**: Organize and download classified images
- **Knowledge Base**: Search and browse recycling information
- **Beautiful UI**: Clean, responsive design with animations

## 🎯 Supported Categories

| Category | Icon | Recyclable |
|----------|------|------------|
| Battery | 🔋 | Yes (Hazardous) |
| Biological | 🍂 | Yes (Compost) |
| Brown Glass | 🍺 | Yes |
| Cardboard | 📦 | Yes |
| Clothes | 👕 | Yes (Textile) |
| Green Glass | 🍾 | Yes |
| Metal | 🥫 | Yes |
| Paper | 📄 | Yes |
| Plastic | ♻️ | Conditional |
| Shoes | 👟 | Yes (Textile) |
| Trash | 🗑️ | No |
| White Glass | 🥛 | Yes |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Trained Keras model file

### Installation

1. **Clone the repository**
```bash
git clone (https://github.com/Shpetim10/Garbage-Classifier-with-Computer-Vision.git)
cd garbage-classifier
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your trained model (This line is applicable only when you do not want to use our trained model)**
```bash
# Place your model file in the models directory
cp /path/to/your/model.keras models/best_model.keras
```

5. **Run the application**
```bash
streamlit run app.py
```

6. **Open your browser**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
garbage-classifier/
├── app.py                      # Main Streamlit application
├── model_utils.py              # Model loading and prediction
├── knowledge_base.py           # Recyclability information system
├── image_processing.py         # Image processing utilities
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── models/
│   └── best_model.keras       # Your trained model
├── feedback/
│   └── predictions.log        # User feedback logs
```

## 🎨 Usage

### Basic Workflow

1. **Upload Images**: Click "Upload Files" or "Use Camera"
2. **Process**: Click "Classify Images" button
3. **Review Results**: View predictions with confidence scores
4. **Check Recyclability**: Expand info sections for recycling instructions
5. **Provide Feedback**: Help improve the model with your feedback
6. **Export**: Download organized ZIP file or CSV report

### Advanced Features

#### Test-Time Augmentation (TTA)
Enable in sidebar for more accurate predictions (slower):
```python
# In sidebar settings
✓ Use Test-Time Augmentation
```

#### Confidence Threshold
Adjust sensitivity in sidebar:
- Lower threshold: More predictions, less confident
- Higher threshold: Fewer predictions, more confident

#### Batch Export
Process multiple images and download as organized ZIP:
1. Upload multiple images
2. Click "Classify Images"
3. Go to "View Results" tab
4. Click "Download ZIP (Organized by Class)"

## 🧠 Model Information

The application uses a deep learning model trained on garbage classification:

- **Architecture**: EfficientNetB3-based CNN
- **Input Size**: 300×300 pixels
- **Classes**: 12 categories
- **Accuracy**: ~98% on test set
- **Framework**: TensorFlow/Keras

## 🌐 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository and branch
5. Deploy!

## 📊 Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 98% |
| F1 Score| 98%  |
| Macro F1-Score| 98% |

## 🔧 Configuration

### Adjust Thresholds

Edit in `model_utils.py`:
```python
classifier = GarbageClassifier(
    confidence_threshold=0.70,      # High confidence cutoff
    out_of_scope_threshold=0.50     # Out-of-scope cutoff
)
```
## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Known Issues

- Large models may not work on Streamlit Cloud free tier
- Camera feature requires HTTPS in production
- 
## 👏 Acknowledgments

- TensorFlow team for the framework
- Streamlit for the amazing web framework
- Dataset creators and maintainers on Kaggle: (https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/Shpetim10/Garbage-Classifier-with-Computer-Vision/issues)
- **Email**: sshabanaj23@epoka.edu.al

## 🌍 Environmental Impact

This project aims to:
- 🌱 Reduce waste going to landfills
- ♻️ Increase recycling rates through education
- 🌏 Promote environmental awareness
- 📊 Provide data for waste management optimization

---

If you find this project helpful, please give it a ⭐ on GitHub!
