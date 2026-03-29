![Banner](https://capsule-render.vercel.app/api?type=waving&color=0:8E2DE2,100:4A00E0&height=250&section=header&text=Deep%20Learning%20Scaffolded%20Project&fontSize=45&fontColor=ffffff)

# 🔥 Project Badges

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-NeuralNetworks-red?logo=pytorch)
![GPU](https://img.shields.io/badge/GPU-Supported-green?logo=nvidia)
![Colab](https://img.shields.io/badge/Google-Colab-yellow?logo=googlecolab)
![Kaggle](https://img.shields.io/badge/Kaggle-Datasets-blue?logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)



# 📌 Overview

This repository contains a **modular deep learning project scaffold** designed to help researchers and developers build scalable machine learning systems.

The project demonstrates a **complete deep learning workflow**, including:

* Data preprocessing
* Feature engineering
* Model building
* Training pipeline
* Evaluation
* Deployment-ready architecture

The scaffold allows developers to **experiment with deep learning models while maintaining clean project structure and reusable components**.

**PROJECT:**
**AI-Based Marine Species Detection and Behaviour Monitoring System**

**SIH HACKATHON PROBLEM STATEMENT 2025:**

Embedded Intelligent Microscopy System (SIH25042/SIH25043): Development of AI-powered systems to identify and count marine organisms for biodiversity assessment.

Marine Carbon Sink Monitoring(SIH25048): Research and tools to manage fish populations as a crucial component of the ocean carbon sink (blue carbon).

Environmental Impact Analysis(SIH250): Solutions to mitigate negative impacts on fish stocks from marine pollution, construction, and aquaculture.

**Marine Animals Multimodal Dataset**

A comprehensive multimodal dataset combining audio recordings and images of 32 marine species.

**Dataset Summary**

Total samples: 24,911

Species: 32

Audio files: 1,357 unique recordings

Images: 581 (309 matched + 272 from iNaturalist)

**Features**

species (string): Species name

label (int32): Numeric label (0–31)

audio (Audio): Audio recording of the species

image (Image): Species image

image_index (int32): Image number for this species

total_images (int32): Total images for this species

source (string): "LLM-Vision-Marine-Animals" or "iNaturalist"



**Data Sources**

Audio: ardavey/marine_ocean_mammal_sound

Matched images: yeyimilk/LLM-Vision-Marine-Animals

Additional images: Downloaded from iNaturalist API with research-grade observations and CC-compatible licenses.

CLASS_NAMES = [ 'Atlantic_Spotted_Dolphin', 'Bearded_Seal', 'Beluga', 'Blue_Whale', 'Bowhead_Whale', 'Common_Dolphin', 'Dugong', 'Fin_Whale', 'Gray_Seal', 'Gray_Whale', 'Harbor_Porpoise', 'Harbor_Seal', 'Harp_Seal', 'Hooded_Seal', 'Humpback_Whale', 'Killer_Whale', 'Leopard_Seal', 'Minke_Whale', 'Narwhal', 'North_Atlantic_Right_Whale', 'Northern_Elephant_Seal', 'Pacific_White_Sided_Dolphin', 'Pantropical_Spotted_Dolphin', 'Pilot_Whale', 'Ribbon_Seal', 'Ringed_Seal', 'Ross_Seal', 'Southern_Elephant_Seal', 'Sperm_Whale', 'Spinner_Dolphin', 'Spotted_Seal', 'Weddell_Seal' ]

<img width="479" height="274" alt="image" src="https://github.com/user-attachments/assets/624b4b2e-4635-4924-b372-df9578416136" />


**MODELS TRAINED:**

<img width="844" height="320" alt="image" src="https://github.com/user-attachments/assets/5e0e243a-e5bb-4faa-8511-13010d09574b" />


# 🎯 Objectives

The goal of this project is to:

* Provide a **clean deep learning pipeline**
* Enable **rapid experimentation with models**
* Demonstrate **best practices for ML project structure**
* Support **deep learning frameworks like TensorFlow and PyTorch**
* Enable **future MLOps integration**


# 🧠 Deep Learning Workflow

```mermaid
flowchart TD

A[Raw Dataset] --> B[Data Cleaning]

B --> C[Feature Engineering]

C --> D[Train Neural Network]

D --> E[Model Validation]

E --> F[Hyperparameter Tuning]

F --> G[Final Model]

G --> H[Prediction System]
```



# 🖼 Model Pipeline Visualization

```mermaid
graph LR

A[Input Data] --> B[Preprocessing]

B --> C[Feature Extraction]

C --> D[Deep Learning Model]

D --> E[Training]

E --> F[Evaluation]

F --> G[Deployment]
```
# 🧠 MLP Architecture Diagram
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/2192791b-1ce5-4f87-ba46-2e0bc60beca6" />

# 🧠 CNN Architecture Diagram
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/923a3da3-0d31-4300-bed6-572a95e90387" />

# 🧠 Pretrained CNN Architecture Diagram
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/c6accabd-5761-4d58-be66-a14faf1012df" />

# 🧠 RNN Architecture Diagram
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/b80f1632-07f0-47f4-b769-e310e3a338c1" />

# 🧠 LSTM Architecture Diagram
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/2fc8ba28-0fd6-46ce-bc87-29df62028fd8" />

# 🧠 GRU Architecture Diagram
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/f8c06481-efea-4cac-95ad-8caff27fdc05" />

# 🧠 GAN Architecture Diagram
<img width="318" height="158" alt="image" src="https://github.com/user-attachments/assets/0010efcb-1c71-4095-83d2-e8b7f206e3b4" />

# 🧠 AE Architecture Diagram
<img width="600" height="329" alt="image" src="https://github.com/user-attachments/assets/fabe154a-9f10-4342-9114-497a628e23ec" />





# 📊 Model Performance

| Model          | Purpose                  | Performance             |
| -------------- | ------------------------ | ----------------------- |
| MLP            | Classification           | 79.66%                  |
| CNN            | Image Classification     | 93.94%                  |
| Pretrained CNN | Transfer Learning        | 98.72%                  |
| RNN            | Sequence Learning        | 85.75%                  |
| LSTM           | Temporal Modeling        | 92.96%                  |
| GRU            | Efficient Sequence Model | 94.38%                  |
| GAN            | Image Generation         | Qualitative (visual)    |
| Autoencoder    | Reconstruction           | Low reconstruction loss |




# 📂 Project Structure

```
Deep-learning_scaffolded-project
│
├── data
│   ├── raw
│   └── processed
│
├── notebooks
│   └── experiments.ipynb
│
├── src
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│
├── models
│   └── trained_models
│
├── requirements.txt
├── README.md
└── main.py
```



# ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yehaa2004/Deep-learning_scaffolded-project.git
```



### Install Dependencies

```bash
pip install -r requirements.txt
```



### Run Training

```bash
python main.py
```



# 🚀 Example Usage

```python
from src.model import build_model
from src.train import train_model

model = build_model()

train_model(model)
```



# 🛠 Technologies Used

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Programming language |
| TensorFlow   | Deep learning        |
| PyTorch      | Neural networks      |
| Scikit-learn | ML utilities         |
| Pandas       | Data processing      |
| NumPy        | Numerical computing  |
| Matplotlib   | Visualization        |



# 🔬 Future Improvements

Future enhancements may include:

* Vision Transformers
* Model explainability (SHAP / LIME)
* Distributed training
* Docker deployment
* REST API serving
* MLOps pipelines



# 📚 References

* TensorFlow Documentation
* PyTorch Deep Learning Guide
* Neural Network Architecture Research Papers



# 👨‍💻 Author

**Yehaasary KM**

**CB.SC.P2AIE25032**

GitHub
[https://github.com/yehaa2004](https://github.com/yehaa2004)


