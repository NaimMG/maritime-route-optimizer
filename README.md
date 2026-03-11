# 🚢 Maritime Route Optimizer

> End-to-end maritime route optimization using real AIS vessel tracking data, Graph Neural Networks, and A* pathfinding.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/Chasston)

## 🎯 Problem

Given a departure port and a destination port, find the optimal maritime route that minimizes travel time and fuel consumption while avoiding adverse weather conditions and dangerous currents.

## 🏗️ Architecture
```
AIS Data (real vessel positions)
        ↓
  Data Pipeline (cleaning, filtering)
        ↓
  Feature Engineering (weather, currents, distance)
        ↓
  Graph Neural Network (dynamic edge cost prediction)
        ↓
  A* Pathfinding (optimal route search)
        ↓
  FastAPI + Gradio Demo
```

## 📦 Stack

| Layer | Technology |
|-------|-----------|
| Data processing | Pandas, GeoPandas |
| ML Model | PyTorch, PyTorch Geometric (GNN) |
| Optimization | Custom A* with dynamic costs |
| API | FastAPI |
| Demo | Gradio → HuggingFace Spaces |

## 📁 Project Structure
```
maritime-route-optimizer/
├── data/
│   ├── raw/          # AIS raw data (not versioned)
│   ├── processed/    # Cleaned datasets
│   └── external/     # Weather, ports reference data
├── notebooks/        # Exploratory analysis
├── src/
│   ├── data/         # Ingestion & cleaning pipeline
│   ├── features/     # Feature engineering
│   ├── models/       # GNN + A* optimizer
│   └── api/          # FastAPI endpoints
├── app/              # Gradio interface
├── tests/            # Unit tests
└── configs/          # YAML configuration files
```

## 🚀 Getting Started
```bash
# Clone the repo
git clone https://github.com/NaimMG/maritime-route-optimizer.git
cd maritime-route-optimizer

# Create virtual environment
python3 -m venv maritimeoptimizer
source maritimeoptimizer/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Data

This project uses real **AIS (Automatic Identification System)** data — the GPS-like tracking system mandatory on all vessels over 300 tons. Data sourced from [AISHub](https://www.aishub.net/) / [Marine Cadastre](https://marinecadastre.gov/).

## 🔬 Methodology

1. **EDA** — Explore vessel trajectories, port density, seasonal patterns
2. **Graph construction** — Ports as nodes, historical routes as edges
3. **GNN training** — Learn dynamic edge costs (weather impact, traffic, fuel)
4. **Route optimization** — A* search using GNN-predicted costs
5. **Evaluation** — Compare against great-circle baseline and real routes

## 👤 Author

**Naim** — [GitHub](https://github.com/NaimMG) · [HuggingFace](https://huggingface.co/Chasston)

---
*Project built step by step as a real-world data science portfolio piece.*
