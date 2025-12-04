# SNN Log Anomaly Detection

Code for the MSc thesis “Spiking Neural Networks for Log Anomaly Detection: Engineering and Integration in Software Monitoring Pipelines” (Blekinge Institute of Technology, 2026).

The repo provides an end‑to‑end log anomaly detection pipeline with:
- Spiking Neural Network (SNN)
- Transformer baseline
- Isolation Forest baseline

It includes data preprocessing, spike encoding, training, evaluation, plots, and basic CI/CD.
---
## Quick start

```bash
git clone https://github.com/raul-parada/SE_thesis_snn.git
cd SE_thesis_snn
sudo docker build -t se_thesis_snn:latest .
sudo docker run -it se_thesis_snn:latest
'''
---
## Main files

- `config.yaml` – pipeline and dataset configuration  
- `dataloader.py` – loads and preprocesses LogHub logs  
- `spikeencoder.py` – spike encoding (rate, temporal, latency)[file:47]  
- `modelsnn.py` – SNN model + trainer  
- `baselineml.py` – Transformer + Isolation Forest baselines  
- `evaluation.py` – detection metrics + engineering metrics (LoC, MI, CC)[file:47]  
- `run_pipeline.py` / `batch_processor.py` – orchestration  
- `generate_plots.py` – plotting utilities  
---
## Citation

If you use this code in research, please cite the thesis:

> Parada Medina, R. Spiking Neural Networks for Log Anomaly Detection: Engineering and Integration in Software Monitoring Pipelines. MSc Thesis, Blekinge Institute of Technology, 2026.
