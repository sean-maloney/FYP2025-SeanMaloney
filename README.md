#AI/ML Parking Finder 🚗🅿️

#Project Overview
This repository is my Final Year Project **AI/ML Parking Finder**, which aims to detect parking spots & vehicles using aerial footage. the long term goal is to estimate the **free vs. occupied parking spaces** automatically

**Phase 1 focuses on:**
- Training a YOLOv8 detector on the **VisDrone** dataset, removing labels for anything other than vehicles
- Fine tuning and testing it on real parking lot footage
- Building a reproducible training and inference pipeline in Python for easy deployability in real world scenario.
- Exporting the annottated videos that show where vehicles are detected

**Phase 2 plans:**
- Responsive and user friendly react front end.
- ML algorithm that draws a line to help navigate a car to closest parking spot.
- Set up Raspberry Pi system for full system integration.
- Implement a live stream system where inference is always running.
- Draw Json objects within the videos/live stream to mark out parking spots and detect when a car goes into them.

## What Does This Project Do Currently?
1. **Download & Prepare Dataset**
   - Automatically downloads the **VisDrone2019-DET** dataset
   - Converts original annotations into **YOLO format**
   