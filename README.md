# Waste Segregation Using Computer Vision

## Project Overview

This project addresses the problem of improper waste segregation in urban areas leading to environmental pollution and health hazards. It uses advanced computer vision techniques to automatically classify waste into biodegradable, recyclable, and hazardous categories, helping smart cities optimize waste management.

## Research and Resources
- https://www.kaggle.com/asdasdasasdas/garbage-classification
- https://www.researchgate.net/publication/355073419_Using_YOLOv5_for_Garbage_Classification
- https://frontiermcr.com/index.php/home/article/download
- https://www.perplexity.ai/
- https://chatgpt.com/
- https://github.com/mihika-shrivastava/garbage-detection

## Data Preparation
- 10 waste classes including battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash.
- Approx. 100 images per class with resizing to 224x224 pixels.
- Augmentation: flipping, rotation, brightness changes.
- Dataset split: 70% training, 15% validation, 15% testing.
- Organized in class-specific folders.

## Model Architecture
- Classification by MobileNet mapping each waste type to:
  - ‘battery’: hazardous
  - ‘biological’, ‘trash’: biodegradable
  - Others: recyclable

## Evaluation

- Test accuracy: 88.47%
- Detailed metrics include precision, recall, f1-score, and confusion matrix available in `confusion_matrix.jpg`.

## System Logic

- Classified waste is mapped to recommended bins by color:
  - Green for biodegradable
  - Blue for recyclable
  - Red for hazardous
- Confidence score shown alongside classification.

## Streamlit App
- Allows image upload to detect and classify waste.
- Provides prediction with confidence and bin color recommendation.
- [Live app here](https://iadai201-1000197-aarav-agarwal.streamlit.app/)
- Screenshots included within this repo.

## Installation and Usage
1. Clone the repo
2. Install dependencies:
pip install -r requirements.txt
3. Run the app:
streamlit run app.py
4. Upload images and view classified results with bin suggestions.

## Author
**Aarav Agarwal**  
Candidate Registration Number: 1000197  
Course: Artificial Intelligence CRS  
School: Birla Open Minds International School

---

Feel free to reach out for any questions, and thanks for checking out the project!

