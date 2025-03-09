# MMDP-Assignment

# National Anthems Multimodal Analysis Project

This repository contains a comprehensive multimodal analysis of national flags and anthems, exploring correlations between visual symbolism, linguistic features, and musical characteristics across different nations.

## Project Overview

This project studies the relationships between three modalities of national symbols:
1. **Visual Analysis**: Examining flag colors, patterns, and symbols
2. **Textual Analysis**: Analyzing national anthem lyrics for themes and sentiment
3. **Audio Analysis**: Exploring musical elements of national anthem recordings
4. **Multimodal Correlation**: Discovering relationships between these three modalities

The goal is to identify potential cultural correlations and patterns across nations through their national symbols.

## Dataset

The dataset consists of:

- **Flag Images**: SVG files of national flags from at least 100 nations
- **Anthem Texts**: English translations of national anthems
- **Anthem Audio**: MP3 recordings of national anthems

## Project Structure

The repository is organized as follows:

```
MMDP-Assignment/
├── Scripts/
│   ├── __pycache__/                 
│   ├── Task1_a.ipynb           
│   |── Task1_b.ipynb
│   ├── Task1_c.ipynb
│   ├── Task1_d.ipynb
│   |── Task1_e.ipynb
│   ├── Task3_a.ipynb
│   ├── Task3_b.ipynb
│   |── Task3_c.ipynb
│   ├── Task3_d.ipynb
│   ├── Task3_e.ipynb
│   ├── flag_analyzer.ipynb
│   ├── radio_stream_recorder.ipynb
│   └── visualizer_audio.ipynb        
├── Task_1/
│   ├── Task1_a/
│       ├── CulturalVisualCorpus/ 
│       ├── Image classification/ 
│       └── Plots/ 
│   ├── Task1_b/
│       ├── GlobalContextualTextCorpus/ 
│       ├── CrossDomainAnalysis/ 
│       └── Use cases/ 
│            ├── Content Recommendation System/
│            └── Text Classification/ 
│   ├── Task1_c/
│       ├── radio_sonic_archive/ 
│       └── analysis/ 
│   ├── Task1_d/
│       ├── india_climate_data/ 
│       └── Use cases/ 
│            ├── Agricultural crop recommendation system/
│            ├── Scheduled data collection/
│            └── Climate Zone Classification/ 
│   └── Task1_e/
│       ├── apy.csv
│       └── cleaned_data.csv 
├── Task_3/
│   ├── Task3_a/
│       ├── anthem_translation/ 
│       ├── flag_analysis_results/ 
│       ├── nationalanthems_audio/ 
│       └── svg/ 
│   ├── Task3_b/
│       ├── visualizations/
│       └── flag_analysis_report.md
│   ├── Task3_c/
│   ├── Task3_d/
│   └── Task3_e/
└── README.md
```

## Tasks and Implementation

### A. Data Collection

1. **Flag Images**: 
   - Collection of SVG images of national flags for at least 100 nations
   - Storage with proper metadata (country name, source URL)

2. **Anthem Texts**:
   - English translations of national anthems
   - Text preprocessing and cleaning
   - Storage in structured format

3. **Anthem Audio**:
   - MP3 recordings of national anthems
   - Metadata extraction (duration, format details)

### B. Visual Analysis

1. Analysis of flag characteristics:
   - Color distribution and dominant colors
   - Pattern recognition and symbol identification
   - Complexity metrics
   - Visual clustering of flags by similarity

2. Visualization of results:
   - Color distribution charts
   - Flag clustering visualizations

### C. Textual Analysis

1. Analysis of anthem lyrics:
   - Stopword removal and text preprocessing
   - Theme detection (patriotism, freedom, war, peace, nature, religion)
   - Sentiment analysis
   - Word frequency analysis
   - Topic modeling using NMF

2. Visualization of results:
   - Word clouds
   - Theme distribution charts
   - PCA plots for anthem text clustering

### D. Audio Analysis

1. Analysis of anthem recordings:
   - Extraction of audio features (tempo, pitch, energy, volume)
   - Musical pattern identification
   - Audio fingerprinting

2. Visualization of results:
   - Audio feature distribution
   - Clustering of anthems by musical similarity

### E. Multimodal Correlation Analysis

1. Cross-modal correlation:
   - Text-Image correlations: Do flag colors correlate with anthem themes?
   - Text-Audio correlations: Do anthem themes relate to musical characteristics?
   - Image-Audio correlations: Is there a relationship between flag visuals and musical elements?

2. Multimodal clustering:
   - Identifying groups of countries with similar patterns across modalities
   - PCA visualization of multimodal relationships

3. Cross-modal prediction:
   - Predicting features of one modality from another
   - Evaluation of prediction accuracy

## Implementation Details

### Libraries and Dependencies

- **Data Collection**: requests, BeautifulSoup, selenium, ffmpeg
- **Image Processing**: PIL, OpenCV, SVG parsing libraries
- **Text Analysis**: NLTK, TextBlob, scikit-learn
- **Audio Processing**: librosa, pydub
- **Visualization**: matplotlib, seaborn
- **Data Manipulation**: pandas, numpy
- **Machine Learning**: scikit-learn

### Key Implementations

1. **Textual Analysis**:
   - Complete preprocessing pipeline with stopword removal
   - Theme identification using keyword matching
   - Topic modeling with Non-negative Matrix Factorization (NMF)
   - Clustering analysis with K-means

2. **Image Analysis**:
   - Color extraction from SVG files
   - Complexity metrics calculation
   - Visual feature extraction

3. **Audio Analysis**:
   - Feature extraction using librosa
   - Tempo and energy analysis
   - Pitch and volume metrics

4. **Multimodal Correlation**:
   - Combined feature matrix creation
   - Correlation analysis across modalities
   - Visualization of key relationships
   - Cross-modal prediction experiments

## Results and Insights

The analysis reveals several interesting patterns:

1. **Visual-Textual Correlations**: Relationships between flag colors and anthem themes
2. **Musical-Textual Relationships**: How anthem themes relate to musical characteristics
3. **Country Clusters**: Groups of nations with similar multimodal patterns
4. **Cultural Insights**: What these relationships tell us about national identity

## Usage

To run the analysis:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Execute the notebooks in order:
   ```
   jupyter notebook notebooks/
   ```

3. Alternatively, run the complete analysis pipeline:
   ```
   python src/main.py
   ```

4. View results in the `results/` directory

## Future Work

- Expand the dataset to include more countries
- Incorporate additional modalities (e.g., national symbols, currencies)
- Develop interactive visualizations for exploring correlations
- Apply more advanced machine learning techniques for pattern discovery

## Contributors

- Rishita Agarwal, 220150016, Data Science and Artificial Intelligence Department

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sources: National anthem repositories, flag databases
- Academic references on multimodal analysis and cultural symbolism
