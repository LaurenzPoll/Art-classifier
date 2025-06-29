# 2. Assess Situation

## 2.1. Inventory of Resources

### Personnel
- **Myself:** Advanced AI student (solo project)
- **Teachers and peers:** Technical guidance and feedback

### Data
- **ART500K Toy dataset:** 43,555 labeled art with image and metadata (artist, movement, genre, etc.). [deepart.hkust.edu.hk](https://deepart.hkust.edu.hk/ART500K/art500k.html) 7gb in size
- **Labels:** CSV file with headers: "ID	FILE	AUTHOR	BORN-DIED	TITLE	DATE	TECHNIQUE	LOCATION	FORM	TYPE	SCHOOL	TIMELINE	URL" (tab delimiter)

### Computing
- **Hardware:** Personal laptop (GPU - NVIDEA GeForce RTX 3050 Ti)
- **Storage:** Sufficient for toy dataset; No capacity for the full 500K image dataset

### Software
- **Languages & tools:**
  - Python (with Jupyter/VS Code)
  - GitHub
- **Libraries:**
  - pandas, numpy, scikit-learn
  - PyTorch or TensorFlow
  - OpenCV, PIL
  - spaCy, NLTK, or HuggingFace Transformers


## 2.2. Requirements, Assumptions, and Constraints

### Requirements
- **Schedule:** 4 weeks (8 full working days)
- **Results:** Prototype capable of grouping/curating paintings
- **Data use:** Must comply with ART500K licensing (free for academic use)

### Assumptions
- Image labels (styles, artist, period) are sufficiently accurate for supervised learning
- Metadata is available for all or most artworks in the dataset

### Constraints
- **Time:** Short window; prototype scope must remain focused.
- **Compute:** Primarily CPU, with some access to GPU if configured.
- **Data Quality:** Possible label inconsistency or missing metadata.


## 2.3. Risks & Contingencies

| Risk                                    | Risk Factor | Contingency                                                                 |
|-----------------------------------------|-------------|-----------------------------------------------------------------------------|
| Inconsistent or incomplete labels       | Medium      | Analyze and clean data; possibly restrict to well-annotated subset          |
| Limited compute for CNN training        | Medium      | Use pre-trained models; limit image resolution; restrict model complexity   |
| Project scope creep                     | Medium      | Set clear milestones                                                        |
| Local GPU configuration fails           | Medium      | Default to CPU; reduce model/training size; extend timeline if possible     |
| Dataset licensing or ethical use issues | Low         | Strictly use ART500K toy dataset for academic work                          |


## 2.4. Terminology

| Term                | Definition                                                                              |
|---------------------|-----------------------------------------------------------------------------------------|
| **Curation**        | The act of selecting and organizing artworks for display in an exhibition.              |
| **Art style**       | A set of visual characteristics or techniques that typify an artwork (e.g., Impressionism). |
| **Metadata**        | Information describing the artwork (artist, year, style, etc.).                         |
| **Embedding**       | A vector representation of an image or text for comparison or clustering.              |
| **NLP**             | Natural Language Processing—techniques for analyzing and understanding textual data.    |
| **CNN**             | Convolutional Neural Network—used here for image feature extraction/classification.     |
| **Clustering**      | Grouping items based on feature similarity, unsupervised or semi-supervised.            |


## 2.5. Cost–Benefit Sketch

| Cost (Investment)                                   | Benefit (Academic / Practical Gain)                       |
|-----------------------------------------------------|-----------------------------------------------------------|
| ~8 full working days over 4 weeks (solo project)    | Hands-on experience with CNNs, NLP, and art dataset curation; insight into AI’s potential for creative fields |

> **Note:** As a student project, main costs are time and compute.