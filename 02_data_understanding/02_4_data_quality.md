The initial exploration of the ART500K Toy Dataset revealed several data quality aspects relevant to the projects objectives:

**Missing Data:**
- The labels file contained two more records than the actual number of image files. After cross-referencing, records without corresponding image files were identified and dropped.
- The 'FILE' field was also found to have a missing values, this value was excluded.

**Image File Consistency:**
- Final dataset contains only those records with existing image files.

**Categorical Metadata Normalization:**
- Fields such as ‘LOCATION’, ‘FORM’, ‘TYPE’, ‘SCHOOL’, and ‘TIMELINE’ were stripped of whitespace to standardize
- The ‘LOCATION’ field has variations in formatting (capitalization, whitespace, etc.) when the values are actualy equal. These were standardized to improve grouping consistency

**Field Pruning:**
- Columns with high inconsistency or limited immediate relevance (‘TECHNIQUE’ and ‘BORN-DIED’) were dropped to simplify the initial analysis

**Class Focus:**
- To maintain focus and manage data volume, only records with ‘FORM’ set to ‘painting’ should be kept for the first iteration.
- For ‘TYPE’, only the values ‘religious’, ‘portrait’, ‘landscape’, and ‘mythological’ should be used, ensuring clear and well represented categories for initial modeling.

**Image Dimensions:**
- Images vary considerably in both width and height. This diversity will require careful preprocessing (resizing or aspect ratio handling) before input to any CNN

**Unique Value Counts:**
- The main categorical columns showed a good amount of unique values after normalization, supporting their use in supervised learning tasks, for now it seems that unsupervised learning could be possible with this dataset