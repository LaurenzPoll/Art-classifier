## 2.2 Describe Data

**Image Data:**
- **Format:** JPG
- **Number of Images:** 43,453
- **Typical Image Size:** varying sizes (e.g. 335 x 400, 1016 x 1280)
- **File naming convention:** incrementing ID, `XXXXX.jpg`

**Labels/Metadata:**
- **Format:** CSV
- **Number of Records:** 43,455 (one per image + 2 extra or duplicate labels)
- **Fields/Columns:**
  - `ID`: incrementing ID
  - `FILE`: filename to be used for mapping image to row
  - `AUTHOR`: name of author. No specific format to the name
  - `BORD-DIED`: The born_died column is to encode both year and place of birth and death, typically in the format (b. <birth_year>, <birth_place>, d. <death_year>, <death_place>). This format is not consistent across all records
  - `TITLE`: Name given to artwork
  - `DATE`: column to specify in or around what year it is created or thought to be created
  - `TECHNIQUE`: Type of technique used in artwork and in some artwork the size is also encoded. e.g. 'Photo', 'Marble', 'Oil on copper, 56 x 47 cm'
  - `LOCATION`: Column to specify where the artwork was at the moment of recording
  - `FORM`: Form of artwork in broad terms. e.g. 'painting', 'sculpture', 'ceramics'
  - `TYPE`: Type of artwork in broad terms. e.g. 'mythological', 'landscape', 'religious'
  - `SCHOOL`: Presumably school the author studied at
  - `TIMELINE`: Period of artwork, seems to be binned per ~50 years, the `DATE` sometimes falls outside of the assigned bin
  - `URL`: Link to view artwork online


**How Data Was Loaded/Inspected:**
- **Images:** The number, format, and naming of images was inspected using Windows File Explorer
- **Labels/Metadata:** The CSV file was opened and reviewed in Microsoft Excel to check column headers, sample values, and row counts.

No code-based data loading or processing was performed at this step. This approach provided a quick check for obvious mismatches, errors, etc before investing time in more detailed analysis


**Does the data satisfy initial requirements?**
*Partially*
  - The number of images is 2 fewer than stated on the website, indicating either missing or miscounted files.
  - There is a mismatch: 2 extra or duplicate rows in the label file compared to actual image files.
  - The labels contain core metadata fields but do **not** have enough structured information for NLP refinement. Enrichment of the metadata will be required via additional sources.
