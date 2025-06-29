# 1. Determine Business Objectives

## 1.1. Background

- **Role & scope:** Solo student project (Fontys, Eindhoven), 4-week duration.
- **Domain:** art curation in which a human curator traditionally selects and arranges artworks by balancing visual style, period context, narrative flow, and practical constraints.
- **Motivation:** Explore whether an AI system can replicate expert decision-making, potentially broadening access to curated exhibits.

## 1.2. Business Objectives

**Primary Objective:** Build a prototype AI assistant that given a collection of artworks and metadata, can group and recommend pieces for a coherent exhibition.
**Related bussines questions:**
    1. How well can the system discern visual style (e.g. Impressionism vs. Expressionism)?
    2. Can it surface thematic or period-based groupings (e.g. 19th-century landscapes)?
    3. How might a curator use this tool in their existing workflow?

- **Phase 1 (Fit check):**  
  Given a targeted collection and a new artwork, decide whether the artwork belongs in that collection.

- **Phase 2 (Placement Recommendation):**  
  Given a full existing collection and a new artwork, assign it to the appropriate subgroup, suggest its position within the narrative flow, or flag it as an outlier.
  - *Precondition:* Phase 1 meets its success metric.


## 1.3. Business Success Criteria
Objective measures
**Phase 1:**
- *Success metric:* 70% accuracy on test set.
    - iteration 0 achieved 80% accuracy on the test set where the art is a paiting and one of 4 predefined styles

**Phase 2:**
- *Success metric:* ≥ 70% accuracy in correct subgroup assignment, plus teachers feedback.


Subjective/Stakeholder judgments:
- Prototype must be deemed "useful and intuitive" by teachers