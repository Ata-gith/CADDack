# CADDack – Inverse Computational Drug Design

CADDack is a research-driven software framework for **inverse molecular design**.  
Instead of predicting whether a ligand interacts with a protein, CADDack asks the **inverse question**:  
> "Given a desired binding property, what kind of molecular structure should the ligand have?"

This approach blends **computational drug design (CADD)** techniques with **machine learning**, particularly **Graph Neural Networks (GNNs)** and **generative models**, to propose candidate molecules that meet specified target conditions.

---

## Features

- **Ligand & Protein Data Integration** – Fetch and process structures from public databases (e.g., ChEMBL, UniProt, PDB).  
- **QSAR & Descriptor Analysis** – Integrate traditional cheminformatics pipelines using RDKit.  
- **Inverse Molecular Design** – Use conditional generative models to design novel ligands with desired chemical/biological properties.  
- **Graph Neural Network Module** – Encode molecular graphs for predictive and generative tasks.  
- **Docking Analysis Support** – Interface with docking tools (AutoDock, HADDOCK) for structural binding validation.

---

## Roadmap

**Short-term goals (0–3 months):**
1. Build data ingestion pipeline from ChEMBL and UniProt.
2. Implement basic QSAR workflows in RDKit.
3. Train baseline GNN for ligand property prediction.

**Mid-term goals (3–6 months):**
1. Integrate inverse molecular design models (e.g., Conditional VAEs, Graph Diffusion Models).
2. Add molecular docking post-processing.
3. Prepare end-to-end ligand design + docking pipeline.

**Long-term goals (>6 months):**
1. Implement active learning loop to iteratively improve predictions.
2. Optimize models for multi-conditional property generation.
3. Publish dataset and code for reproducible research.

---

## References

Gómez-Bombarelli, R., Wei, J. N., Duvenaud, D., Hernández-Lobato, J. M., Sánchez-Lengeling, B., Sheberla, D., Aguilera-Iparraguirre, J., Hirzel, T. D., Adams, R. P., & Aspuru-Guzik, A. (2018). Automatic chemical design using a data-driven continuous representation of molecules. *Science, 361*(6400), 360–365. https://doi.org/10.1126/science.aat2663

Walters, W. P., Murcko, M., & Koes, D. R. (2020). Deep generative molecular design reshapes drug discovery. *Drug Discovery Today, 25*(10), 1702–1711. https://doi.org/10.1016/j.drudis.2020.07.009 (Free full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC9797947/)

Sánchez-Lengeling, B., & Aspuru-Guzik, A. (2018). Inverse molecular design using machine learning: Generative models for matter engineering. *Wiley Interdisciplinary Reviews: Computational Molecular Science, 8*(5), e1608. https://doi.org/10.1002/wcms.1608

Kang, S., & Cho, K. (2018). Conditional molecular design with deep generative models. *arXiv preprint*. https://arxiv.org/abs/1805.00108

Gao, W., Nguyen, T., & Coley, C. W. (2024). Graph diffusion transformers for multi-conditional molecular generation. *arXiv preprint*. https://arxiv.org/abs/2401.13858

Polishchuk, P., Madzhidov, T., Nugmanov, R., & Bodrov, A. (2022). Inverse QSAR: Reversing descriptor-driven prediction pipeline using attention-based conditional variational autoencoder (ACoVAE). *ChemRxiv*. https://doi.org/10.26434/chemrxiv-2022-09z88

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.
