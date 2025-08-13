# CADDack

**Inverse Computational Drug Design Toolkit**  
CADDack is an open-source research toolkit for **inverse molecular design** — predicting the chemical features and molecular structures necessary to interact with a target enzyme or achieve a desired bioactivity profile.  
Unlike traditional QSAR models that predict activity from a given ligand, CADDack aims to solve the **inverse problem**: _“What kind of chemical makeup would a ligand need to bind this target?”_

---

## ​ Features

- **Inverse QSAR Modeling** – From target properties to molecular features.
- **Generative Model Integration** – Supports graph-based molecular generation (GNNs, VAEs, diffusion models).
- **Ligand & Protein Data Retrieval** – Automated access to ChEMBL, UniProt, and other databases.
- **Docking & Scoring Pipelines** – Optional AutoDock/HADDOCK integration for post-generation validation.
- **Descriptor Analysis** – Identify functional groups and descriptors that drive bioactivity.
- **Synthetic Accessibility Filtering** – Ensure generated molecules are chemically feasible.

---

## ​ Scientific Foundations

CADDack is inspired by cutting-edge research in **inverse molecular design** and **conditional generative models**.  
Key references include:

1. Gómez-Bombarelli et al. – *Inverse molecular design using machine learning*  
   (https://www.science.org/doi/10.1126/science.aat2663)  
2. Walters et al. – *Deep generative molecular design reshapes drug discovery*  
   (https://pmc.ncbi.nlm.nih.gov/articles/PMC9797947/)  
3. Sánchez-Lengeling & Aspuru-Guzik – *Generative models for molecular discovery*  
   (https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1608)  
4. Kang & Cho – *Conditional molecular design with deep generative models*  
   (https://arxiv.org/abs/1805.00108)  
5. Gao et al. – *Graph Diffusion Transformers for Multi-Conditional Molecular Generation*  
   (https://arxiv.org/abs/2401.13858)  
6. Polishchuk et al. – *Inverse QSAR with ACoVAE*  
   (https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/630886410187d9100aa1ca44/original/inverse-qsar-reversing-descriptor-driven-prediction-pipeline-using-attention-based-conditional-variational-autoencoder-a-co-vae.pdf)

---

## ​ Installation

```bash
git clone https://github.com/yourusername/CADDack.git
cd CADDack
conda env create -f environment.yml
conda activate caddack
