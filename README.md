## Alternate Geometric and Semantic Denoising Diffusion for Protein Inverse Folding (ECML-PKDD 2025)

## 🎯Introduction

In this work, we propose an Alternate Geometric and Semantic Denoising Diffusion (AGSDD) that performs two types of denoising, i.e., geometric denoising and semantic denoising in turn, in the joint Geo-semantic residue representation: (1) the geometric denoising module uses a geometric contextual aggregator to encode global contextual information from the entire protein structure and selectively distributes information to each residue; and (2) the semantic denoising module uses a learnable key-value dictionary of residue-types to facilitate communication between them so that learned residue features can be more accurately aligned to proper residue types.


## Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/llllly26/AGSDD
    cd AGSDD
    ```
2.  **Install Dependencies:**
    ```bash
    conda env create -f envs.yml
    ```
3.  **Download Data and checkpoint:** The model checkpoint could be downloaded from [[**HERE**](https://drive.google.com/file/d/1YO4IKAHfkBcTjjFGwDIjHmwOYcBVt0gG/view?usp=drive_link)]. The processed data could be downloaded from [**[HERE](https://drive.google.com/drive/folders/1RS1gD9EaEUF9Pp7xaDqp-AGHyMOKnBVG?usp=drive_link)**]

## Train
```bash
cd AGSDD 
python  main.py --lr 5e-4 --wd 1e-5 --drop_out 0.1 --depth 6 --hidden_dim 128 --embedding --embedding_dim 128 --norm_feat --noise_type uniform
```

## Test
```bash
python test.py
```

## ☀️Acknowledgements
Our implementation is inspired by the following open-source projects: [PiFold](https://github.com/A4Bio/PiFold) and [GraDe_IF](https://github.com/ykiiiiii/GraDe_IF).  Thanks for their valuable contribution!

## ✨Citation
If you find that this work is useful for your research, please kindly give a star ⭐ and consider citation:
```
