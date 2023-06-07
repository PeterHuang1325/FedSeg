# Robust Federated MRI Segmentation
In this repo, we provide an FL framework which incorporates following features:

- Simple γ-mean aggregator
- γ-logistic dice loss
- Data-driven γ-selection methods for simple γ-mean and γ-logistic
- Locally adaptive FL paradigm: Ditto(2021) [Ditto paper](https://arxiv.org/abs/2012.04221)+FedBN(2021) [FedBN paper](https://arxiv.org/abs/2102.07623)

to be robust to **Byzantine perturbation**, **mislabeled data** in clients and is locally adaptive with **data heterogeneity** across clients.

![FL](/images/rob_plot.png)
- *Note: part of the code structure is inspired from* [FedDG](https://github.com/liuquande/FedDG-ELCFS)

## Usage
1. **Prepare data:**
- Create the folder as following example:
     ``` 
     ├── dataset
        ├── client1
           ├── data_npy
               ├── sample1.npy, sample2.npy, xxxx
        ├── clientxxx
        ├── clientxxx
   ``` 
     
  
 - Run ``dataset/prepare_lgg.ipynb`` and `dataset/prepare_prostate.ipynb` to create clients' data.
2. **Run experiments:**
- LGG experiment:
  - Without perturbation:
     - Run `python run_glioma_robust_cln.py  --mode='fedbn' --aggr='gam_mean' --loss_fn='auto_gamma_dice_loss'`
  - With random perturbation:
     - Run `python run_glioma_robust_byz.py  --mode='fedbn' --aggr='gam_mean' --loss_fn='auto_gamma_dice_loss'`
     
- Prostate experiment:
  - Without perturbation:
     - Run `python run_prostate_robust_cln.py  --mode='fedbn' --aggr='gam_mean' --loss_fn='auto_gamma_dice_loss'`
  - With random perturbation:
     - Run `python run_prostate_robust_byz.py  --mode='fedbn' --aggr='gam_mean' --loss_fn='auto_gamma_dice_loss'`
     
### MRI Data download
1. LGG (Low-Grade Glioma) [LGG link](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
2. Prostate Cancer [Prostate link](https://liuquande.github.io/SAML/)








