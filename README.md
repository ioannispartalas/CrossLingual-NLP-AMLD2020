# CrossLingual-NLP-AMLD2020
Material for the hands-on workshop in the "Applied Machine Learning Days at EPFL 2020"

## Setup
You should have installed the following python packages:
```
numpy
pandas
scikit-learn
umap-learn
seaborn
```

The notebooks use the [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch) embeddings. We provide a script to download them and extract them. You can do this by: 
```bash
bash download_conceptNet.sh
```

You will need to install the [LASER](https://github.com/facebookresearch/LASER) library. To do so you can just run the following bash script:
```bash
bash install_laser.sh
```

Finally, download the dataset that we will use during the workshop:

## Structure of the workshop
The workshop structure is as follows:

1. Brief introduction in text classification: [Intro](https://github.com/ioannispartalas/CrossLingual-NLP-AMLD2020/blob/master/Cross-lingual%20document%20classification.ipynb) 
2. Introduction in cross-lingual word embeddings: [Cross-lingual word embeddings intro](https://github.com/ioannispartalas/CrossLingual-NLP-AMLD2020/blob/master/notebooks/Brief_into_to_Cross_Lingual_embeddings.ipynb) 
3. Cross-lingual document classification:
    1. Zero-shot learning using word embeddings.
    2. Multi-lingual classification using sentence embeddings.
