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

