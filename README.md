# CrossLingual-NLP-AMLD2020
Material for the hands-on workshop in the ["Applied Machine Learning Days at EPFL 2020"](https://appliedmldays.org/workshops/cross-lingual-natural-language-processing)

Authors:
- [Ioannis Partalas](https://ioannispartalas.github.io/about/)
- [Georgios Balikas](https://balikasg.github.io/)
- Eric Bruno

## Setup
You should have installed the following python 3 packages:
```
numpy
pandas
scikit-learn
torch
umap-learn
seaborn
xgboost
```

In case you use Colab all these packages should be available. If it is not the case you can just use magic:
```bash
!pip install umap-learn torch seaborn
```

The notebooks use the [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch) embeddings. We provide a script to download them and extract them. You can do this by: 
```bash
bash download_conceptNet.sh
```

You will need to install the [LASER](https://github.com/facebookresearch/LASER) library. To do so you can just run the following bash script:
```bash
bash install_laser.sh
```

Finally, download the dataset that we will use during the workshop from the [Semeval 2016 competition on aspect-based sentiment analysis](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools). You will need to download the following datasets:

* English, Dutch, Russian, Spanish and Turkish from the domain restaurants subtask 1.
* Arabic from the Hotels domain subtask 1.
* The test data with the gold annotations for subtask 1.

These datasets use xml format but we will need to convert them in csv for this workshop. You can do that by executing the following script that you will find in the src folder:
```bash
python semeval2csv.py --infile INFILE --outfile OUTFILE [--train]
```
where you just need to specify the input and output files respectively and whether it is a train or test set. Create a directory named ***datasets*** under data and put their the generated csv files. You will have to change accordingly the naming convention for loading the files in the Dataset class.

## Structure of the workshop
The workshop structure is as follows:

1. Brief introduction in text classification: [Intro](https://github.com/ioannispartalas/CrossLingual-NLP-AMLD2020/blob/master/notebooks/AMLD%20Intro.ipynb) 
2. Introduction in cross-lingual word embeddings: [Cross-lingual word embeddings intro](https://github.com/ioannispartalas/CrossLingual-NLP-AMLD2020/blob/master/notebooks/Brief_into_to_Cross_Lingual_embeddings.ipynb) 
3. Cross-lingual document classification:
    1. [Zero-shot learning.](https://github.com/ioannispartalas/CrossLingual-NLP-AMLD2020/blob/master/notebooks/Cross-lingual%20document%20classification.ipynb)
    2. Few-shot learning and fine tuning.

## Exercises
1. Repeat the experiments for various language pairs
2. Add hyper-parameter tuning
3. Repeat the experiments with other embeddings (e.g., [MUSE](https://github.com/facebookresearch/MUSE), [Ferreira et al.](http://www.cs.cmu.edu/~afm/projects/multilingual_embeddings.html), etc..)
4. For the target language use another domain
5. Explore the world of Transformers (BERT etc.). You can take a look at [huffingface](https://github.com/huggingface/transformers)
