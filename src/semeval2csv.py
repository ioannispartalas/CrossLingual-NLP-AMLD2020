import xml.etree.ElementTree as ET
import pandas as pd
import argparse
import sys


def semeval2csv(in_file,out_file,train=True):
    """
    Utility function to convert the xml version of the SemEval Aspect-Based Sentiment Analysis datasets to csv format.
    
    Arguments:
    ----------
    in_file: full path to the xml file
    out_file: file to write the data in csv format
    train: boolean, whether it is the train file or not. In case of False we assume that the test set with the annotations is given.
    """
    tree = ET.parse(in_file)
    root = tree.getroot()
    
    sentences = []
    polarities = []
    for r in root.findall("Review"):
        for s in r.findall("sentences"):
            for sentence in s.findall("sentence"):
                if train==False and sentence.get("OutOfScope")=="TRUE":
                    continue
                sentences.append(sentence.find('text').text)
                p = []
                for o in sentence.findall("Opinions"):
                    for op in o.findall("Opinion"):
                        p.append(op.get("polarity"))
                polarities.append(p)
    
    pd.DataFrame({"text":sentences,"polarities":polarities}).to_csv(out_file)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Convert semeval xml to csv.')
    parser.add_argument("--infile",action="store")
    parser.add_argument("--outfile",action="store")
    parser.add_argument('--train', default=False, action='store_true')
    args = parser.parse_args()

    semeval2csv(args.infile,args.outfile,args.train)
    
