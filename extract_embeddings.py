import numpy as np, codecs


f = codecs.open("numberbatch-17.06.txt", 'r', encoding='utf8')
f.readline()
dimension = 300

langs_to_extract = ["en","fr","es","nl","ru","tr","ar"]


files_dico = {}

# Open files for each language
for l in langs_to_extract:
    files_dico[l] = codecs.open("concept_net_1706.300."+l,"w",encoding="utf-8")

for i in f:
    elems = i.split()
    key, val = " ".join(elems[:-dimension]), " ".join(elems[-dimension:])

    elems = key.split("/")
    if elems[2] in langs_to_extract:
        word = elems[3]
        files_dico[elems[2]].write("%s %s\n"%(word,val))
        

for k in files_dico:
    files_dico[k].close()

