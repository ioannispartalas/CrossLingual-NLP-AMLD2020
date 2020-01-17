import numpy as np, codecs


f = codecs.open("numberbatch-19.08.txt", 'r', encoding='utf8')
f.readline()
dimension = 300

langs_to_extract = ["en","fr","es","nl","ru"]

vectors = {}
en, es = {}, {}
cnt_en, cnt_es = 0, 0

files_dico = {}

# Open files for each language
for l in langs_to_extract:
    files_dico[l] = codecs.open("concept_net_1908.300."+l,"w",encoding="utf-8")

for i in f:
    elems = i.split()
    key, val = " ".join(elems[:-dimension]), " ".join(elems[-dimension:])

    elems = key.split("/")
    if elems[2] in langs_to_extract:
        word = elems[3]
        files_dico[elems[2]].write("%s %s\n"%(word,val))
        

for k in files_dico:
    files_dico[k].close()

#for i in f:
#    elems = i.split()
#    key, val = " ".join(elems[:-dimension]), " ".join(elems[-dimension:])
#
#    elems = key.split("/")
#    if elems[2]=='en':
#        en[elems[3]]=val
#        cnt_en += 1
#        if cnt_en % 10000 == 0:
#            with codecs.open("concept_net_1706.300.en", 'a', encoding='utf8') as out:
#                for key, val in en.items():
#                    out.write("%s %s\n"%(key, val))
#            cnt_en, en = 0, {}
#
#    elif elems[2]=='es':
#        es[elems[3]]=val
#        cnt_es += 1
#        if cnt_es % 10000 == 0:
#            with codecs.open("concept_net_1706.300.es", 'a', encoding='utf8') as out:
#                for key, val in es.items():
#                    out.write("%s %s\n"%(key, val))
#            cnt_es, es = 0, {}
#    else:
#        pass
#
#
#with codecs.open("concept_net_1908.300.en", 'a', encoding='utf8') as out:
#    for key, val in en.items():
#        out.write("%s %s\n"%(key, val))
#
#with codecs.open("concept_net_1908.300.es", 'a', encoding='utf8') as out:
#    for key, val in es.items():
#        out.write("%s %s\n"%(key, val))
#
