import numpy as np, codecs


f = codecs.open("numberbatch-19.08.txt", 'r', encoding='utf8')
f.readline()
dimension = 300

vectors = {}
en, es = {}, {}
cnt_en, cnt_es = 0, 0

for i in f:
    elems = i.split()
    key, val = " ".join(elems[:-dimension]), " ".join(elems[-dimension:])

    elems = key.split("/")
    if elems[2]=='en':
        en[elems[3]]=val
        cnt_en += 1
        if cnt_en % 10000 == 0:
            with codecs.open("concept_net_1706.300.en", 'a', encoding='utf8') as out:
                for key, val in en.items():
                    out.write("%s %s\n"%(key, val))
            cnt_en, en = 0, {}

    elif elems[2]=='es':
        es[elems[3]]=val
        cnt_es += 1
        if cnt_es % 10000 == 0:
            with codecs.open("concept_net_1706.300.es", 'a', encoding='utf8') as out:
                for key, val in es.items():
                    out.write("%s %s\n"%(key, val))
            cnt_es, es = 0, {}
    else:
        pass


with codecs.open("concept_net_1908.300.en", 'a', encoding='utf8') as out:
    for key, val in en.items():
        out.write("%s %s\n"%(key, val))

with codecs.open("concept_net_1908.300.es", 'a', encoding='utf8') as out:
    for key, val in es.items():
        out.write("%s %s\n"%(key, val))

