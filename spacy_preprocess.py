#!/usr/bin/env python3 
# Convert conll annotations to Examples
# Coref uses a special CoNLL format, so the usual spaCy converters don't work.

import sys
import re
from collections import defaultdict

import spacy
from spacy.tokens import Doc, DocBin
from icecream import ic

DOCID_REGEX = "#begin document \((.*)\); part (\d*)"

def convert_conll_doc_to_example(cdoc):
    pass

def read_file(fname, outname):
    with open(fname) as infile:
        text = infile.read()

    nlp = spacy.blank("en")
    db = DocBin()

    
    docs = text.split("\n#end document\n")
    for doc in docs:
        name = None
        sents = doc.split("\n\n")
        clustermap = defaultdict(list)

        words = []
        sent_starts = []

        for sent in sents:
            lines = sent.split("\n")
            if name is None:
                top = lines.pop(0)
                # doc id line looks like this:
                #begin document (/some/path); part 000
                matches = re.match(DOCID_REGEX, top)
                if not matches:
                    # happens on the last line
                    break

                name = f"{matches.group(1)}_{matches.group(2)}"

            for line in lines:
                if not line: continue # ignore blanks
                # note: file is not tsv, it uses ~aligned spaces~
                fields = line.split()
                surface = fields[3]
                # weird escapes
                if surface in ("/.", "/?"):
                    surface = surface[1:]

                sent_starts.append(int(fields[2]) == 0)
                words.append(surface)

                clusters = fields[-1]
                if clusters == "-":
                    clusters = []
                else:

                    # this has to be sorted because the same cluster can be
                    # annotated twice on the same token, and a cluster can
                    # contain itself.
                    # Example: "He himself..."
                    #   He       (30
                    #   himself  (30)|30)
                    clusters = sorted(clusters.split("|"), reverse=True)

                    tokid = len(words) - 1
                    for mention in clusters:
                        if mention[0] == "(" and mention[-1] == ")":
                            cid = int(mention[1:-1])
                            clustermap[cid].insert(0, (tokid, tokid + 1 ) )
                        elif mention[0] == "(":
                            cid = int(mention[1:])
                            clustermap[cid].append( tokid ) # this will be popped
                        elif mention[-1] == ")":
                            cid = int(mention[:-1])
                            start = clustermap[cid].pop()
                            clustermap[cid].insert(0, (start, tokid + 1 ) )
        doc = Doc(nlp.vocab, words=words, sent_starts=sent_starts)
        for key, vals in clustermap.items():
            spans = [doc[ss:ee] for ss, ee in vals]
            skey = f"coref_clusters_{len(doc.spans) + 1}"
            doc.spans[skey] = spans
            #print(key)
            for val in vals:
                pass
                #print("    ", doc[val[0]: val[1]])
        db.add(doc)
        if len(db) >= 100: break

    print(f"Serializing {len(db)} documents")
    db.to_disk(outname)

if __name__ == "__main__":
    read_file(sys.argv[1], sys.argv[2])
