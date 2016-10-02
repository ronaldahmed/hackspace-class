from nltk.compat import string_types, python_2_unicode_compatible, unicode_repr
from nltk.corpus.reader import SyntaxCorpusReader
from nltk.corpus.reader import xmldocs
from nltk import tree
from nltk.util import LazyMap
from nltk.corpus.reader.util import concat
import ipdb
from readers.utils import simplify_tagset

def pos_lemma(element):
    if element:
        # element viewed as a list is non-empty (it has subelements)
        subtrees = map(pos_lemma, element)
        subtrees = [t for t in subtrees if t is not None]
        return tree.Tree(element.tag, subtrees)
    else:
        # element viewed as a list is empty. we are in a terminal.
        if element.get('elliptic') == 'yes':
            return None
        else:
            lemma = element.get('lem') or 'unk'
            pos = element.get('pos') or 'unk'
            return tree.Tree(pos + ' ' + lemma, [element.get('wd')])


def pos_ne_parsed(element):
    if element:
        # element viewed as a list is non-empty (it has subelements)
        subtrees = map(pos_ne_parsed, element)
        subtrees = [t for t in subtrees if t is not None]
        return tree.Tree(element.tag, subtrees)
    else:
        # element viewed as a list is empty. we are in a terminal.
        if element.get('elliptic') == 'yes':
            return None
        else:
            ne = element.get('ne') or 'O'
            pos = element.get('pos') or 'unk'
            return tree.Tree(pos + ' ' + ne, [element.get('wd')])


def ne_parsed(element):
    if element:
        # element viewed as a list is non-empty (it has subelements)
        subtrees = map(ne_parsed, element)
        subtrees = [t for t in subtrees if t is not None]
        return tree.Tree(element.tag, subtrees)
    else:
        # element viewed as a list is empty. we are in a terminal.
        if element.get('elliptic') == 'yes':
            return None
        else:
            return tree.Tree(element.get('ne') or 'O', [element.get('wd')])


def parsed(element):
    if element:
        # element viewed as a list is non-empty (it has subelements)
        subtrees = map(parsed, element)
        subtrees = [t for t in subtrees if t is not None]
        return tree.Tree(element.tag, subtrees)
    else:
        # element viewed as a list is empty. we are in a terminal.
        if element.get('elliptic') == 'yes':
            return None
        else:
            return tree.Tree(simplify_tagset(element.get('pos')) or element.get('ne') or 'unk', [element.get('wd')])
            #return tree.Tree(element.get('pos') or 'unk', [element.get('wd')])


def tagged(element):
    # http://www.w3schools.com/xpath/xpath_syntax.asp
    # XXX: XPath '//*[@wd]' not working
    #return [(x.get('wd'), x.get('pos') or x.get('ne')) for x in element.findall('*//*[@wd]')] + [('.', 'fp')]
    return filter(lambda x: x != (None, 'unk'), parsed(element).pos())


def untagged(element):
    # http://www.w3schools.com/xpath/xpath_syntax.asp
    # XXX: XPath '//*[@wd]' not working
    #return [x.get('wd') for x in element.findall('*//*[@wd]')] + [('.', 'fp')]
    return filter(lambda x: x is not None, parsed(element).leaves())


def ne_tagged(element):
    return filter(lambda x: x != (None, None), ne_parsed(element).pos())


def pos_lemma_tagged(element):
    return filter(lambda x: x != ('unk', 'unk'), pos_lemma(element).pos())


def pos_ne_tagged(element):
    temp = pos_ne_parsed(element).pos()
    temp = [tuple([k[0]] + k[1].split(' '))  for k in temp]
    return filter(lambda x: x != (None,'unk','O'), temp)

def reformat_IOB(sent):
    res = []
    for w,label in sent:
        if not w:
            continue
        ws = w.split('_')
        if label!='O':
            new_label = 'B-'+label
            res.append( (ws[0],new_label) )
            if len(ws)>1:
                i_lab = 'I-'+label
                res.extend([(ww,i_lab) for ww in ws[1:]])
        else:
            res.append( (w,label) )
    return res

def reformat_posne_IOB(sent):
    res = []
    for w,pos,label in sent:
        if not w:
            continue
        ws = w.split('_')
        if label!='O':
            new_label = 'B-'+label
            res.append( (tuple([ws[0],pos]),new_label) )
            if len(ws)>1:
                i_lab = 'I-'+label
                res.extend([(tuple([ww,pos]),i_lab) for ww in ws[1:]])
        else:
            res.append( (tuple([w,pos]),label) )
    return res


class AncoraCorpusReader(SyntaxCorpusReader):

    #def __init__(self, xmlreader):
    #    self.xmlreader = xmlreader
    def __init__(self, path='datasets/ancora-2.0/'):
        #self.xmlreader = xmldocs.XMLCorpusReader(path + 'TODO', '.*\.xml')
        self.xmlreader = xmldocs.XMLCorpusReader(path + '3LB-CAST', '.*\.xml')
    
    def joinSentences(self,fun, max_docs='inf'):
        fileids = self.xmlreader.fileids()
        if max_docs != 'inf':
            fileids = fileids[:max_docs]
        docs = [list(self.xmlreader.xml(fileid)) for fileid in fileids]
        #pdb.set_trace()
        out = []
        for doc in docs:
            temp = LazyMap(fun, doc)
            out.append([list(el) for el in temp])
        return out

    def parsed_sents(self, fileids=None):
        if not fileids:
            fileids = self.xmlreader.fileids()
        return LazyMap(parsed, concat([list(self.xmlreader.xml(fileid)) for fileid in fileids]))

    def tagged_sents(self,max_docs='inf'):
        res = self.joinSentences(tagged, max_docs=max_docs)
        #return [sent for sent in res if len(sent) > 0]
        return res

    def pos_lemma_sents(self,max_docs='inf'):
        res = self.joinSentences(pos_lemma_tagged, max_docs=max_docs)
        return res

    def ne_tagged_sents(self, fileids=None):
        if not fileids:
            fileids = self.xmlreader.fileids()
        res = LazyMap(ne_tagged, concat([list(self.xmlreader.xml(fileid)) for fileid in fileids]))
        res = [reformat_IOB(list(el))  for el in res]
        return [sent for sent in res if len(sent)>0]

    def sents(self, fileids=None):
        # FIXME: not lazy!
        if not fileids:
            fileids = self.xmlreader.fileids()
        res = LazyMap(untagged, concat([list(self.xmlreader.xml(fileid)) for fileid in fileids]))
        res = [list(el) for el in res]
        return [sent for sent in res if len(sent) > 0]

    def tagged_words(self, fileids=None):
        return concat(self.tagged_sents(fileids))
        
    def pos_ne_tagged_sents(self,fileids=None):
        if not fileids:
            fileids = self.xmlreader.fileids()
        res = LazyMap(pos_ne_tagged, concat([list(self.xmlreader.xml(fileid)) for fileid in fileids]))
        res = [reformat_posne_IOB(list(el))  for el in res]
        return [sent for sent in res if len(sent) > 0]
