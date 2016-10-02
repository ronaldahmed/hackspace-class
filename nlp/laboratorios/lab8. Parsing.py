import nltk
from nltk.probability import *
from sklearn.cross_validation import train_test_split
from readers import ancora
import re
import ipdb

##########################################################################

no_dup_spaces = re.compile('\s+')
no_bl = re.compile('\n')

##########################################################################

if __name__== "__main__":
    print("Leyendo data...")
    reader = ancora.AncoraCorpusReader()
    treebank = reader.parsed_sents()
    treebank = [t for t in treebank if len(t.leaves())<15]
    #treebank = treebank[:100]
    
    # normalizando
    print("Normalizando treebank -> CNF...")
    for t in treebank:
        t.collapse_unary(collapsePOS=True)
        t.chomsky_normal_form()
    
    # dividiendo data
    test_perc = 0.2
    train_set,test_set = train_test_split(treebank,test_size=test_perc,random_state=42)

    # Obtener reglas
    print("Leyendo reglas...")
    rules = []
    for t in train_set:
        rules += t.productions()

    # Definir simbolo raiz
    S = nltk.Nonterminal('sentence')

    # inferir PCFG
    print("Inferiendo PCFG de la data")
    grammar = nltk.induce_pcfg(S, rules)
    print("Reglas del PCFG:",len(grammar.productions()))

    print("---------------------------------------------")
    # consultar regla Z -> X Y para Z=NC
    pos = 'NC'
    print("Muestra de reglas para",pos)
    consulta = grammar.productions(lhs=nltk.Nonterminal(pos))
    for i in range(min(len(consulta),10)):
        print(consulta[i])
    print("---------------------------------------------")

    # definir parser
    parser = nltk.InsideChartParser(grammar)
    #parser.trace(2)

    """
    test = 'el perro ladra'.split()

    res = parser.parse(test)
    for t in res:
        print(t)
        t.draw()
    """
    print("Guardando predicciones para evaluacion...")
    gold = open('gold','w')
    test = open('test','w')
    for t in test_set:
        gold_tree = no_bl.sub('',str(t))
        gold_tree = no_dup_spaces.sub(' ',gold_tree)
        try:
            tt_iter = parser.parse(t.leaves())
            found = True
            test_tree = ''
            for tt in tt_iter:
                if not tt:
                    found=False
                    break
                test_tree = no_bl.sub('',str(tt) )
                test_tree = no_dup_spaces.sub(' ',test_tree)
                break
            
            if found and test_tree!='':
                gold.write(gold_tree+'\n')
                test.write(test_tree+'\n')
        except:
            print("--word not found in training data...")