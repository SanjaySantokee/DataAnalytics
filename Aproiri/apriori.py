##source: http://adataanalyst.com/machine-learning/apriori-algorithm-python-3-0/

import numpy as np

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return list(map(frozenset, C1))#use frozen set so we can use it as a key in a dict  , and a set cannot be changed in a frozen set

#generates L1 from C1. Also returns a dictionary with support values.
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

#----Testing-------
dataSet = loadDataSet()
print(dataSet)

C1 = createC1(dataSet)
print(C1)

#D is a dataset in the setform.
D = list(map(set,dataSet))

L1,suppDat0 = scanD(D,C1,0.5)
print(L1)
print(suppDat0)
#------------------

def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

print("\n\nSecond Testing\n")
#------Testing---------------
L,suppData = apriori(dataSet)
print("L:", L)
print("L0:", L[0])
print("L1:", L[1])
print("L2:", L[2])
print("L3:", L[3])
print(suppData)
#----------------------------

# using the confidence to see what should be considered an assoc rule or not

#goes through L, if combo in L then it uses the calconseq.(rulesFromConseq)

def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList   

#calcConf() calculates the confidence of the rule and then find out the which rules meet the minimum confidence.
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

#checks on the reverse, if more than one item you can call this func to reverse it and change the order
#takes combination of more than one item, uses this func to create different arrangements and uses calc conf to see if these diff arrangements meet the min conf back again, if it is then it is part of the rules.

#rulesFromConseq() generates more association rules from our initial dataset. 
#This takes a frequent itemset and H, which is a list of items that could be on the right-hand side of a rule.
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
print("\n\n\n testing three\n")
#----Testing-----------
L,suppData= apriori(dataSet,minSupport=0.5)
rules= generateRules(L,suppData, minConf=0.7)
print(rules)
#----------------------