"""
Author： YuYue Yang
Date：2021/10/15~
Assignment 2 Kaggle score 0.66900
Achieve BM25
b = 0.8125, K1=1.925
Doc Part
    IDF weight calculation:math.log10(1 + ((N - value + 0.5) / (value + 0.5)))
    TF weight calculation: count > 0 -> count
Que Part
    TF weight calculation: count / temp_total => number of occurrences of a single word / total number of words in the article
cos_sim:Molecular = (param_K1 + 1) * (cur_tf.get(word, 0)) * Doc_IDFList[word]
        z = (1 - param_b) + (param_b * dlen[i] / avgLength)
        Denominator = param_K1 * z + (cur_tf.get(word, 0))
        tmp_ans += (Molecular / Denominator)
DataSet：5000 Documents, 50 Queries(.txt)
Running time: 0.047 hours
"""
import gc
import math
import os
import time


# Build Dictionary
def BuildDic():
    # global wordSet
    global texts
    # wordSet = set()
    stopwords = set(['for', 'a', 'of', 'the', 'and', 'to', 'in'])
    texts = []
    # search file list step by step
    for docfile in DocFileList:
        f = open(Doc_str + docfile)
        temp_split = f.read().lower().split(' ')
        # Remove words that appear only once
        subDoc = []
        for word in set(temp_split):
            if temp_split.count(word) > 1:
                subDoc.append(word)
        texts.append(subDoc)
        f.close()
    print(len(texts))
    print("===============Build Dictionary End===============")


# ComputeIDF
def ComputeIDF():
    # Initialization variable--------------------
    global Doc_IDFList
    global N, Ni
    N = len(DocFileList)
    Ni = {}
    Doc_IDFList = {}
    for docfile in DocFileList:
        f = open(Doc_str + docfile)
        temp_split = f.read().split(' ')
        for word in set(temp_split):
            Ni[word] = Ni.get(word, 0) + 1
        f.close()
    # Start IDF calculation
    for word, value in Ni.items():
        Doc_IDFList[word] = math.log10(1 + ((N - value + 0.5) / (value + 0.5)))

    print("Doc_IDFList：" + str(len(Doc_IDFList)))
    print("===============ComputeIDF End===============")


# ComputTF
def ComputeTF():
    # Initialization variable -----------------
    global Doc_TFList
    global dlen
    # dlen is the length of the document, avgLength is the average length of the document
    global avgLength
    Doc_TFList = []
    dlen = []
    tfDic = {}
    for docfile in DocFileList:
        path_string = Doc_str + docfile
        file = open(path_string)
        temp_split = file.read().split(' ')
        dlen.append(len(temp_split))
        # Record the number of occurrences of each word in each document
        for word in set(temp_split):
            count = temp_split.count(word)
            if count > 0:
                tfDic[word] = count
        Doc_TFList.append(tfDic)
        # Zero setting
        tfDic = {}
    avgLength = sum(dlen) / len(DocFileList)
    print(len(Doc_TFList))
    print("avgLength：" + str(avgLength))
    print("===============ComputeTF End===============")
    print("\n")


def cos_similarity(q_index, queList):
    global param_b
    param_b = 0.84  # 0.75
    param_K1 = 1.96  # 1.85
    tmp_ans = 0
    for i in range(0, 4999):
        cur_tf = Doc_TFList[i]
        for word in queList:
            if word not in cur_tf:
                continue
            Molecular = (param_K1 + 1) * (cur_tf.get(word, 0)) * Doc_IDFList[word]
            z = (1 - param_b) + (param_b * dlen[i] / avgLength)
            Denominator = param_K1 * z + (cur_tf.get(word, 0))
            tmp_ans += (Molecular / Denominator)
        use2sort[DocFileList[i]] = tmp_ans
        tmp_ans = 0
        # Molecular, Denominator = 0, 0


# Write first line to result.txt
def pre_write2res():
    write_path = 'D:/Python/NLP_DataSet/q_50_d_5000_2/result.txt'
    f = open(write_path, 'a')
    f.write("Query,RetrievedDocuments" + '\n')
    f.close()


# Write result to result.txt
def write2res(quefile, sort_res):
    write_path = 'D:/Python/NLP_DataSet/q_50_d_5000_2/result.txt'
    f = open(write_path, 'a')
    f.write(quefile.replace('.txt', '') + ',')
    for i in range(0, 4999):
        f.write(sort_res[i][0].replace('.txt', ''))
        if i <= 4998:
            f.write(" ")
    f.write('\n')
    f.close()


if __name__ == '__main__':
    global DocFileList, QueFileList
    global Doc_str, Que_str
    global doc_index
    global use2sort
    start = time.time()
    Doc_str = 'D:/Python/NLP_DataSet/q_50_d_5000_2/docs/'
    Que_str = 'D:/Python/NLP_DataSet/q_50_d_5000_2/queries/'
    DocFileList = os.listdir(Doc_str)
    QueFileList = os.listdir(Que_str)
    doc_index = 0
    que_index = 0
    use2sort = dict.fromkeys(DocFileList, 0)

    pre_write2res()  # First. Write the first line
    BuildDic()
    ComputeTF()
    ComputeIDF()

    print("Build Time：" + str((time.time() - start) / 60 / 60))

    gc.collect()
    for quefile in QueFileList:
        f = open(Que_str + quefile)
        q_split = f.read().split(' ')
        cos_similarity(que_index, q_split)
        que_index += 1
        # Sort
        sort_res = sorted(use2sort.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        print(str(quefile) + " End")
        # Zero setting
        use2sort = dict.fromkeys(DocFileList, 0)
        # write result in file
        write2res(str(quefile), sort_res)

    endt = time.time()
    print("Running Time：" + str((endt - start) / 60 / 60))  # Print running time of system
