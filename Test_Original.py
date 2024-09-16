import os
# os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import argparse
import time
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')

from torch.optim import lr_scheduler
# import pickle
import cv2 as cv
from loss import SupConLoss, OrthoHashLoss
from models import MyDataset
# from models.compnet import co3net
from models.ccnet_ import ccnet, ccnet_hash, ccnet_hash_can
from models.compnet import co3net,co3net_hash_can
from models.Hash_ccnet import hash_ccnet, hash_ccnet_no
from utils import *
#from line_profiler import LineProfiler

import copy
import random
import re

def generate_p_list(K):
    return [1/K] * K

def generate_entry(s, i):

    ## *->1 1->2 0->0
    m = len(s)
    k = i + 1  # 确定的位数
    v = ['1'] * m
    specified_bits = random.sample(range(m), k)

    # 其中 i 个位的值与 s 中的位不同
    for idx in range(i):
        bit = specified_bits[idx]
        v[bit] = '2' if s[bit] == '0' else '0'

    # 其他确定的位与 s 中的位相同
    for idx in range(i, k):
        bit = specified_bits[idx]
        if s[bit] == '1':
            v[bit] = '2'
        elif s[bit] == '0':
            v[bit] = '0'
        # v[bit] = s[bit]

    return ''.join(v)


# code_str = code_str.replace('1', '2')
# # print(code_str)
# code_str = code_str.replace('*', '1')

#(featDB_train, r, p_list)
# @profile
def k_hidden_algorithm(s_list, r, p_list):
    NDBs_list = []

    for s in s_list:
        # NDBs = set()
        m = len(s)
        N = m * r
        Q = [0] + [sum(p_list[:i + 1]) for i in range(len(p_list))]

        ind = 0
        # print(ind)p
        NDBs = np.zeros((N,2048))
        while ind < N:
            rnd = random.random()
            i = next(i for i, q in enumerate(Q) if q > rnd)  # 找到满足条件的i
            # print(i)
            v = generate_entry(s, i)
            numbers = np.fromiter(v, dtype=int) - 1
            numbers = np.expand_dims(numbers, axis=0)
            NDBs[ind] = numbers
                
            ind = ind + 1

        NDBs_list.append(NDBs.transpose())

    return NDBs_list


#
# def neg_bank_generation(ndb):
#     temp_matrix = []
#     for i in range(len(ndb)):
#         temp = ndb[i]
#         j=0
#         print(i)
#         for code_str in temp:
#             # break
#             # print(code_str)
#             code_str = code_str.replace('1','2')
#             # print(code_str)
#             code_str = code_str.replace('*','1')
#             # print(code_str)
#             numbers = np.fromiter(code_str,dtype=int) - 1
#             # print('1')
#             numbers = np.expand_dims(numbers,axis=0)
#             # print('1')
#             if j == 0:
#                 single_fea = numbers
#             else:
#                 single_fea = np.concatenate((single_fea,numbers),axis=1)
#             j = j + 1
#
#         temp_matrix.append(single_fea)
#             # break
#     return temp_matrix
#

def test(model):

    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    path_hard = os.path.join(path_rst, 'rank1_hard')

    # train_set_file = './data/train_IITD.txt'
    # test_set_file = './data/test_IITD.txt'
    # path_rst = ''
    
    trainset = MyDataset(txt=train_set_file, transforms=None, train=False)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False)

    # batch_size = 512  # 128

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=2)

    fileDB_train = getFileNames(train_set_file)
    fileDB_test = getFileNames(test_set_file)

    # output dir
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    net = model

    net.cuda()
    net.eval()

    # feature extraction:

    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_train):

        data = datas[0]

        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]

    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    print('\n')

    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_test):

        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    if batch_id != 1:
        print('aaaa')

    print('completed feature extraction.')
    print('featDB_test.shape: ', featDB_test.shape)

    print('\nFeature Extraction Done!')

    print('start feature matching ...\n')

    print('Verification EER of the test-test set ...')

    print('Start EER for Test-Test Set! \n')

    # verification EER of the test set
    s = []  # matching score
    l = []  # intra-class or inter-class matching
    ntest = featDB_test.shape[0]
    ntrain = featDB_train.shape[0]

    for i in range(ntest):
        feat1 = featDB_test[i]

        for j in range(ntrain):
            feat2 = featDB_train[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_train[j]:  # same palm
                l.append(1)
            else:
                l.append(-1)

    if not os.path.exists(path_rst+'veriEER'):
        os.makedirs(path_rst+'veriEER')
    if not os.path.exists(path_rst+'veriEER/rank1_hard/'):
        os.makedirs(path_rst+'veriEER/rank1_hard/')

    with open(path_rst+'veriEER/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')

    print('\n------------------')
    print('Rank-1 acc of the test set...')
    # rank-1 acc
    cnt = 0
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]

        dis = np.zeros((ntrain, 1))

        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1

        idx = np.argmin(dis[:])

        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1
        else:
            testname = fileDB_test[i]
            trainname = fileDB_train[idx]
            # store similar inter-class samples
            im_test = cv.imread(testname)
            im_train = cv.imread(trainname)
            img = np.concatenate((im_test, im_train), axis=1)
            cv.imwrite(path_rst + 'veriEER/rank1_hard/%6.4f_%s_%s.png' % (
                np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(path_rst + 'veriEER/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)

    print('\n\nReal EER of the test set...')
    # dataset EER of the test set (the gallery set is not used)
    s = []  # matching score
    l = []  # genuine / impostor matching
    n = featDB_test.shape[0]
    for i in range(n - 1):
        feat1 = featDB_test[i]

        for jj in range(n - i - 1):
            j = i + jj + 1
            feat2 = featDB_test[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_test[j]:
                l.append(1)
            else:
                l.append(-1)

    print('feature extraction about real EER done!\n')

    with open(path_rst + 'veriEER/scores_EER_test.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')

def test_binary(model):

    print('Start Binarized Testing!!')
    print('Start Binarized Testing!!')
    print('Start Binarized Testing!!')

    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    binary_path_rst = path_rst.replace("rst_test","rst_test_binary")

    # path_rst +
    path_hard = os.path.join(binary_path_rst, 'rank1_hard')

    # train_set_file = './data/train_IITD.txt'
    # test_set_file = './data/test_IITD.txt'

    trainset = MyDataset(txt=train_set_file, transforms=None, train=False)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False)

    batch_size = 128  # 128

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=0)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=0)

    fileDB_train = getFileNames(train_set_file)
    fileDB_test = getFileNames(test_set_file)

    # output dir
    if not os.path.exists(binary_path_rst):
        os.makedirs(binary_path_rst)

    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    net = model

    net.cuda()
    net.eval()

    # feature extraction:

    featDB_train = []
    iddb_train = []
    codes = []

    str_codes = []
    for batch_id, (datas, target) in enumerate(data_loader_train):
        # datas
        # break
        data = datas[0]

        data = data.cuda()
        target = target.cuda()
        codes = net.getFeatureCode_binary(data)
        codes = codes.cpu().detach().numpy()

        temp_code = []
        # [code_list.append(np.array_str(codes[i])) for i in range(data.shape[0])]   # 维度有问题
        [temp_code.append(codes[i].tolist()) for i in range(data.shape[0])]
        y = target.cpu().detach().numpy()

        for j in range(len(temp_code)):
            code_list_ = [str(ss) for ss in temp_code[j]]
            code_list = ''.join(code_list_)
            str_codes.append(code_list)

        if batch_id == 0:
            # featDB_train = str_codes
            iddb_train = y
        else:
            # featDB_train = np.concatenate((featDB_train, codes), axis=0)
            # featDB_train = featDB_train.append(code_list)
            # featDB_train = featDB_train + str_codes
            iddb_train = np.concatenate((iddb_train, y))
        # break

    featDB_train = str_codes
    classNumel = len(set(iddb_train))
    num_training_samples = len(featDB_train)

    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    print('\n')
    #
    # for i in range(len(featDB_train))
    #     if '.' in featDB_train[i]:
    #         print('oo')
    #

    ### Generate Neg Database
    print('Generate Negative Database')
    ### Hyper Para
    K = 4
    r = 2
    p_list = generate_p_list(K)
    # neg_code = []
    neg_code = k_hidden_algorithm(featDB_train, r, p_list)
    print('generate neg bank')
    # neg_bank = neg_bank_generation(neg_code)

    #
    #
    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_test):

        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode_binary(data)

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    if batch_id != 1:
        print('aaaa')

    print('completed feature extraction.')
    print('featDB_test.shape: ', featDB_test.shape)

    print('\nFeature Extraction Done!')

    print('start feature matching ...\n')

    print('Verification EER of the test-test set ...')

    print('Start EER for Test-Test Set! \n')

    # verification EER of the test set
    s = []  # matching score
    l = []  # intra-class or inter-class matching
    ntest = featDB_test.shape[0]
    # ntrain = featDB_train.shape[0]
    ntrain = len(featDB_train)

    # for i in range(len(ntrain)):

    for i in range(ntest):
        # break 
        feat1 = featDB_test[i]
        # feat1 = feat1.reshape(feat1.shape[0],1)
        print('[Now / Total]: ', i, '/', ntest)
        for j in range(ntrain):
            # break
            # feat2 = featDB_train[j]
            feat2 = neg_code[j]
            # feat2 = feat2.transpose()
            dis = feat1 @ feat2
            dis = np.sum(dis) / feat2.shape[1]
            dis = np.arccos(np.clip(dis,-1,1)) / np.pi
            # dis = np.dot(feat1,feat2) / feat2.shape[1]

            # cosdis = np.dot(feat1, feat2)
            # cosdis = cosdis / feat1.size
            # dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
            # dis = 0.5 * (nbit - torch.matmul(codes.sign(), centroids.sign().t()))

            s.append(dis)

            if iddb_test[i] == iddb_train[j]:  # same palm
                l.append(1)
            else:
                l.append(-1)

    if not os.path.exists(binary_path_rst+'veriEER'):
        os.makedirs(binary_path_rst+'veriEER')
    if not os.path.exists(binary_path_rst+'veriEER/rank1_hard/'):
        os.makedirs(binary_path_rst+'veriEER/rank1_hard/')

    with open(binary_path_rst+'veriEER/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + binary_path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + binary_path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')

    print('\n------------------')
    print('Rank-1 acc of the test set...')
    # rank-1 acc
    cnt = 0
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]

        dis = np.zeros((ntrain, 1))

        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1

        idx = np.argmin(dis[:])

        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1
        else:
            testname = fileDB_test[i]
            trainname = fileDB_train[idx]
            # store similar inter-class samples
            im_test = cv.imread(testname)
            im_train = cv.imread(trainname)
            img = np.concatenate((im_test, im_train), axis=1)
            cv.imwrite(binary_path_rst + 'veriEER/rank1_hard/%6.4f_%s_%s.png' % (
                np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(binary_path_rst + 'veriEER/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)
    #
    # print('\n\nReal EER of the test set...')
    # # dataset EER of the test set (the gallery set is not used)
    # s = []  # matching score
    # l = []  # genuine / impostor matching
    # n = featDB_test.shape[0]
    # for i in range(n - 1):
    #     feat1 = featDB_test[i]
    #
    #     for jj in range(n - i - 1):
    #         j = i + jj + 1
    #         feat2 = featDB_test[j]
    #
    #         cosdis = np.dot(feat1, feat2)
    #         # dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
    #         cosdis = cosdis / feat1.size
    #         dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
    #
    #         s.append(dis)
    #
    #         if iddb_test[i] == iddb_test[j]:
    #             l.append(1)
    #         else:
    #             l.append(-1)
    #
    # print('feature extraction about real EER done!\n')
    #
    # with open(binary_path_rst + 'veriEER/scores_EER_test.txt', 'w') as f:
    #     for i in range(len(s)):
    #         score = str(s[i])
    #         label = str(l[i])
    #         f.write(score + ' ' + label + '\n')
    #
    # sys.stdout.flush()
    # os.system('python ./getGI.py' + '  ' + binary_path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')
    # os.system('python ./getEER.py' + '  ' + binary_path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')



def test_binary_hash(model):

    print('Start Binarized Testing!!')
    print('Start Binarized Testing!!')
    print('Start Binarized Testing!!')

    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    binary_path_rst = path_rst.replace("rst_test","rst_test_hash_binary")

    # path_rst +
    path_hard = os.path.join(binary_path_rst, 'rank1_hard')

    # train_set_file = './data/train_IITD.txt'
    # test_set_file = './data/test_IITD.txt'

    trainset = MyDataset(txt=train_set_file, transforms=None, train=False)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False)

    batch_size = 512  # 128

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=2)

    fileDB_train = getFileNames(train_set_file)
    fileDB_test = getFileNames(test_set_file)

    # output dir
    if not os.path.exists(binary_path_rst):
        os.makedirs(binary_path_rst)

    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    net = model

    net.cuda()
    net.eval()

    # feature extraction:

    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_train):

        data = datas[0]

        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode_hash_binary(data)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]

    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    print('\n')

    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_test):

        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode_hash_binary(data)

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    if batch_id != 1:
        print('aaaa')

    print('completed feature extraction.')
    print('featDB_test.shape: ', featDB_test.shape)

    print('\nFeature Extraction Done!')

    print('start feature matching ...\n')

    print('Verification EER of the test-test set ...')

    print('Start EER for Test-Test Set! \n')

    # verification EER of the test set
    s = []  # matching score
    l = []  # intra-class or inter-class matching
    ntest = featDB_test.shape[0]
    ntrain = featDB_train.shape[0]

    for i in range(ntest):
        feat1 = featDB_test[i]

        for j in range(ntrain):
            feat2 = featDB_train[j]

            cosdis = np.dot(feat1, feat2)
            cosdis = cosdis / feat1.size
            print('Beofre Clip: cosdis')
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
            # dis = 0.5 * (nbit - torch.matmul(codes.sign(), centroids.sign().t()))
            print('After:',dis)
            s.append(dis)

            if iddb_test[i] == iddb_train[j]:  # same palm
                l.append(1)
            else:
                l.append(-1)

    if not os.path.exists(binary_path_rst+'veriEER'):
        os.makedirs(binary_path_rst+'veriEER')
    if not os.path.exists(binary_path_rst+'veriEER/rank1_hard/'):
        os.makedirs(binary_path_rst+'veriEER/rank1_hard/')

    with open(binary_path_rst+'veriEER/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + binary_path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + binary_path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')

    print('\n------------------')
    print('Rank-1 acc of the test set...')
    # rank-1 acc
    cnt = 0
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]

        dis = np.zeros((ntrain, 1))

        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1

        idx = np.argmin(dis[:])

        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1
        else:
            testname = fileDB_test[i]
            trainname = fileDB_train[idx]
            # store similar inter-class samples
            im_test = cv.imread(testname)
            im_train = cv.imread(trainname)
            img = np.concatenate((im_test, im_train), axis=1)
            cv.imwrite(binary_path_rst + 'veriEER/rank1_hard/%6.4f_%s_%s.png' % (
                np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(binary_path_rst + 'veriEER/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)

    print('\n\nReal EER of the test set...')
    # dataset EER of the test set (the gallery set is not used)
    s = []  # matching score
    l = []  # genuine / impostor matching
    n = featDB_test.shape[0]
    for i in range(n - 1):
        feat1 = featDB_test[i]

        for jj in range(n - i - 1):
            j = i + jj + 1
            feat2 = featDB_test[j]

            cosdis = np.dot(feat1, feat2)
            # dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
            cosdis = cosdis / feat1.size
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_test[j]:
                l.append(1)
            else:
                l.append(-1)

    print('feature extraction about real EER done!\n')

    with open(binary_path_rst + 'veriEER/scores_EER_test.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + binary_path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')
    os.system('python ./getEER.py' + '  ' + binary_path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')



# perform one epoch
def fit(epoch, model, data_loader, phase='training'):
    if phase != 'training' and phase != 'testing':
        raise TypeError('input error!')

    if phase == 'training':
        model.train()

    if phase == 'testing':
        # print('test')
        model.eval()

    running_loss = 0
    running_correct = 0
    task_loss = 0
    run_hash_loss = 0
    # or_hash_criterion = OrthoHashLoss()

    for batch_id, (datas, target) in enumerate(data_loader):

        data = datas[0]
        data = data.cuda()

        data_con = datas[1]
        data_con = data_con.cuda()

        target = target.cuda()
        if phase == 'training':
            optimizer.zero_grad()
            output, fe1 = model(data, target)
            output2, fe2 = model(data_con, target)
            fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                output, fe1 = model(data, None)
                output2, fe2 = model(data_con, None)
                fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

        ce = criterion(output, target)
        ce2 = con_criterion(fe, target)
        # ce_hash = or_hash_criterion(output,fe,target)

        # ce = criterion(output,fe1,target)

        loss = weight1 * ce + weight2 * ce2
        # loss = ce
        task_loss = task_loss + loss.data.cpu().numpy()

        # loss = (1 - hash_weight) * loss #+ hash_weight * ce_hash
        if phase == 'training':
            loss.backward(retain_graph=None)  #
            optimizer.step()

#        run_hash_loss = run_hash_loss + ce_hash.cpu().detach().numpy()
        ## log
        running_loss += loss.data.cpu().numpy()

        preds = output.data.max(dim=1, keepdim=True)[1]  # max returns (value, index)
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().numpy()

        ## update


    ## log info of this epoch
    total = len(data_loader.dataset)
    loss = running_loss / total
    # task_loss = task_loss / total
    # run_hash_loss = run_hash_loss / total
    accuracy = (100.0 * running_correct) / total

    # if epoch % 10 == 0:
    #     print('epoch %d: \t%s loss is \t%7.5f ; task loss is \t%7.5f ; hash loss is \t%7.5f ; \t%s accuracy is \t%d/%d \t%7.3f%%' % (
    #     epoch, phase, loss, task_loss, run_hash_loss, phase, running_correct, total, accuracy))

    if epoch % 10 == 0:
        print(
            'epoch %d: \t%s loss is \t%7.5f ; \t%s accuracy is \t%d/%d \t%7.3f%%' % (
                epoch, phase, loss, phase, running_correct, total, accuracy))

    return loss, accuracy

if __name__== "__main__" :

    parser = argparse.ArgumentParser(
        description="CO3Net for Palmprint Recfognition"
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch_num", type=int, default=3000)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--weight1", type=float, default=0.8)
    parser.add_argument("--weight2", type=float, default=0.2)
    parser.add_argument("--id_num", type=int, default=460,
                        help="IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500 IITD_Right 230 Tongji_LR 300")
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--redstep", type=int, default=500)

    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=500)  ## 200 for Multi-spec 500 for RED
    parser.add_argument("--hash_weight",type = float,default=0.2)
    parser.add_argument("--weight_chan", type=float, default=0.8)

    ##Training Path
    parser.add_argument("--train_set_file", type=str, default='./data/train_IITD_self.txt')
    parser.add_argument("--test_set_file", type=str, default='./data/test_IITD_self.txt')

    ##Store Path
    parser.add_argument("--des_path", type=str, default='./results/checkpoint/')
    parser.add_argument("--path_rst", type=str, default='./results/rst_test/')
    parser.add_argument("--last_path",type=str,default='/data/YZY/Negative_Palm/Only_Hy_No_Tanh_Med_Arc_2/checkpoint/net_params.pth')
    # parser.add_argument("--last_path",type=str,default='./checkpoint/net_params.pth')
    # parser.add_argument("--best_path",type=str,default='/data/YZY/Negative_Palm/Only_Hy_No_Tanh_Med_Arc_2/checkpoint/net_params_best.pth')
    parser.add_argument("--best_path",type=str,default='./checkpoint/net_params_best.pth')
    args = parser.parse_args()
    # args = parser.parse_args()

    # print(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    batch_size = args.batch_size
    epoch_num = args.epoch_num
    num_classes = args.id_num
    weight1 = args.weight1
    weight2 = args.weight2
    weight_chan = args.weight_chan
    hash_weight = args.hash_weight

    print('weight of cross:', weight1)
    print('weight of contra:', weight2)
    print('tempture:', args.temp)

    des_path = args.des_path
    path_rst = args.path_rst

    if not os.path.exists(des_path):
        os.makedirs(des_path)
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    # path
    train_set_file = args.train_set_file
    test_set_file = args.test_set_file

    # dataset
    trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False, imside=128, outchannels=1)

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2, shuffle=True)
    data_loader_test = DataLoader(dataset=testset, batch_size=128, num_workers=2, shuffle=True)

    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    print('------Init Model------')
    # net = co3net(num_classes=num_classes)
    # best_net = co3net(num_classes=num_classes)
    # net = ccnet(num_classes=num_classes,weight=weight_chan)
    # best_net = ccnet(num_classes=num_classes,weight=weight_chan)

    net = ccnet_hash_can(num_classes=num_classes,weight=weight_chan)
    best_net = ccnet_hash_can(num_classes=num_classes,weight=weight_chan)
    # net = co3net_hash_can(num_classes=num_classes)
    # best_net = co3net_hash_can(num_classes=num_classes)

    # net.load_state_dict(torch.load(args.last_path))
    # best_net.load_state_dict(torch.load(args.best_path))
    # best_net.load_state_dict(torch.load(args.best_path),strict=False)
    temp_net = ccnet_hash_can(num_classes = 600,weight = weight_chan)
    for key in temp_net.state_dict().keys():
        if 'arclayer' not in key:
            best_net.state_dict()[key].data.copy_(temp_net.state_dict()[key])

    # net.cuda()

    print('ORIGINAL')
    # print('Test NDB')
    # test_binary(best_net)

    # print('------------\n')
    # print('Hash Binary Test')
    # print('Hash Binary Test')
    # test_binary_hash(net)
    # print('Last')
    # print('Last')
    # print('Last')
    # test(net)
    # print('Last_Binary')
    # print('Last_Binary')
    # print('Last_Binary')
    # test_binary(net)
    # print('------------\n')
    # print('Best')
    # print('Best')
    # print('Hash Binary Test')
    # print('Hash Binary Test')
    # test_binary_hash(best_net)
    print('best_best')
    print('best_best')
    print('best_best')
    test(best_net)
    # print('Best Binary')
    # print('Best Binary')
    # print('Best Binary')
    # test_binary(best_net)

    # test_binary_hash()