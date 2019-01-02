# coding=UTF-8
from __future__ import absolute_import, division, print_function
from codecs import open as open

import torch
import pdb
import tqdm
import numpy as np
import argparse
import sys
import os
import random

from django.http import HttpResponse
from django.shortcuts import render

torch.random.manual_seed(0)
random.seed(0)
dim_PE = 100
PE_const = 1000
PE_tmp_divider = [float(np.power(PE_const, i / float(dim_PE))) for i in range(dim_PE)]
 
def index(request):
    return render(request, 'index.html',)
    # , {'post_list': post_list})

def results(request):
    if request.method == 'POST':
        first_word = request.POST.get("first_word", None)
        print(first_word)
        # poem = infer(model, final, words, word2int, dataset.emb)
        poem = generate(first_word)
        return render(request, 'results.html', {'data': poem})

def generate(first_word):
    # return [u"春风飘雨霁，",u"天地已无尘，",u"水色连山色，",u"山深水上清。"]
    checkpoint = torch.load('./model/production.pth')
    hidden_size = 128
    model, final, words, word2int, emb = checkpoint['model'], checkpoint['final'], checkpoint['words'], checkpoint['word2int'], checkpoint['emb']
    print('Finish Loading')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    final.to(device)
    if True:
        try:
            poem = infer(model, final, words, word2int, emb, hidden_size = model.hidden_size, start=first_word)
        except KeyError:
            poem = u'此字在语料库中未出现过，请更换首字'
    return poem

def infer(model, final, words, word2int, emb, hidden_size=256, start=u'春', n=1):
    dim_PE = 100
    PE_const = 1000
    device = torch.device('cpu') if isinstance(final.weight, torch.FloatTensor) else final.weight.get_device()
    h = torch.zeros((1, n, hidden_size))
    x = torch.nn.functional.embedding(torch.full((n,), word2int[start], dtype=torch.long), emb).unsqueeze(0)
    ret = [[start] for i in range(n)]
    for i in range(19):
        # add PE dims
        pe = torch.tensor(pos2PE((i % 5) + 1), dtype=torch.float).repeat(1, n, 1)
        
        x, h, pe = x.to(device), h.to(device), pe.to(device)
        x = torch.cat((x, pe), dim=2)
        x, h = model(x, h)
        # h = torch.rand((1, n, hidden_size))
        w = prob_sample(torch.nn.functional.softmax(final(x.view(-1, hidden_size)), dim=-1).data.cpu().numpy())
        # w = torch.argmax(, dim = 1).cpu()
        x = torch.nn.functional.embedding(torch.from_numpy(w), emb).unsqueeze(0)
        for j in range(len(w)):
            ret[j].append(words[w[j]])
            if i % 5 == 3:
                if sys.version_info.major == 2:
                    ret[j].append(u"，" if i < 18 else u"。")
                else:
                    ret[j].append("，" if i < 18 else "。")
    ret_list = []
    for i in range(n):
        if sys.version_info.major == 2:
            ret_list.append(u"".join(ret[i]))
        else:
            ret_list.append("".join(ret[i]))
    return ret_list

def pos2PE(pos):
    PE_tmp = pos * np.ones(dim_PE) / PE_tmp_divider
    PE_tmp[0::2] = np.sin(PE_tmp[0::2])
    PE_tmp[1::2] = np.cos(PE_tmp[1::2])
    return PE_tmp

def prob_sample(w_list, topn = 10):
    samples = []
    for weights in w_list:
        idx = np.argsort(weights)[::-1]
        t = np.cumsum(weights[idx[:topn]])
        s = np.sum(weights[idx[:topn]])
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        samples.append(idx[sample])
    return np.array(samples)