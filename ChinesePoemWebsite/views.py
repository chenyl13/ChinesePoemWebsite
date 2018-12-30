# coding=UTF-8
from django.http import HttpResponse
from django.shortcuts import render
# from main import infer
 
def index(request):
    return render(request, 'index.html',)
    # , {'post_list': post_list})

def results(request):
    if request.method == 'POST':
        first_word = request.POST.get("first_word", None)
        print first_word
        # poem = infer(model, final, words, word2int, dataset.emb)
        poem = infer()
        return render(request, 'results.html', {'data': poem})

def infer():
    return [u"春风飘雨霁，",u"天地已无尘，",u"水色连山色，",u"山深水上清。"]