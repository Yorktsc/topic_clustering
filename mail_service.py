from bosonnlp import BosonNLP
import numpy as np
import datetime
import pymysql as ps
import csv
import re
from gensim import models
from gensim import similarities
from gensim import corpora
from math import sqrt
import smtplib, ssl
from email.mime.text import MIMEText
import json
import logging
import traceback
import requests
import sys
from gensim.parsing.preprocessing import preprocess_string

"""
1.stopwords
2.parse
"""

"""HyperParameters"""
stopwordspath = "/home/jack/sicheng/chinese_stop_words.txt"
nlp = BosonNLP('ESArJA-D.35336.97a5OVDALCAk')
POS2filter = ['p', 'pba', 'pbei', 'c', 'u', 'uzhe', 'ule', 'uguo', 'ude', 'usuo', 'udeng', 'uyy', 'udh', 'uzhi', 'ulian', 'y', 'o', 'w','wkz','wky','wyz','vshi','wyy','wj','ww','wt','wd','wf','wn','wm','ws','wp','wb','wh','email', 'tel', 'id', 'ip', 'url']
RARE_WORD_LEVEL = 2
NUM_TOPICS = 500

def remove_punctuation(text):
    """
    Remove Punctuations among Chinese Documents
    """
    punctuation = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+“”【】 《》"
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text

def remove_stop_words(corpus, stopwordspath):
    """
    Input is a list of token lists, aka corpus, e.x. [["我是","唐思成"],["我","有点","呆"]]
    Output is the corpus without stop words
    """
    stopwords = {}.fromkeys([ line.rstrip() for line in open(stopwordspath,"r",encoding='utf-8')])
    for word_list in corpus:
        for word in word_list:
            if word in stopwords:
                word_list.remove(word)
        return corpus


def load_from_database(flag):
    """
    return list of original text strings

    '/data/repo/news/datap/out-w2v/' + _path.split('.')[0] + '.txt'
    """
    titles = []
    paths = []
    urls = []
    db = ps.connect("rm-uf6it2u39agqevqh8.mysql.rds.aliyuncs.com","readonly","readonly","datap")
    cursor = db.cursor()
    test_command = "select distinct title,path,url from datap_news_html where id between 15021437 and 15021540"
    if flag == "CN":
        diff = datetime.timedelta(seconds = 3600 * 4)
        endtime = str(datetime.datetime.now())[:-7]
        starttime = str(datetime.datetime.now() - diff)[:-7]
        command = "select distinct title,path,url from datap_news_html where pdateTime between \'" + starttime + "\' and \'" + endtime + "' and source in ('sina', 'hexun', 'eastmoney', '10jqka', 'caijing', 'yicai', 'stockstar', 'wangyi', 'tencent', 'jrj', 'caixin', 'ftchinese', 'huxiu', 'ce', 'wallstreet', 'cnfol', 'sohu', 'ifeng', 'xueqiu', '36kr') and pageType != 'FOREIGN_NEWS'"
        cursor.execute(command)
    elif flag == "EN":
        diff = datetime.timedelta(seconds = 3600 * 12)
        endtime = str(datetime.datetime.now())[:-7]
        starttime = str(datetime.datetime.now() - diff)[:-7]
        command = "select distinct title,path,url from datap_news_html where pdateTime between \'" + starttime + "\' and \'" + endtime + "' and pageType = 'FOREIGN_NEWS' group by title, source"
        cursor.execute(command)
    else:
        cursor.execute(test_command)
    for row in cursor.fetchall():
        titles.append(row[0])
        paths.append(row[1])
        urls.append(row[2])
    return titles, paths, urls

def get_contents(paths):
    contents = []
    for path in paths:
        path = '/data/repo/news/datap/out-w2v/' + path.split('.')[0] + '.txt'
        f = open(path, "r")
        content = f.read()
        contents.append(content)
    return contents

def corpus2list(corpus, length):
    """
    Transforms the gensim corpus type into a normal list of lists/matrix,
    only return a random sample of given size.
    Input:gensim sparse corpus
    np.random.randint(low, high, size)
    Output:
    np.array(matrix): Normal Matrix
    randlist: List of indices of texts that are contained in the output matrix
    """
    matrix = []
    for vec in corpus:
        arr = np.zeros(length)
        for tup in vec:
            arr[tup[0]] = tup[1]
        matrix.append(arr)
    return np.array(matrix)

def filter_rare_words(texts, ts = RARE_WORD_LEVEL):
    """
    Function filter out words that occur only once throught the corpus in order to boost the performance
    Input --- The str of the file path
    Output --- List of Lists, each sublist corresponds to a list of Chinese tokens, e.g: ["我们","都是","中国人"]
    """
    freqdict = defaultdict(default)
    for text in texts:
        for token in text:
            freqdict[token] += 1
    word_once = set()
    for key in freqdict.keys():
        if freqdict[key] <= ts:
            word_once.add(key)
    texts = [
    [token for token in text if token not in word_once]
    for text in texts
    ]
    return texts

def parse_texts(inTxtLst, Flag):
    """
    inTxtLst is a list of original string texts
    retTxtLst is a list of parsed string texts with white spaces connecting tokens
    """
    if Flag == "CN":
        retTxtLst=[]
        texts = []
        id2text = {}
        id = 0
        for line in inTxtLst:
            text = remove_punctuation(line)
            a = nlp.tag(text)[0]
            res = []
            for i in range(len(a['word'])):
                if a['tag'][i] not in POS2filter:
                    res.append(a['word'][i])
            id2text[id] = line
            texts.append(res)
            id += 1
            if id % 100 == 0:
                print(id)
        print("finished parsing")
        texts = remove_stop_words(texts, stopwordspath)
        return texts, id2text
    elif Flag == "EN":
        retTxtLst=[]
        texts = []
        id2text = {}
        id = 0
        for line in inTxtLst:
            res = preprocess_string(line)
            id2text[id] = line
            texts.append(res)
            id += 1
            if id % 100 == 0:
                print(id)
        print("finished parsing")
        return texts, id2text

def single_pass(tfidf_corpus, threshold):
    centers = [np.array(tfidf_corpus[0])]
    clusters = [[np.array(tfidf_corpus[0])]]
    clusters_id = [[0]]
    for i in range(len(tfidf_corpus)):
        if i == 0:
            continue
        vec = np.array(tfidf_corpus[i])
        temp = []
        for j in range(len(centers)):
            center = centers[j]
            score = cos_sim(vec, center)
            temp.append(score)
        if max(temp) > threshold:
            index = temp.index(max(temp))
            clusters[index].append(vec)
            centers[index] = sum(clusters[index]) / len(clusters[index])
            clusters_id[index].append(i)
        else:
            clusters.append([vec])
            centers.append(vec)
            clusters_id.append([i])
    return clusters, centers,clusters_id

def sparse_sum(list_vec):
    return

def cos_sim(vec1, vec2):
    num = np.dot(vec1, vec2)
    den = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return num/den

def sparse_l2norm(vec):
    sum = 0.0
    for tup in vec:
        sum += vec[1] * vec[1]
    return sqrt(sum)

def sparse_dot_product(v1, v2):
    i1 = 0
    i2 = 0
    result = 0.0
    while i1 < len(v1) and i2 < len(v2):
        if v1[i1][0] == v2[i2][0]:
            result += v1[i1][1] * v2[i2][1]
            i1 += 1
            i2 += 1
        elif v1[i1][0] > v2[i2][0]:
            i2 += 1
        else:
            i1 += 1
    return result

def print_cluster_results(id2text, clusters_id):
    clusters_id = sorted(clusters_id, key=lambda x : - len(x))
    for cluster_id in range(len(clusters_id)):
        if len(clusters_id[cluster_id]) <= 3:
            continue
        print("Now it's displaying cluster {}, the size of the cluster is {}".format(cluster_id, len(clusters_id[cluster_id])) + "+" * 50)
        for id in clusters_id[cluster_id]:
            print(id2text[id] + '\n')

def mail_contents(titles, urls, clusters_id, Language):
    clusters_id = sorted(clusters_id, key=lambda x : - len(x))
    res = []
    for cluster_id in range(len(clusters_id)):
        if len(clusters_id[cluster_id]) <= 5 and Language == "CN":
            continue
        if len(clusters_id[cluster_id]) >= 100 and Language == "CN":
            continue
        if len(clusters_id[cluster_id]) <= 1 and Language == "EN":
            continue
        cluster = []
        #print("Now it's displaying cluster {}, the size of the cluster is {}".format(cluster_id, len(clusters_id[cluster_id])) + "+" * 50)
        for id in clusters_id[cluster_id]:
            title = titles[id]
            url = urls[id]
            cluster.append(title + ': ' + url)
            #print(title + ': ' + url + '\n')
        res.append(cluster)
    return res

def prepare_mail(res):
    #port = 465  # For SSL
    #smtp_server = "smtp.mxhichina.com"
    #sender_email = "sicheng.tang@tigerobo.com"  # Enter your address
    #receiver_email = "sicheng.tang@tigerobo.com"  # Enter receiver address
    #password = '990316fylqq...'
    date = str(datetime.datetime.now())
    content = ""
    for cluster in res:
        content += "New Cluster, size of the cluster is {}: ".format(len(cluster)) + "+" * 50 + "\n"
        for line in cluster:
            content += line + '\n'
    filename = "/home/jack/sicheng/" + date + ".txt"
    with open(filename, "w") as file:
        file.write(content)
    return filename
    #context = ssl.create_default_context()
    #msg = MIMEText(content, 'plain', 'utf-8')
    #subject = "新闻热点聚类"
    #msg['Subject'] = subject
    #msg['From'] = "sicheng.tang@tigerobo.com"
    #msg['To'] = "sicheng.tang@tigerobo.com"
    #msg["Accept-Language"] = "zh-CN"
    #msg["Accept-Charset"] = "ISO-8859-1,utf-8"

    #with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    #    server.login(sender_email, password)
    #    server.sendmail(sender_email, receiver_email, msg.as_string())

def send_mail(filename, language):
    date = str(datetime.datetime.now())
    with open(filename, "r") as file:
        content = file.read()
    if language == "CN":
        alert_mail_send("sicheng.tang@tigerobo.com,shuangxi.zhao@tigerobo.com,hongjun.li@tigerobo.com,shuzhao.li@tigerobo.com,yifan.li@tigerobo.com", "新闻热点聚类 " + date, content, language)
    elif language == "EN":
        alert_mail_send("sicheng.tang@tigerobo.com,shuangxi.zhao@tigerobo.com,hongjun.li@tigerobo.com,shuzhao.li@tigerobo.com,yifan.li@tigerobo.com,xinyuan.zhang@tigerobo.com", "新闻热点聚类 " + date, content, language)
    #alert_mail_send("sicheng.tang@tigerobo.com", "新闻热点聚类 " + date, content, language)

def alert_mail_send(to, subject, text, language):
    try:
        atta = json.dumps([{ 'filename': '2019-06-17 10:55:20.468291.txt', 'content' : text}])
        resp = requests.post(url='http://10.0.6.146:3009/send',
                             data={
                                 "to": to,
                                 "subject": subject + language,
                                 "text": text,
                             })
        print(resp.text)
        resp_json = json.loads(resp.text)
        logging.getLogger(__file__).info(resp_json)
        if resp_json['status'] == 'success':
            return True, None
        else:
            return False, str(resp_json)
    except Exception as err:
        traceback.print_exc()
        return False, str(err)

if __name__ == '__main__':
    Language = sys.argv[1]
    if Language == "CN":
        sens = 0.85
    elif language == "EN":
        sens = 0.7
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    titles, paths, urls = load_from_database(Language)
    contents = get_contents(paths)
    print("finished loading database"+ str(datetime.datetime.now()))
    texts, id2text = parse_texts(contents, Language)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    print("finished TF-IDF" + str(datetime.datetime.now()))
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics = NUM_TOPICS)
    corpus_lsi = lsi[corpus_tfidf]
    print("finished LSI"+ str(datetime.datetime.now()))
    corpus_lsi = corpus2list(corpus_lsi, len(dictionary))
    print("finished sparse matrix to list"+ str(datetime.datetime.now()))
    clusters, centers, clusters_id = single_pass(corpus_lsi, sens)
    print("finished single_pass"+ str(datetime.datetime.now()))
    #print_cluster_results(id2text, clusters_id)
    res = mail_contents(titles, urls, clusters_id, Language)
    filename = prepare_mail(res)
    send_mail(filename, Language)
    """
    print("finished sending CN Email"+ str(datetime.datetime.now()))
    titlesE, pathsE, urlsE = load_from_database("EN")
    contentsE = get_contents(pathsE)
    print("finished loading database"+ str(datetime.datetime.now()))
    textsE, id2textE = parse_texts(contentsE)
    dictionaryE = corpora.Dictionary(textsE)
    corpusE = [dictionaryE.doc2bow(text) for text in textsE]
    tfidfE = models.TfidfModel(corpusE)
    corpus_tfidfE = tfidfE[corpusE]
    print("finished TF-IDF"+ str(datetime.datetime.now()))
    lsiE = models.LsiModel(corpus_tfidfE, id2word=dictionaryE, num_topics = NUM_TOPICS)
    corpus_lsiE = lsiE[corpus_tfidfE]
    print("finished LSI"+ str(datetime.datetime.now()))
    corpus_lsiE = corpus2list(corpus_lsiE, len(dictionaryE))
    print("finished sparse matrix to list"+ str(datetime.datetime.now()))
    clustersE, centersE, clusters_idE = single_pass(corpus_lsiE, 0.5)
    print("finished single_pass"+ str(datetime.datetime.now()))
    #print_cluster_results(id2text, clusters_id)
    resE = mail_contents(titlesE, urlsE, clusters_idE, False)
    filenameE = prepare_mail(resE)
    send_mail(filenameE)
    """
