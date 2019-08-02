from __future__ import print_function, unicode_literals
import sys
import time
import json

import requests


CLUSTER_PUSH_URL = 'http://api.bosonnlp.com/cluster/push/'
CLUSTER_ANALYSIS_URL = 'http://api.bosonnlp.com/cluster/analysis/'
CLUSTER_RESULT_URL = 'http://api.bosonnlp.com/cluster/result/'
CLUSTER_CLEAR_URL = 'http://api.bosonnlp.com/cluster/clear/'
CLUSTER_STATUS_URL = 'http://api.bosonnlp.com/cluster/status/'
TASK_ID = 'task_%d' % int(time.time())

# 注意：在测试时请更换为您的API Token
session = requests.Session()
session.headers['X-Token'] = 'ESArJA-D.35336.97a5OVDALCAk'
session.headers['Content-Type'] = 'application/json'


def cluster_status():
    resp = session.get(CLUSTER_STATUS_URL + TASK_ID)
    resp.raise_for_status()
    return resp.json()["status"]


def detail_results(docs, idx, result):
    print('=' * 50)
    print('第%d个聚类中共有%s份文档,如下:' % (idx + 1, result['num']))
    for doc in result['list']:
        print(docs[doc])
    print('-' * 20)
    print('本聚类的中心文档为:')
    print(docs[result['_id']])


def main():
    print('任务ID:', TASK_ID)

    print('读入数据...')
    with open('text_cluster.txt', 'rb') as f:
        docs = [line.decode('utf-8') for line in f if line]

    print('正在上传数据...')
    for i in range(0, len(docs), 100):
        data = json.dumps([{'_id': i + idx, 'text': text} for idx, text in enumerate(docs[i:i+100])])
        resp = session.post(CLUSTER_PUSH_URL + TASK_ID, data=data.encode('utf-8'))
        resp.raise_for_status()

    print('开始分析...')
    resp = session.get(CLUSTER_ANALYSIS_URL + TASK_ID)
    resp.raise_for_status()

    while True:
        status = cluster_status()
        if status == 'DONE':
            print('\n获取分析结果...')
            resp = session.get(CLUSTER_RESULT_URL + TASK_ID)
            resp.raise_for_status()
            clusters = resp.json()

            print('一共生成了%d个聚类' % len(clusters))
            clusters = sorted(clusters, key=lambda cluster: len(cluster['list']), reverse=True)
            for idx, cluster in enumerate(clusters):
                detail_results(docs, idx, cluster)

            resp = session.get(CLUSTER_CLEAR_URL + TASK_ID)
            resp.raise_for_status()
            break
        elif status == 'NOT FOUND':
            print('找不到聚类任务。')
            break
        elif status == 'ERROR':
            print('任务失败，请稍后重试。')
            break
        else:
            print('.', end='')
            sys.stdout.flush()
            time.sleep(0.05)


if __name__ == '__main__':
    main()
