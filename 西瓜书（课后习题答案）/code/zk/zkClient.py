#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Created by YWJ on 2018/1/26
from kazoo.client import KazooClient
from kazoo.client import KazooState

zk = KazooClient(hosts='127.0.0.1:2181')
if __name__ == '__main__':
    pass