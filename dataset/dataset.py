# coding=utf-8
from __future__ import absolute_import
import os

class DataSet(object):
    def __init__(self, name, root_dir="~/datasets"):
        self.name = name
        self.root_dir = os.path.expanduser(root_dir)
        self.config = dict()
        self.config_path = os.path.expanduser(self.root_dir)

    def load_config(self):