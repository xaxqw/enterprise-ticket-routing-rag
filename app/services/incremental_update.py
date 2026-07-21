"""
增量更新：只处理新增或修改的文档，不用每次全量重建
"""
import os
import hashlib
import json


class IncrementalUpdater:
    def __init__(self, index_path="./data/processed/file_index.json"):
        self.index_path = index_path
        self.index = self._load_index()

    def _load_index(self):
        """加载文件索引记录"""
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """保存索引"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def _get_file_hash(self, file_path):
        """计算文件MD5指纹"""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def get_changed_files(self, directory):
        """扫描目录，找出新增和修改的文件"""
        new_files = []
        modified_files = []

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if not os.path.isfile(filepath):
                continue

            file_hash = self._get_file_hash(filepath)

            if filename not in self.index:
                new_files.append(filepath)
            elif self.index[filename] != file_hash:
                modified_files.append(filepath)

        return new_files, modified_files

    def update_index(self, file_path):
        """处理完后更新索引"""
        filename = os.path.basename(file_path)
        self.index[filename] = self._get_file_hash(file_path)
        self._save_index()
