import os
import csv
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class History:
    def __init__(self, save_path: Optional[str] = None):
        self.history = []
        self.save_path = save_path
        self.csv_initialized = False
        self.headers = []
        if save_path:
            self._check_and_create_path()

    def _check_and_create_path(self):
        directory = os.path.dirname(self.save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def _initialize_csv(self, record: Dict):
        self.headers = list(record.keys())
        with open(self.save_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.headers)
            writer.writeheader()
        self.csv_initialized = True

    def append(self, record: Dict):
        self.history.append(record)
        if not self.save_path:
            return
        if not self.csv_initialized:
            self._initialize_csv(record)
        self._save_to_csv(record)

    def _save_to_csv(self, record: Dict):
        with open(self.save_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.headers)
            writer.writerow(record)

    def __getitem__(self, index):
        return self.history[index]

    def __iter__(self):
        return iter(self.history)

    def __len__(self):
        return len(self.history)
