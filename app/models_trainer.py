import multiprocessing
import os
import subprocess


class ModelsTrainer:

    def __init__(self, list_models: list) -> None:
        self.list_models = list_models

    def train(self, model) -> None:

        command = f'python {model}.py'

        subprocess.run(command.split())

    def train_models(self) -> None:
        with multiprocessing.Pool() as pool:
            pool.map(self.train, self.list_models)
