from annotation_extractor import AnnotationExtractor
from dotenv import load_dotenv
import multiprocessing
from models_trainer import ModelsTrainer


def main():
    load_dotenv()
    list_models = ['vgg16_train', 'vgg16_train_from_scratch',
                   'vgg19_train', 'vgg19_train_from_scratch']
    models_trainer = ModelsTrainer(list_models)
    models_trainer.train_models()
    # annotation = AnnotationExtractor()
    # annotation.extract_annotation()
    # annotation.split_dataset()


if __name__ == "__main__":
    # multiprocessing.freeze_support()
    main()
