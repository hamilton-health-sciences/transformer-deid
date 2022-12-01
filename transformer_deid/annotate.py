import torch
import argparse
import os
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from data import DeidDataset
from tokenization import convert_subtokens_to_label_list, merge_sequences
from transformers import AutoModelForTokenClassification, AutoTokenizer
from load_data import load_data, create_deid_dataset
from train import which_transformer_arch

logger = logging.getLogger(__name__)


def annotate(modelDir: str, test_dataset: DeidDataset):
    """Annotates dataset with PHI labels.

       Args:
            modelDir: directory containing config.json, pytorch_model.bin, and training_args.bin
                e.g., './transformer_models/{base architecture}_model_{epochs}'
            test_dataset: DeidDataset, see data.py

       Returns:
            new_labels: list of lists of [entity type, start, length] objects for each document
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForTokenClassification.from_pretrained(modelDir).to(
        device)
    model.eval()

    logger.info(
        f'Running predictions over {len(test_dataset.encodings.encodings)} split sequences from {len(test_dataset.ids)} documents.'
    )
    result = [
        get_logits(encoding, model, device)
        for encoding in tqdm(test_dataset.encodings.encodings,
                             total=len(test_dataset.encodings.encodings))
    ]
    predicted_label = np.argmax(result, axis=2)

    id2label = model.config.id2label
    predicted_label = [[id2label[token] for token in doc]
                       for doc in predicted_label]

    labels = []
    for i, doc in enumerate(predicted_label):
        labels += [
            convert_subtokens_to_label_list(doc, test_dataset.encodings[i])
        ]

    new_labels = merge_sequences(labels, test_dataset.ids)

    return new_labels


def get_logits(encodings, model, device):
    """ Return predicted labels from the encodings of a *single* text example. """
    result = model(input_ids=torch.tensor([encodings.ids]).to(device),
                   attention_mask=torch.tensor([encodings.attention_mask
                                                ]).to(device))
    logits = result['logits'].cpu().detach().numpy()
    return logits[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a transformer-based PHI deidentification model.')

    parser.add_argument('-i',
                        '--test_path',
                        type=str,
                        help='string to diretory containing txt directory for the test set.')

    parser.add_argument('-m', '--model_path', type=str, help='model directory')

    parser.add_argument('-o',
                        '--output_path',
                        help='output path in which to optionally save the annotations',
                        default=None)

    args = parser.parse_args()

    return args


def main(args):
    train_path = args.test_path
    out_path = args.output_path
    modelDir = args.model_path

    baseArchitecture = os.path.basename(modelDir).split('_')[-3].lower()
    __, tokenizerArch, __ = which_transformer_arch(baseArchitecture)
    tokenizer = AutoTokenizer.from_pretrained(tokenizerArch)

    data_dict = load_data(Path(train_path))
    test_dataset = create_deid_dataset(data_dict, tokenizer)

    annotations = annotate(modelDir, test_dataset)
    if out_path is not None:
        # TODO: save annotations in out_path
        raise NotImplementedError('sorry! can\'t save yet!')

    else:
        return annotations, test_dataset.ids


if __name__ == '__main__':
    args = parse_args()
    main(args)
