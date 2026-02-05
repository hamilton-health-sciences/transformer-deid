import torch
import argparse
import os
import numpy as np
from pathlib import Path
import logging
from collections import Counter
from tqdm import tqdm
from huggingface_hub import login
from transformer_deid.data import DeidDataset
from transformer_deid.tokenization import merge_sequences, encodings_to_label_list
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformer_deid.load_data import load_data, create_deid_dataset, save_labels
from transformer_deid.train import which_transformer_arch

logger = logging.getLogger(__name__)


def annotate(model, test_dataset: DeidDataset, device):
    """Annotates dataset with PHI labels.

       Args:
            model: trained HuggingFace model
            test_dataset: DeidDataset, see data.py
            device: either cuda or cpu

       Returns:
            new_labels: list of lists of [entity type, start, length] objects for each document
    """
    model = model.to(device)
    model.eval()

    logger.debug("Model num_labels=%s", getattr(model.config, "num_labels", None))
    logger.debug("Model id2label=%s", getattr(model.config, "id2label", None))
    logger.debug("Model label2id=%s", getattr(model.config, "label2id", None))

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
    flat_labels = [label for doc in predicted_label for label in doc]
    label_counts = Counter(flat_labels)
    logger.debug("Predicted label counts (top 10): %s", label_counts.most_common(10))
    if result:
        first_logits = result[0]
        max_logits = np.max(first_logits, axis=1)
        mean_max_logit = float(np.mean(max_logits))
        std_max_logit = float(np.std(max_logits))
        sample_len = min(50, len(predicted_label[0]))
        sample_counts = Counter(predicted_label[0][:sample_len]).most_common(5)
        logger.debug(
            "First sequence logits: mean max=%.4f std max=%.4f (len=%s).",
            mean_max_logit,
            std_max_logit,
            first_logits.shape[0],
        )
        logger.debug(
            "First sequence sample label counts (first %s tokens): %s",
            sample_len,
            sample_counts,
        )

    labels = [encodings_to_label_list(doc, test_dataset.encodings[i], id2label=id2label) for i, doc in enumerate(predicted_label)]

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
    baseArchitecture = os.path.basename(modelDir).split('-')[0].lower()
    _, _, base_model_name = which_transformer_arch(baseArchitecture)
    try:
        tokenizer = AutoTokenizer.from_pretrained(modelDir, use_fast=True)
        probe = tokenizer("probe", add_special_tokens=False)
        if getattr(tokenizer, "vocab_size", 0) in (0, None) or not probe.get("input_ids"):
            raise ValueError("Tokenizer appears unusable.")
        logger.debug("Tokenizer loaded from model path: %s", modelDir)
    except Exception as exc:
        logger.warning(
            "Failed to load tokenizer from model path (%s); falling back to base tokenizer %s. Error: %s",
            modelDir,
            base_model_name,
            exc,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

    data_dict = load_data(Path(train_path))
    test_dataset = create_deid_dataset(data_dict, tokenizer)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Running using {device}.')

    model = AutoModelForTokenClassification.from_pretrained(modelDir)
    logger.debug("Model loaded from: %s", getattr(model.config, "_name_or_path", None))

    annotations = annotate(model, test_dataset, device)

    if out_path is not None:
        logger.info(f'Saving annotations to {out_path}.')
        save_labels(annotations, data_dict['guid'], out_path)

    else:
        return annotations, test_dataset.ids


if __name__ == '__main__':
    args = parse_args()
    main(args)
