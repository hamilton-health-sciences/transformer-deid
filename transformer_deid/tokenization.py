from bisect import bisect_left, bisect_right
import logging
from typing import List, Optional, Union, TextIO

from tqdm import tqdm
from transformer_deid.label import Label

logger = logging.getLogger(__name__)

def encode_tags(tags, encodings, tag2id):
    encoded_labels = [
        [-100 if tag == 'PAD' else tag2id[tag] for tag in doc] for doc in tags
    ]
    return encoded_labels


def assign_tags(
    encodings,
    labels,
    pad_token_label='PAD',
    default_label='O'
) -> list:
    """
    Assign labels to tokens in tokenized text.
    
    Effectively creates a list the same size as the encodings with the label names.
    Accomplishes this with the offset_mapping.

    label_offset: subtract this off label indices. Helps facilitate slicing
    documents into sub-parts.
    """

    token_labels = []

    # number of input examples
    N = len(encodings.encodings)
    for t in range(N):
        token_labels.append(
            assign_tags_to_single_text(
                encodings[t],
                labels[t],
                pad_token_label=pad_token_label,
                default_label=default_label
            )
        )

    return token_labels


def assign_tags_to_single_text(
    encoding,
    labels,
    pad_token_label='PAD',
    default_label='O'
):
    tokens = encoding.ids
    offsets = [o[0] for o in encoding.offsets]
    lengths = [o[1] for o in encoding.offsets]

    # assign default label to all tokens
    token_labels = [default_label] * len(tokens)

    # iterate through labels and assign entity types
    for label in labels:
        entity_type = label.entity_type

        # determine start/stop of the label
        start, offset = label.start, label.length
        stop = start + offset

        # find first token occurring on or after the label start index
        i = bisect_left(offsets, start)
        if i == len(offsets):
            # we have labeled a token at the end of the text
            # also catches the case that we label a part of a token
            # at the end of the text, but not the entire token
            token_labels[-1] = entity_type
        else:
            # find the last token which is within this label
            j = bisect_left(offsets, stop)

            # assign all tokens between [start, stop] to this label
            token_labels[i:j] = [entity_type] * (j - i)

    # determine which tokens are special tokens *or* sub-word tokens
    # these are assigned a pad token so the loss is not calculated over them
    # special tokens have words == None, subword tokens have the same word as the previous token
    token_labels = [
        pad_token_label if (encoding.words[i] is None) or
        ((i >= 0) and
         (encoding.words[i] == encoding.words[i - 1])) else token_label
        for i, token_label in enumerate(token_labels)
    ]

    return token_labels

def split_sequences(tokenizer, texts, labels=None):
    """
    Split long texts into subtexts of max length.
    If labels is provided, labels will be split in correspondence with texts.
    Return new list of split texts and new list of labels (if applicable).
    """
    # tokenize the text
    encodings = tokenizer(texts, add_special_tokens=False)
    seq_len = tokenizer.max_len_single_sentence

    # identify the start/stop offsets of the new text
    sequence_offsets = []
    logger.info('Determining offsets for splitting long segments.')
    for i, encoded in tqdm(enumerate(encodings.encodings), total=len(encodings.encodings)):
        offsets = [o[0] for o in encoded.offsets]
        token_sw = [False] + [
            encoded.word_ids[i + 1] == encoded.word_ids[i]
            for i in range(len(encoded.word_ids) - 1)
        ]
        # iterate through text and add create new subsets of the text
        start = 0
        subseq = []
        while start < len(offsets):
            # ensure we do not start on a sub-word token
            while token_sw[start]:
                start -= 1

            stop = start + seq_len
            if stop < len(offsets):
                # ensure we don't split sequences on a sub-word token
                # do this by shortening the current sequence
                while token_sw[stop]:
                    stop -= 1
            else:
                # end the sub sequence at the end of the text
                stop = len(offsets)

            subseq.append(start)

            # update start of next sequence to be end of current one
            start = stop
        
        sequence_offsets.append(subseq)

    
    new_text = []
    if labels:
        new_labels = []

    logger.info('Splitting text.')
    for i, subseq in tqdm(enumerate(sequence_offsets), total=len(encodings.encodings)):
        for j, start in enumerate(subseq):
            if j + 1 >= len(subseq):
                stop = len(encodings[i])
            else:
                stop = subseq[j+1]
            
            text_start = encodings[i].offsets[start][0]
            if stop >= len(encodings[i]):
                text_stop = encodings[i].offsets[-1][0] + encodings[i].offsets[-1][1]
            else:
                text_stop = encodings[i].offsets[stop][0]

            # extract the text from the offsets
            new_text.append(texts[i][text_start:text_stop])

            if labels:
                # subselect labels across examples, shifting them by the text offset
                subsetted_labels = [
                    label.shift(-text_start) for label in labels[i] if label.within(text_start, text_stop)
                ]
                new_labels.append(subsetted_labels)

    if labels:
        return new_text, new_labels
    else:
        return new_text

def expand_id_to_token(token_pred, ignore_value=None):
    # get most frequent label_id for this token
    p_unique, p_counts = np.unique(token_pred, return_counts=True)

    if len(p_unique) <= 1:
        return token_pred

    # remove our ignore index
    if ignore_value is not None:
        idx = np.where(p_unique == ignore_value)[0]
        if len(idx) > 0:
            # we know p_unique is unique, so get the only element
            p_unique = np.delete(p_unique, idx[0])
            p_counts = np.delete(p_counts, idx[0])

    if len(p_unique) == 1:
        idx = 0
    else:
        # TODO: warn user if we broke a tie by taking lowest ID
        idx = np.argmax(p_counts)

    # re-create the array with only the most frequent label
    token_pred = np.ones(len(token_pred), dtype=int) * p_unique[idx]
    return token_pred


def tokenize_text(tokenizer, text):
    """Split text into tokens using the given tokenizer."""
    if isinstance(tokenizer, stanfordnlp.pipeline.core.Pipeline):
        doc = tokenizer(text)
        # extract tokens from the parsed text
        tokens = [
            token.text for sentence in doc.sentences
            for token in sentence.tokens
        ]
    elif isinstance(tokenizer, spacy.tokenizer.Tokenizer):
        doc = tokenizer(text)
        # extract tokens from the parsed text
        tokens = [token.text for token in doc]
    else:
        if tokenizer is None:
            tokenizer = r'\w'
        # treat string as a regex
        tokens = re.findall(tokenizer, text)
    return tokens


def generate_token_arrays(
    text,
    text_tar,
    text_pred,
    tokenizer=None,
    expand_predictions=True,
    split_true_entities=True,
    ignore_value=None
):
    """
    Evaluate performance of prediction labels compared to ground truth.


    Args
        text_tar - N length numpy array with integers for ground truth labels
        text_pred - N length numpy array with integers for predicted labels
        tokenizer - Determines the granularity level of the evaluation.
            None or '' - character-wise evaluation
            r'\w' - word-wise evaluation
        expand_predictions - If a prediction is partially made for a
            token, expand it to cover the entire token. If not performed,
            then partially labeled tokens are treated as missed detections.
        split_true_entities - The ground truth label for a single token
            may correspond to two distinct classes (e.g. if word splitting,
            John/2010 would be one token but have two ground truth labels).
            Enabling this argument splits these tokens.
        ignore_value - Ignore a label_id in the evaluation. Useful for ignoring
            the 'other' category.
    """
    tokens_base = tokenize_text(tokenizer, text)

    tokens = []
    tokens_pred = []
    tokens_true = []
    tokens_start, tokens_length = [], []

    n_tokens = 0

    start = 0
    for token in tokens_base:
        # sometimes we have empty tokens on their own
        if len(token) == 0:
            continue
        start = text.find(token, start)
        token_true = text_tar[start:start + len(token)]
        token_pred = text_pred[start:start + len(token)]

        if all(token_true == -1) & all(token_pred == -1):
            # skip tokens which are not labeled
            start += len(token)
            n_tokens += 1
            continue

        if split_true_entities:
            # split the single token into subtokens, based on the true entity
            idxDiff = np.diff(token_true, prepend=0)
            if any(idxDiff > 0):
                # split
                idxDiff = np.diff(token_true, prepend=0)
                subtok_start = 0
                subtoken_true, subtoken_pred = [], []
                for subtok_end in np.where(idxDiff > 0)[0]:
                    subtoken_true.append(token_true[subtok_start:subtok_end])
                    subtoken_pred.append(token_pred[subtok_start:subtok_end])
                    subtok_start = subtok_end
                if subtok_end < len(token_true):
                    # add final token
                    subtoken_true.append(token_true[subtok_start:])
                    subtoken_pred.append(token_pred[subtok_start:])
            else:
                # in this case, there is only 1 label_id for the entire token
                # so we can just wrap in a list for the iterator later
                subtoken_true = [token_true]
                subtoken_pred = [token_pred]
        else:
            # do not split a token if there is more than 1 ground truth
            # consequently, tokens with multiple labels will be treated
            # as equal to the most frequent label
            subtoken_true = [token_true]
            subtoken_pred = [token_pred]

        # now iterate through our sub-tokens
        # often this is a length 1 iterator
        for token_true, token_pred in zip(subtoken_true, subtoken_pred):
            if len(token_true) == 0:
                continue

            if expand_predictions:
                # expand the most frequent ID to cover the entire token
                token_pred = expand_id_to_token(token_pred, ignore_value=-1)
                token_true = expand_id_to_token(token_true, ignore_value=-1)

            # get the length of the token for later
            token_len = len(token_true)

            # aggregate IDs for this token into the most frequent value
            if len(token_true) == 0:
                token_true = -1
            else:
                token_true = mode(token_true, ignore_value)
            if len(token_pred) == 0:
                token_pred = -1
            else:
                token_pred = mode(token_pred, ignore_value)

            # append the prediction for this token
            tokens_true.append(token_true)
            tokens_pred.append(token_pred)
            tokens.append(text[start:start + token_len])
            tokens_start.append(start)
            tokens_length.append(token_len)

            start += token_len
            # keep track of total tokens assessed
            n_tokens += 1

    # now we have a list of tokens with preds
    tokens_true = np.asarray(tokens_true, dtype=int)
    tokens_pred = np.asarray(tokens_pred, dtype=int)

    return tokens_true, tokens_pred, tokens, tokens_start, tokens_length


# methods for assigning labels // tokenizing
def get_token_labels(
    self,
    encoded,
    labels,
    pad_token_label_id=-100,
    default_label='O',
    label_offset_shift=0
):
    """
    label_offset_shift: subtract this off label indices. Helps facilitate slicing
    documents into sub-parts.
    """
    # construct sub-words flags
    # TODO: does this vary according to model?
    token_sw = [False] + [
        encoded.words[i + 1] == encoded.words[i]
        for i in range(len(encoded.words) - 1)
    ]

    # initialize token labels as the default label
    # set subword tokens to padded token
    token_labels = [
        pad_token_label_id if sw else default_label for sw in token_sw
    ]

    # when building examples for model evaluation, there are no labels
    if labels is None:
        label_ids = [
            self.label_set.label_to_id[default_label]
            for i in range(len(token_labels))
        ]
        return token_labels, label_ids

    offsets = [o[0] for o in encoded.offsets]
    for label in labels:
        entity_type = label.entity_type
        start, offset = label.start, label.length
        if label_offset_shift > 0:
            start -= label_offset_shift
            if start < 0:
                continue
        stop = start + offset

        # get the first offset > than the label start index
        i = bisect_left(offsets, start)
        if i == len(offsets):
            # we have labeled a token at the end of the text
            # also catches the case that we label a part of a token
            # at the end of the text, but not the entire token
            if not token_sw[-1]:
                token_labels[-1] = entity_type
        else:
            # find the last token which is within this label
            j = bisect_left(offsets, stop)

            # assign all tokens between [start, stop] to this label
            # *except* if it is a padding token (so the model ignores subwords)
            new_labels = [
                entity_type if not token_sw[k] else pad_token_label_id
                for k in range(i, j)
            ]
            token_labels[i:j] = new_labels

    label_ids = [
        self.label_set.label_to_id[l]
        if l != pad_token_label_id else pad_token_label_id for l in token_labels
    ]

    return token_labels, label_ids


def tokenize_with_labels(
    self, tokenizer, example, pad_token_label_id=-100, default_label='O'
):
    text = example.text

    # tokenize the text, retain offsets, subword locations, and lengths
    encoded = tokenizer.encode(text)
    offsets = [o[0] for o in encoded.offsets]
    lengths = [o[1] - o[0] for o in encoded.offsets]

    # TODO: do we need to fix it?
    # fix the offset of the final token, if special
    # if offsets[-1] == 0:
    #     offsets[-1] = len(text)

    word_tokens = encoded.tokens
    # construct sub-words flags
    # TODO: does this vary according to model?
    token_sw = [False] + [
        encoded.words[i + 1] == encoded.words[i]
        for i in range(len(encoded.words) - 1)
    ]

    token_labels = self.get_token_labels(
        encoded, example.labels, pad_token_label_id=-100, default_label='O'
    )

    return word_tokens, token_labels, token_sw, offsets, lengths


def convert_examples_to_features(
    self,
    examples: List,
    label_list: List[str],
    tokenizer,
    feature_overlap=None,
    include_offsets=False,
):
    """
    Loads a data file into a list of `InputFeatures`s

        `feature_overlap` - Split a single long example into multiple training observations. This is
        useful for handling examples containing very long passages of text.
            None (default): truncates each example at max_seq_length -> one InputFeature per InputExample.
            [0, 1): controls how much overlap between consecutive segments.
                e.g. `feature_overlap=0.1` means the last 10% of InputFeature 1 will equal first 10%
                of InputFeature 2, assuming that the InputExample is long enough to require splitting.
    """
    pad_token_label_id = -100
    features = []
    n_obs = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        # tokenize the example text
        encoded = tokenizer._tokenizer.encode(
            example.text, add_special_tokens=False
        )
        token_sw = [False] + [
            encoded.words[i + 1] == encoded.words[i]
            for i in range(len(encoded.words) - 1)
        ]
        token_offsets = np.array(encoded.offsets)

        seq_len = tokenizer.max_len_single_sentence
        if feature_overlap is None:
            feature_overlap = 0
        # identify the starting offsets for each sub-sequence
        new_seq_jump = int((1 - feature_overlap) * seq_len)

        # iterate through subsequences and add to examples
        start = 0
        while start < token_offsets.shape[0]:
            # ensure we do not start on a sub-word token
            while token_sw[start]:
                start -= 1

            stop = start + seq_len
            if stop < token_offsets.shape[0]:
                # ensure we don't split sequences on a sub-word token
                # do this by shortening the current sequence
                while token_sw[stop]:
                    stop -= 1
            else:
                # end the sub sequence at the end of the text
                stop = token_offsets.shape[0] - 1

            text = example.text[token_offsets[start, 0]:token_offsets[stop, 0]]
            encoded = tokenizer._tokenizer.encode(text)
            encoded.pad(tokenizer.model_max_length)

            # assign labels based off the offsets
            _, label_ids = self.get_token_labels(
                encoded,
                example.labels,
                pad_token_label_id=pad_token_label_id,
                label_offset_shift=token_offsets[start, 0]
            )

            features.append(
                InputFeatures(
                    input_ids=encoded.ids,
                    attention_mask=encoded.attention_mask,
                    token_type_ids=encoded.type_ids,
                    label_ids=label_ids,
                    # input_offsets=[o[0] for o in encoded.offsets],
                    # input_lengths=[o[1] - o[0] for o in encoded.offsets]
                )
            )
            n_obs += 1

            # update start of next sequence to be end of current one
            start = start + new_seq_jump

    return features
