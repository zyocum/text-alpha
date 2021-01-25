#!/usr/bin/env python3

"""This script computes Krippendorff's alpha, a measure of inter-annotator 
agreement (a.k.a. IAA or inter-annotator reliability).  The formulation of IAA
here considers textual annotations at a character level, so the expected input
are parallel corpora with textual annotations where labels are assigned to 
spans of text marked with character offsets.

The formula for expected and observed agreement, and the per-label and overall
alpha scores are based on this paper:

Krippendorff, Klaus. "Measuring the reliability of qualitative text analysis data."
Quality and quantity 38 (2004): 787-800.
https://repository.upenn.edu/cgi/viewcontent.cgi?article=1042&context=asc_papers
"""

import json
import sys
import uuid

from dataclasses import dataclass
from functools import lru_cache
from glob import iglob
from itertools import combinations
from itertools import product
from operator import itemgetter
from pathlib import Path
from types import MappingProxyType

def unique_id(data):
    """Get a UUID from a data blob
    
    This uses uuid3 so that the same blob will produce a consistent ID"""
    return uuid.uuid3(uuid.NAMESPACE_DNS, data)

def extent(obj):
    """Get the start and end offset attributes of a dict-like object
    
    a = {'startOffset': 0, 'endOffset': 5}
    b = {'startOffset': 0, 'endOffset': 10}
    c = {'startOffset': 5, 'endOffset': 10}
    
    extent(a) -> (0, 5)
    extent(b) -> (0, 10)
    extent(c) -> (5, 10)
    extent({}) -> (-1, -1)
    
    """
    return obj.get('startOffset', -1), obj.get('endOffset', -1)

def find_files(directory='.', pattern='*', recursive=True):
    yield from iglob('{}/**/{}'.format(directory, pattern), recursive=recursive)

class ADMCorpus(list):
    """A list-like class modeling a collection of ADM JSON files."""
    def __init__(
        self,
        directory,
        pattern='*.adm.json',
        recursive=True
    ):
        super().__init__()
        self.directory = directory
        if not Path(self.directory).is_dir():
            raise ValueError(f'{self.directory!r} is not a directory!')
        self.pattern = pattern
        self.recursive = recursive
        self.load()
    
    def load(self):
        self.extend([
            ADM(f) for f in find_files(
                directory=self.directory,
                pattern=self.pattern,
                recursive=self.recursive
            )
        ])

class ADM(dict):
    """A dict-like class modeling an ADM with some extra bells & whistles"""
    def __init__(self, filename):
        self.filename = Path(filename)
        if not self.filename.is_file():
            raise ValueError(f'{filename!r} is not a file!')
        self.load()
        self.data = self['data']
        self.attributes = self['attributes']
        self.metadata = self['documentMetadata']
        if not 'uuid' in self.metadata:
            self.metadata['uuid'] = [unique_id(self.data).hex]
        self.uuid = min(self.metadata['uuid'])
        if not 'source' in self.metadata:
            self.metadata['source'] = [self.filename.absolute().as_posix()]
    
    def __repr__(self):
        return f'<ADM docid={self.uuid}>'
    
    def __len__(self):
        return len(self.data.encode('UTF-16BE')) // 2
    
    def load(self):
        with open(self.filename, mode='r') as f:
            adm = json.loads(f.read())
            if 'entityMentions' in adm['attributes']:
                message = (
                    f'file: {self.filename}\n'
                    'This ADM JSON contains an "entityMentions" attribute '
                    'which signifies an older ADM version than this program '
                    'supports.  Please convert your ADM to the latest version '
                    'before proceeding.'
                )
                print(message, file=sys.stderr)
                sys.exit(1)
            self.update(adm)
    
    def mentions(self):
        """Generate a list of entity mentions augmented with their
        named entity types.
        """
        for entity in self.attributes['entities']['items']:
            for mention in entity['mentions']:
                mention['type'] = entity.get('type')
                yield mention
    
    def chunks(self, label=None):
        start, end = 0, 0
        prev_start, prev_end = 0, 0
        for mention in sorted(self.mentions(), key=extent):
            if (label is None) or (mention.get('type') == label):
                start, end = extent(mention)
                if prev_end < start:
                    # unlabeled gap
                    yield {'label': None, 'start': prev_end, 'end': start}
                # labeled extent
                yield {'label': mention['type'], 'start': start, 'end': end}
                prev_start, prev_end = start, end
        if end < len(self):
            # gap after the last labeled extent
            yield {'label': None, 'start': end, 'end': len(self)}
    
    def subjects(self, annotator, label=None):
        for chunk in self.chunks(label=label):
            yield LabeledExtent(
                **chunk,
                annotator=annotator,
                adm=self,
            )

@dataclass
class LabeledExtent:
    """A dataclass modeling individual, categorical textual annotation labels 
    with start/end offsets referring to the text of a particular annotated 
    document."""
    annotator: str
    adm: ADM
    label: str = None
    start: int = -1
    end: int = -1
    
    def __len__(self):
        return self.end - self.start

@lru_cache(maxsize=32, typed=True)
def sniff(
    annotators,
    pattern='*.adm.json',
    recursive=False,
    labels=None,
    verbose=False
):
    """Compute, cache, and return (as a tuple) the following:
    
    1. The total length of all documents annotated by at least 2 annotators 
    (i.e., what Krippendorff refers to as "continuum length").
    
    2. A mapping of document IDs to the parallel annotated documents annotated 
    by each annotator.
    
    3. The frequency with which annotators assigned each label in their 
    annotations.
    
    This function is cached for convenience because these values can be
    reused when computing both the observed and expected agreements.
    """
    corpora = {
        annotator: ADMCorpus(annotator, pattern=pattern, recursive=recursive)
        for annotator in sorted(annotators)
    }
    anns = {ann: i for i, ann in enumerate(corpora)}
    docids = {}
    for annotator, corpus in corpora.items():
        for adm in corpus:
            if adm.uuid not in docids:
                docids[adm.uuid] = [None] * len(anns)
            docids[adm.uuid][anns[annotator]] = adm
    continuums = {}
    label_counts = {}
    docid_counts = {}
    uncovered = set()
    for docid in docids:
        adms = list(filter(None, docids[docid]))
        if len(adms) > 1:
            for adm in adms:
                for label in (
                    s.label for s in adm.subjects(annotator=None)
                    if s.label is not None
                ):
                    if docid not in continuums:
                        continuums[docid] = len(adm)
                    if docid not in docid_counts:
                        docid_counts[docid] = {}
                    if (labels is None) or (label in labels):
                        if label not in label_counts:
                            label_counts[label] = 0
                        label_counts[label] += 1
                        if label not in docid_counts[docid]:
                            docid_counts[docid][label] = 0
                        docid_counts[docid][label] += 1
        else:
            uncovered.add(docid)
    for docid in uncovered:
        docids.pop(docid)
    if verbose:
        print(
            f'Assessing {len(docids)} documents covered in parallel by '
            f'{len(annotators)} annotators ...',
            file=sys.stderr
        )
    return (
        MappingProxyType(continuums),
        MappingProxyType(docids),
        MappingProxyType(label_counts),
        MappingProxyType(docid_counts)
    )

@lru_cache(maxsize=32, typed=True)
def observation(
    annotators,
    pattern='*.adm.json',
    recursive=True,
    labels=None,
    verbose=False
):
    """Compute per-label observed agreement."""
    continuums, docids, label_counts, docid_counts = sniff(
        annotators,
        pattern=pattern,
        recursive=recursive,
        labels=labels,
        verbose=verbose
    )
    anns = {ann: i for i, ann in enumerate(annotators)}
    observations = {label: 0 for label in label_counts}
    for label in label_counts:
        for ann_i, ann_j in combinations(annotators, 2):
            i, j = anns[ann_i], anns[ann_j]
            for docid in docids:
                adm_i = docids[docid][i]
                adm_j = docids[docid][j]
                if adm_i and adm_j:
                    subjects_g = list(adm_i.subjects(ann_i, label=label))
                    subjects_h = list(adm_j.subjects(ann_j, label=label))
                    for subjects in product(subjects_g, subjects_h):
                        subject_g, subject_h = sorted(subjects, key=len)
                        if all((
                            subject_g.label == label,
                            subject_h.label == label,
                            -len(subject_g) < (subject_g.start - subject_h.start),
                            (subject_g.start - subject_h.start) < len(subject_h)
                        )):
                            observations[label] += (subject_g.start - subject_h.start) ** 2.0 + \
                                            (subject_g.start + len(subject_g) - \
                                             subject_h.start - len(subject_h)) ** 2.0
                        elif all((
                            subject_g.label == label,
                            subject_h.label != label,
                            (len(subject_h) - len(subject_g)) >= (subject_g.start - subject_h.start),
                            (subject_g.start - subject_h.start) >= 0
                        )):
                            observations[label] += len(subject_g) ** 2.0
                        elif all((
                            subject_g != label,
                            subject_h == label,
                            (len(subject_h) - len(subject_g)) <= (subject_g.start - subject_h.start),
                            (subject_g.start - subject_h.start) <= 0
                        )):
                            observations[label] += len(subject_h) ** 2.0
        observations[label] *= (len(annotators) ** 2.0) - len(annotators)
        observations[label] /= len(anns) * (len(anns) - 1) * (sum(continuums.values()) ** 2.0)
    return observations

@lru_cache(maxsize=32, typed=True)
def expectation(
    annotators,
    pattern='*.adm.json',
    recursive=True,
    labels=None,
    verbose=False
):
    """Compute per-label expected agreement."""
    continuums, docids, label_counts, docid_counts = sniff(
        annotators,
        pattern=pattern,
        recursive=recursive,
        labels=labels,
        verbose=verbose
    )
    continuum_length = sum(continuums.values())
    anns = {ann: i for i, ann in enumerate(annotators)}
    expectations = {label: 0 for label in label_counts}
    possible_locations = {label: 0 for label in label_counts}
    denominator = {label: 0 for label in label_counts}
    for label in label_counts:
        possible_locations = sum(
            (
               (len(annotators) * continuums[docid]) *
               (len(annotators) * (continuums[docid] - 1))
            ) for docid in continuums if docid_counts[docid].get(label)
        )
        denominator[label] = possible_locations
        for i, ann_i in enumerate(annotators):
            for docid in docids:
                adm_i = docids[docid][i]
                if adm_i:
                    subjects_g = list(adm_i.subjects(ann_i, label=label))
                    for g, subject_g in enumerate(subjects_g):
                        if subject_g.label == label:
                            expectations[label] += (
                                ((label_counts[label] - 1) / 3.0) * (
                                    (2.0 * len(subject_g) ** 3.0) -
                                    (3.0 * len(subject_g) ** 2.0) +
                                    len(subject_g)
                                )
                            )
                            for j, ann_j in enumerate(annotators):
                                adm_j = docids[docid][j]
                                if adm_j:
                                    subjects_h = list(adm_j.subjects(ann_j, label=label))
                                    for h, subject_h in enumerate(subjects_h):
                                        if subject_h.label != label:
                                            if len(subject_h) >= len(subject_g):
                                                expectations[label] += (len(subject_g) ** 2.0) * \
                                                                (len(subject_h) - len(subject_g) + 1)
                            denominator[label] -= len(subject_g) * (len(subject_g) - 1)
        expectations[label] *= (2.0 / continuum_length)
        expectations[label] /= denominator[label]
    return expectations

def alpha(
    annotators,
    pattern='*.adm.json',
    recursive=True,
    labels=None,
    verbose=False
):
    """Compute per-label alpha."""
    observed = observation(
        annotators,
        pattern=pattern,
        recursive=recursive,
        labels=labels,
        verbose=verbose
    )
    expected = expectation(
        annotators,
        pattern=pattern,
        recursive=recursive,
        labels=labels,
        verbose=verbose
    )
    assert set(observed) == set(expected)
    alpha = {}
    for label in observed:
        try:
            alpha[label] = 1 - (observed[label] / expected[label])
        except ZeroDivisionError:
            alpha[label] = float('nan')
    return alpha

def overall_alpha(
    annotators,
    pattern='*.adm.json',
    recursive=True,
    labels=None,
    verbose=False
):
    """Compute overall alpha (aggregated over all labels)."""
    observed = observation(
        annotators,
        pattern=pattern,
        recursive=recursive,
        labels=labels,
        verbose=verbose
    )
    expected = expectation(
        annotators,
        pattern=pattern,
        recursive=recursive,
        labels=labels,
        verbose=verbose
    )
    assert set(observed) == set(expected)
    try:
        return 1 - (sum(observed.values()) / sum(expected.values()))
    except ZeroDivisionError:
        return float('nan')

def main(
    annotators,
    pattern='*.adm.json',
    recursive=True,
    labels=None,
    pairwise=False,
    verbose=False
):
    """Command-line driver function."""
    if pairwise:
        pairs = list(combinations(annotators, 2))
        scores = {}
        matrix = [[''] * (len(annotators) - 1) for _ in annotators]
        index = {v: k for k, v in enumerate(annotators)}
        for pair in pairs:
            ann_i, ann_j = pair
            i, j = index[ann_i], index[ann_j]
            matrix[i][j-1] = '{:0.3f}'.format(
                overall_alpha(
                    pair,
                    pattern=pattern,
                    recursive=recursive,
                    labels=labels,
                    verbose=verbose
                )
            )
        header = [''] + [a.name for a in annotators][1:]
        print(*header, sep='\t')
        for ann, row in zip(annotators[:-1], matrix):
            row = [ann.name] + row
            print(*row, sep='\t')
    else:
        for label, score in sorted(
            alpha(
                annotators,
                pattern=pattern,
                recursive=recursive,
                labels=labels,
                verbose=verbose
            ).items(),
            key=itemgetter(1),
            reverse=True
        ):
            print(f'{label}\t{score}')
        overall_score = overall_alpha(
            annotators,
            pattern=pattern,
            recursive=recursive,
            labels=labels,
            verbose=verbose
        )
        print(f'overall α\t{overall_score}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument(
        'annotators',
        nargs='+',
        type=Path,
        help='paths to annotator directories (at least two)'
    )
    parser.add_argument(
        '-g', '--glob-pattern',
        default='*.adm.json',
        help='glob pattern for matching annotation files within each annotator directory'
    )
    parser.add_argument(
        '-R', '--non-recursive',
        action='store_false',
        default=True,
        help='if this is specified, annotator directories will not be searched recursively for annotatin files',
    )
    parser.add_argument(
        '-l', '--labels',
        nargs='+',
        default=None,
        help='allow-list of labels to check (by default, all labels in the data will be assessed)',
    )
    parser.add_argument(
        '-p', '--pair-wise',
        action='store_true',
        help='assess IAA pair-wise for each pair of annotators'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='write verbose output to stderr'
    )
    args = parser.parse_args()
    main(
        tuple(args.annotators),
        pattern=args.glob_pattern,
        recursive=args.non_recursive,
        labels=args.labels if args.labels is None else tuple(args.labels),
        pairwise=args.pair_wise,
        verbose=args.verbose
    )
