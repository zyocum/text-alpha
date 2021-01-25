#!/usr/bin/env python3

import json

from pathlib import Path
from operator import itemgetter

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

CONTINUUM = {'startOffset': 150, 'endOffset': 450}

SECTIONS = {
    'i': (
        {'startOffset': 150, 'length': 75, 'v': 0, 'label': 'c'},
        {'startOffset': 225, 'length': 70, 'v': 1, 'label': 'c'},
        {'startOffset': 295, 'length': 75, 'v': 0, 'label': 'c'},
        {'startOffset': 370, 'length': 30, 'v': 1, 'label': 'c'},
        {'startOffset': 400, 'length': 50, 'v': 0, 'label': 'c'},
        {'startOffset': 150, 'length': 30, 'v': 0, 'label': 'k'},
        {'startOffset': 180, 'length': 60, 'v': 1, 'label': 'k'},
        {'startOffset': 240, 'length': 60, 'v': 0, 'label': 'k'},
        {'startOffset': 300, 'length': 50, 'v': 1, 'label': 'k'},
        {'startOffset': 350, 'length': 100, 'v': 0, 'label': 'k'},
    ),
    'j': (
        {'startOffset': 150, 'length': 70, 'v': 0, 'label': 'c'},
        {'startOffset': 220, 'length': 80, 'v': 1, 'label': 'c'},
        {'startOffset': 300, 'length': 55, 'v': 0, 'label': 'c'},
        {'startOffset': 355, 'length': 20, 'v': 1, 'label': 'c'},
        {'startOffset': 375, 'length': 25, 'v': 0, 'label': 'c'},
        {'startOffset': 400, 'length': 20, 'v': 1, 'label': 'c'},
        {'startOffset': 420, 'length': 30, 'v': 0, 'label': 'c'},
        {'startOffset': 150, 'length': 30, 'v': 0, 'label': 'k'},
        {'startOffset': 180, 'length': 60, 'v': 1, 'label': 'k'},
        {'startOffset': 240, 'length': 60, 'v': 0, 'label': 'k'},
        {'startOffset': 300, 'length': 50, 'v': 1, 'label': 'k'},
        {'startOffset': 350, 'length': 100, 'v': 0, 'label': 'k'},
    )
}

def mention(contiuum, section):
    offset = min(extent(contiuum))
    return {
        'mentions': [{
            'startOffset': section['startOffset'] - offset,
            'endOffset': section['startOffset'] + section['length'] - offset
        }],
        'type': section['label']
    }

def entity_sort_key(entity):
    return extent(min(entity['mentions'], key=extent))

def adm(
    annotator,
    sections,
    contiuum
):
    contiuum_start = min(extent(contiuum))
    contiuum_length = max(extent(contiuum)) - contiuum_start
    adm = {
        'data': ' ' * (max(extent(CONTINUUM)) - min(extent(CONTINUUM))),
        'attributes': {
            'entities': {
                'type': 'list',
                'itemType': 'entities',
                'items': []
            }
        },
        'documentMetadata': {'annotator': [annotator]}
    }
    entities = []
    for section in sorted(sections, key=itemgetter('startOffset')):
        if section['v']:
            entities.append(mention(CONTINUUM, section))
    adm['attributes']['entities']['items'] = sorted(
        entities,
        key=entity_sort_key
    )
    return adm

def dump(adm, filename):
    with open(filename, mode='w') as f:
        print(json.dumps(adm, ensure_ascii=False, indent=4), file=f)

def main():
    for annotator, sections in SECTIONS.items():
        filename = Path(annotator) / Path('adm-test-1.adm.json')
        filename.parent.mkdir(parents=True, exist_ok=True)
        dump(adm(annotator, sections, CONTINUUM), filename)

if __name__ == '__main__':
    main()
