"""Check AuxDet VOC XML metadata without importing the training stack."""

import argparse
import collections
import os
import xml.etree.ElementTree as ET


def inspect_split(root, split):
    ids_file = os.path.join(root, 'VOC2007', 'ImageSets', 'Main', f'{split}.txt')
    ann_root = os.path.join(root, 'VOC2007', 'Annotations')
    counts = collections.Counter()
    total = 0
    if not os.path.isfile(ids_file):
        return {'split': split, 'missing_split_file': ids_file}
    with open(ids_file, encoding='utf-8') as handle:
        ids = [line.strip().split()[0] for line in handle if line.strip()]
    for image_id in ids:
        total += 1
        path = os.path.join(ann_root, f'{image_id}.xml')
        if not os.path.isfile(path):
            counts['missing_xml'] += 1
            continue
        root_node = ET.parse(path).getroot()
        size = root_node.find('size')
        for key in ('width', 'height'):
            value = size.findtext(key) if size is not None else None
            if value is None or int(value) <= 0:
                counts[f'invalid_{key}'] += 1
        for key, allowed in (('view', {'Air', 'Space', 'Land'}),
                             ('band_type', {'LWIR', 'NIR', 'SWIR'})):
            value = root_node.findtext(key)
            if value is None:
                counts[f'missing_{key}'] += 1
            elif value not in allowed:
                counts[f'invalid_{key}'] += 1
    return {'split': split, 'samples': total, 'issues': dict(counts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    args = parser.parse_args()
    for split in args.splits:
        print(inspect_split(args.data_root, split))


if __name__ == '__main__':
    main()

