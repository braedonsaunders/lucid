"""Exact source-balanced sampling through a bounded virtual sequence index."""
from collections import defaultdict
import math


class SourceBalancedSequences:
    """Keep one RNG draw per sample; every source and its own sequences are uniform.

    The virtual index repeats shorter source groups to their least common
    multiple. It stores indices only, never duplicates image arrays, and refuses
    impractical index periods rather than silently introducing modulo bias.
    """
    def __init__(self, data, rows):
        if len(data) != len(rows) or not rows:
            raise ValueError('matching nonempty training sequences required')
        groups = defaultdict(list)
        seen = set()
        self.identities = []
        for index, row in enumerate(rows):
            if row['split'] != 'train' or row['id'] in seen:
                raise ValueError('unique training sequence identities required')
            seen.add(row['id'])
            self.identities.append(row['id'])
            groups[row['source_id']].append(index)
        self.data = data
        self.sources = tuple(groups)
        self.groups = tuple(tuple(indices) for indices in groups.values())
        self.period = math.lcm(*(len(group) for group in self.groups))
        if self.period > 1_000_000 or self.period * len(self.groups) > 100_000_000:
            raise ValueError('source group sizes require an impractical exact sampling index')
        self.metadata = {'strategy': 'uniform source, then uniform sequence within source',
                         'sources': len(self.groups), 'unique_sequences': len(rows),
                         'virtual_bins': len(self), 'period': self.period,
                         'sequence_counts': {source: len(group) for source, group in zip(self.sources, self.groups)}}

    def __len__(self):
        return len(self.groups) * self.period

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = self.groups[index // self.period]
        actual = group[(index % self.period) % len(group)]
        sample = self.data[actual]
        if sample[2] != self.identities[actual]:
            raise ValueError('sampling metadata does not match loaded sequence order')
        return sample
