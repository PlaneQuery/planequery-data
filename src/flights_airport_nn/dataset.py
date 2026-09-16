from collections.abc import Iterator, Sequence
from datetime import date

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, get_worker_info

from flights_airport_nn.features.features import (
    ALL_FEATURE_CONFIG,
    FeatureConfig,
    create_features_for_day,
)


AirportCandidateSample = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class AirportCandidateDataset(IterableDataset):
    """Create and stream airport-candidate training samples one day at a time."""

    def __init__(
        self,
        dates: Sequence[date],
        feature_config: FeatureConfig = ALL_FEATURE_CONFIG,
    ) -> None:
        super().__init__()
        self.dates = tuple(dates)
        self.feature_config = feature_config

    def __iter__(self) -> Iterator[AirportCandidateSample]:
        dates = self.dates
        worker = get_worker_info()
        if worker is not None:
            dates = dates[worker.id :: worker.num_workers]

        for target_date in dates:
            features = create_features_for_day(target_date)
            feature_values, masks, labels, _ = features.numpy_features(
                self.feature_config
            )

            for values, mask, label in zip(feature_values, masks, labels):
                yield (
                    torch.as_tensor(values, dtype=torch.float32),
                    torch.as_tensor(mask, dtype=torch.bool),
                    torch.as_tensor(label, dtype=torch.float32),
                )


def collate_airport_candidates(
    samples: list[AirportCandidateSample],
) -> AirportCandidateSample:
    """Pad samples so candidates from different days can share a batch."""

    feature_values, masks, labels = zip(*samples)
    return (
        pad_sequence(feature_values, batch_first=True),
        pad_sequence(masks, batch_first=True, padding_value=False),
        torch.stack(labels),
    )
