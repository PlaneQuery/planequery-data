BiGRU

Train on adsb sequence data laballed from BTS flights. There is a mask for where I don't have labels. Label is first in sequence, last in sequence.

Infernence on adsb sequence

Future: Add CRF to enforce transisitions

# Validation metrics
How to do validation. I have BTS which is partial. I have mask =0/1 

I could figure out which tests flights I have based on which ones only move in distance when there is a flight. 

# Common commands
watch -n 1 nvidia-smi