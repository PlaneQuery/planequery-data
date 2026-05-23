import re

# Only fix strings that are EXACTLY 7 chars and match: 'LN' + 5 alphanumerics
# Example: LN747CP -> N747CP
FIX_LN_TO_N_7 = re.compile(r'^(?=.{7}$)LN([A-Z0-9]{5})$')

def normalize_tail(s: str) -> str:
    return FIX_LN_TO_N_7.sub(r'N\1', s.strip().upper())
