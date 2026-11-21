"""Student policies; supplied model and core remain unchanged."""
from __future__ import annotations
import re
import numpy as np
from lab06_data import load_reference

BLIND = ('name', 'university')

def my_extra_refusal(profile, purpose):
    """Validate text structure and the finite programme vocabulary before inference.

    Unicode names are accepted; this is not a real-world identity validator.
    """
    for field in ('name', 'university', 'program'):
        value = getattr(profile, field, None)
        if not isinstance(value, str) or not value.strip() or len(value) > 160:
            return 'invalid_text_field'
        if any(ord(ch) < 32 for ch in value):
            return 'text_control_character'
        if re.search(r'(?i)(ignore\s+.*instructions|\b(system|assistant|user)\s*:|\boutput\s*:|<\|)', value):
            return 'instruction_in_text'
    if profile.program not in load_reference().programs:
        return 'unsupported_program'
    return None

def my_extra_escalation(profile, decision):
    if not decision.reason.strip():
        return 'missing_explanation'
    if (decision.decision == 'reject' and profile.gpa >= 3.5
            and profile.exam_score >= 110 and profile.household_income_bracket <= 2):
        return 'strong_low_income_rejection'
    return None

def choose_band(scores):
    """Observed boundary: midpoint of reject Q90 and shortlist Q10, +/-2 points.

    A single-class baseline uses its median as an explicitly weak fallback.
    No after-run values enter calibration. Workload is measured, never capped silently.
    """
    r, s = scores['reject'], scores['shortlist']
    if r and s:
        centre = round((float(np.percentile(r, 90)) + float(np.percentile(s, 10))) / 2)
    else:
        values = r + s
        if not values:
            raise ValueError('No parsed scores: stop and inspect the baseline')
        centre = round(float(np.median(values)))
    return max(0, centre - 2), min(100, centre + 2)
