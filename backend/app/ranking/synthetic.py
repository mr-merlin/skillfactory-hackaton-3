import random

import numpy as np
import pandas as pd

from .normalize import ORGAN_NOTES

# channel → (aroma_id, note), (aroma_id, note)
CHANNEL_MAP = {
    0: [(1000, "жасмин"), (1001, "мускус")],
    1: [(1002, "бергамот"), (1003, "сандал")],
    2: [(1004, "амбра"), (1005, "ваниль")],
    3: [(1006, "пачули"), (1007, "роза")],
    4: [(1008, "белый кедр"), (1009, "ветивер")],
    5: [(1010, "ирис"), (1011, "мандарин")],
}

NOTE_TO_CHANNEL = {}
for ch, pairs in CHANNEL_MAP.items():
    for _, note in pairs:
        NOTE_TO_CHANNEL[note] = ch


def generate_confused_sessions(
    perfume_vectors: dict[int, np.ndarray],
    note_to_idx: dict[str, int],
    aroma_notes_map: pd.DataFrame,
    n_per_group: int = 5,
    seed: int = 42,
) -> list[tuple[dict[str, float], int]]:
    """Generate synthetic pairs where user_vec comes from SKU_A but target is SKU_B.

    A and B share the same organ-note fingerprint (collision group).
    This teaches the model that exact note match ≠ correct target.
    It simulates real behavior where the user describes a similar-but-different perfume.
    """
    from .profile.build_profile import recipe_to_user_vector

    rng = np.random.default_rng(seed)

    organ_idxs = {n: note_to_idx[n] for n in ORGAN_NOTES if n in note_to_idx}
    channel_to_aromas = {}
    for ch, pairs_ in CHANNEL_MAP.items():
        channel_to_aromas[ch] = [aid for aid, _ in pairs_]

    # Build collision groups
    groups = build_collision_groups(perfume_vectors, note_to_idx)
    # Only groups with 2+ members
    multi_groups = {fp: pids for fp, pids in groups.items() if len(pids) >= 2}

    results = []
    for fp, pids in multi_groups.items():
        active_notes = {ORGAN_NOTES[i] for i, v in enumerate(fp) if v == 1}
        if not active_notes:
            continue

        target_channels = {NOTE_TO_CHANNEL[n] for n in active_notes if n in NOTE_TO_CHANNEL}

        for _ in range(n_per_group):
            # Pick source SKU (to generate user_vec from) and target SKU (different!)
            if len(pids) < 2:
                continue
            source, target = rng.choice(pids, size=2, replace=False)

            # Generate user_vec from SOURCE (not target!)
            target_focus = rng.normal(0.57, 0.22)
            target_focus = float(np.clip(target_focus, 0.15, 0.95))

            total_budget = rng.normal(175, 50)
            total_budget = float(np.clip(total_budget, 60, 350))

            target_budget = total_budget * target_focus
            noise_budget = total_budget * (1 - target_focus)

            n_target_ch = max(len(target_channels), 1)
            n_noise_ch = max(6 - n_target_ch, 1)

            intensities = {}
            for ch in range(6):
                if ch in target_channels:
                    mean_int = target_budget / n_target_ch
                    intensity = rng.normal(mean_int, mean_int * 0.4)
                    intensity = float(np.clip(intensity, 5, 84))
                else:
                    mean_int = noise_budget / n_noise_ch
                    intensity = rng.normal(mean_int, mean_int * 0.5)
                    intensity = float(np.clip(intensity, 3, 70))
                intensities[ch] = int(round(intensity))

            if rng.random() < 0.25 and n_noise_ch > 0:
                noise_chs = [c for c in range(6) if c not in target_channels]
                if noise_chs:
                    boost_ch = rng.choice(noise_chs)
                    intensities[boost_ch] = int(np.clip(
                        intensities[boost_ch] * rng.uniform(1.5, 3.0), 10, 84
                    ))

            if rng.random() < 0.2 and n_target_ch > 1:
                reduce_ch = rng.choice(list(target_channels))
                intensities[reduce_ch] = int(np.clip(
                    intensities[reduce_ch] * rng.uniform(0.2, 0.5), 3, 84
                ))

            comp = pd.DataFrame([
                {"channel_index": ch, "intensity": intensities[ch]}
                for ch in range(6)
            ])
            user_vec = recipe_to_user_vector(
                comp, aroma_notes_map, channel_to_aromas=channel_to_aromas,
            )
            if user_vec:
                # Target is the OTHER SKU from the same group
                results.append((user_vec, int(target)))

    rng_py = random.Random(seed)
    rng_py.shuffle(results)
    return results


def generate_noisy_sessions(
    perfume_vectors: dict[int, np.ndarray],
    note_to_idx: dict[str, int],
    aroma_notes_map: pd.DataFrame,
    n_per_sku: int = 5,
    seed: int = 42,
) -> list[tuple[dict[str, float], int]]:
    """Generate synthetic (user_vec, target_pid) pairs matching real noise patterns.

    Key insight from real data analysis:
    - ALL 12 organ notes are ALWAYS active (all channels have nonzero intensity)
    - Signal is in RELATIVE INTENSITIES, not binary presence/absence
    - target_focus ~ N(0.57, 0.22): only 57% of energy goes to target notes
    - Real channel intensities range 5-84 with mean ~29

    Strategy: generate at CHANNEL level (not note level) to match the organ architecture.
    """
    from .profile.build_profile import recipe_to_user_vector

    rng = np.random.default_rng(seed)

    organ_idxs = {n: note_to_idx[n] for n in ORGAN_NOTES if n in note_to_idx}
    channel_to_aromas = {}
    for ch, pairs in CHANNEL_MAP.items():
        channel_to_aromas[ch] = [aid for aid, _ in pairs]

    results = []
    for pid, vec in perfume_vectors.items():
        # Target's organ notes (binary)
        active_notes = {note for note, idx in organ_idxs.items() if vec[idx] > 0}
        if not active_notes:
            continue

        # Which channels cover target notes
        target_channels = set()
        for note in active_notes:
            if note in NOTE_TO_CHANNEL:
                target_channels.add(NOTE_TO_CHANNEL[note])

        for _ in range(n_per_sku):
            # Draw target_focus from real distribution
            target_focus = rng.normal(0.57, 0.22)
            target_focus = float(np.clip(target_focus, 0.15, 0.95))

            # Generate channel intensities (ALL nonzero, matching real data)
            intensities = {}

            # Total intensity budget (sum of all channels, real mean ~175)
            total_budget = rng.normal(175, 50)
            total_budget = float(np.clip(total_budget, 60, 350))

            # Split budget: target_focus to target channels, rest to others
            target_budget = total_budget * target_focus
            noise_budget = total_budget * (1 - target_focus)

            n_target_ch = max(len(target_channels), 1)
            n_noise_ch = max(6 - n_target_ch, 1)

            for ch in range(6):
                if ch in target_channels:
                    # Target channel: higher intensity with variance
                    mean_int = target_budget / n_target_ch
                    intensity = rng.normal(mean_int, mean_int * 0.4)
                    intensity = float(np.clip(intensity, 5, 84))
                else:
                    # Non-target channel: lower but nonzero
                    mean_int = noise_budget / n_noise_ch
                    intensity = rng.normal(mean_int, mean_int * 0.5)
                    intensity = float(np.clip(intensity, 3, 70))

                intensities[ch] = int(round(intensity))

            # Random perturbation: occasionally boost a non-target channel
            # (simulates user exploration/curiosity)
            if rng.random() < 0.25 and n_noise_ch > 0:
                noise_chs = [c for c in range(6) if c not in target_channels]
                if noise_chs:
                    boost_ch = rng.choice(noise_chs)
                    intensities[boost_ch] = int(np.clip(
                        intensities[boost_ch] * rng.uniform(1.5, 3.0), 10, 84
                    ))

            # Occasionally reduce a target channel (simulates missed/underexpressed pref)
            if rng.random() < 0.2 and n_target_ch > 1:
                reduce_ch = rng.choice(list(target_channels))
                intensities[reduce_ch] = int(np.clip(
                    intensities[reduce_ch] * rng.uniform(0.2, 0.5), 3, 84
                ))

            # Build user vector through real pipeline
            comp = pd.DataFrame([
                {"channel_index": ch, "intensity": intensities[ch]}
                for ch in range(6)
            ])
            user_vec = recipe_to_user_vector(
                comp, aroma_notes_map,
                channel_to_aromas=channel_to_aromas,
            )
            if user_vec:
                results.append((user_vec, pid))

    rng_py = random.Random(seed)
    rng_py.shuffle(results)
    return results


def build_collision_groups(
    perfume_vectors: dict[int, np.ndarray],
    note_to_idx: dict[str, int],
) -> dict[tuple, list[int]]:
    organ_idxs = [note_to_idx[n] for n in ORGAN_NOTES if n in note_to_idx]
    groups = {}
    for pid, vec in perfume_vectors.items():
        fp = tuple(1 if vec[i] > 0 else 0 for i in organ_idxs)
        groups.setdefault(fp, []).append(pid)
    return groups
