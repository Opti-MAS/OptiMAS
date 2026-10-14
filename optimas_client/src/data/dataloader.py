"""
Universal DataLoader for OptiMAS Client

Supports loading various types of datasets, normalizing the schema,
and strictly splitting into train/val/test sets.
Provides epoch-based iteration over the training set. (``UniversalDataLoader``) and adaptive
probability-weighted sampling (``AdaptiveSampler``).
"""

import json
import random
import logging
import math
import os
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Reproducible train/val/test partitions — keep ``cli/train.py`` and ``cli/test.py`` aligned.
# Fixed seed governing train/val/test partitioning (see paper, Appendix A.5).
DEFAULT_DETERMINISTIC_SPLIT_SEED = 42
DEFAULT_TRAIN_VAL_TEST_RATIOS: Tuple[float, float, float] = (0.3, 0.1, 0.6)

WORKBENCH_DEFAULT_QUERY_TYPE = "workbench"


class UniversalDataLoader:
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        seed: int = 42,
        query_type_default: str = WORKBENCH_DEFAULT_QUERY_TYPE,
        train_ratio: float = 0.04,
        val_ratio: float = 0.01,
        test_ratio: float = 0.95,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
    ):
        """
        Initialize the UniversalDataLoader.

        Supports two modes:

        1. **Ratio split** (default): provide ``dataset_path`` and the loader
           will split the data into train/val/test by the given ratios.
        2. **Pre-split files**: provide one or more of ``train_path``,
           ``val_path``, ``test_path``.  Each points to an independent JSONL
           file that is loaded directly — no ratio splitting is performed.
           ``dataset_path`` is ignored when any pre-split path is given.

        Args:
            dataset_path: Path to a single JSONL dataset (ratio-split mode).
            seed: Random seed for reproducible splitting.
            query_type_default: Evaluator routing key when the row has no ``query_type``.
            train_ratio: Proportion of dataset used for training (ratio-split mode).
            val_ratio: Proportion of dataset used for validation (ratio-split mode).
            test_ratio: Proportion of dataset used for testing (ratio-split mode).
            train_path: Path to a pre-split JSONL file for the training set.
            val_path: Path to a pre-split JSONL file for the validation set.
            test_path: Path to a pre-split JSONL file for the test set.
        """
        self.dataset_path = dataset_path
        self.seed = seed
        self.query_type_default = query_type_default

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.all_data: List[Dict[str, Any]] = []
        self.train_set: List[Dict[str, Any]] = []
        self.val_set: List[Dict[str, Any]] = []
        self.test_set: List[Dict[str, Any]] = []

        self.train_indices: List[int] = []
        self.current_idx = 0
        self.current_epoch = 0
        self._epoch_boundary = 0

        if train_path or val_path or test_path:
            self._load_presplit(train_path, val_path, test_path)
        else:
            if dataset_path is None:
                raise ValueError(
                    "Either dataset_path or at least one of "
                    "train_path/val_path/test_path must be provided."
                )
            self._load_data()
            self.split()
        self._validate_partition_ids()

    def _validate_partition_ids(self) -> None:
        seen = set()
        for name, rows in (("train", self.train_set), ("val", self.val_set), ("test", self.test_set)):
            ids = [str(row["query_id"]) for row in rows]
            if len(ids) != len(set(ids)) or seen.intersection(ids):
                raise ValueError(f"Dataset {name} contains duplicate or overlapping query IDs.")
            seen.update(ids)

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load and normalize all items from a JSONL file."""
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    items.append(self._normalize_item(item, line_idx))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_idx + 1} in {path}: {e}") from e
        return items

    def _load_data(self):
        """Load dataset from file and normalize each item."""
        self.all_data = self._load_jsonl(self.dataset_path)

    def _load_presplit(
        self,
        train_path: Optional[str],
        val_path: Optional[str],
        test_path: Optional[str],
    ) -> None:
        """Load pre-split JSONL files directly into train/val/test sets."""
        if train_path:
            self.train_set = self._load_jsonl(train_path)
        if val_path:
            self.val_set = self._load_jsonl(val_path)
        if test_path:
            self.test_set = self._load_jsonl(test_path)
        self.all_data = self.train_set + self.val_set + self.test_set
        logger.info(
            "Pre-split dataset loaded: train=%d | val=%d | test=%d | total=%d",
            len(self.train_set), len(self.val_set), len(self.test_set),
            len(self.all_data),
        )

    def _normalize_item(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Normalize a JSONL row into a standard schema for the evaluation pipeline."""
        normalized = dict(item)

        if str(item.get("benchmark_type", "")).lower() == "workbench":
            normalized["query_type"] = WORKBENCH_DEFAULT_QUERY_TYPE
            normalized["benchmark_type"] = "workbench"

        if not normalized.get("query_id"):
            normalized["query_id"] = (
                item.get("_id")
                or item.get("instance_id")
                or item.get("id")
                or item.get("problem_id")
                or f"query_{idx}"
            )
        normalized["query_id"] = str(normalized["query_id"])

        if "query" not in normalized:
            normalized["query"] = item.get("question") or item.get("prompt") or ""

        qt = normalized.get("query_type")
        if qt is None or qt == "":
            normalized["query_type"] = self.query_type_default

        if normalized.get("query_type") == WORKBENCH_DEFAULT_QUERY_TYPE:
            if "answer" not in normalized and item.get("ground_truth_actions") is not None:
                normalized["answer"] = item["ground_truth_actions"]
        elif "answer" not in normalized:
            normalized["answer"] = (
                item.get("expected_output")
                or item.get("answer")
                or item.get("completion")
                or ""
            )

        if normalized.get("query_type") == WORKBENCH_DEFAULT_QUERY_TYPE:
            from ..workbench.action_format import parse_ground_truth_actions

            normalized["answer"] = parse_ground_truth_actions(normalized.get("answer"))
            if "domains" not in normalized and item.get("domains") is not None:
                normalized["domains"] = item["domains"]

        return normalized

    def split(self):
        """Deterministically split the data into train, val, and test sets."""
        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(not math.isfinite(r) or r < 0 for r in ratios) or sum(ratios) <= 0:
            raise ValueError("Split ratios must be finite, non-negative, and have a positive sum.")
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            logger.warning("Split ratios do not sum to 1.0, normalizing them.")
            total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
            self.train_ratio /= total_ratio
            self.val_ratio /= total_ratio
            self.test_ratio /= total_ratio

        rng = random.Random(self.seed)

        indices = list(range(len(self.all_data)))
        rng.shuffle(indices)

        total = len(self.all_data)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        if train_end == 0 and total >= 1 and self.train_ratio > 0:
            train_end = 1
            val_end = min(total, train_end + int(total * self.val_ratio))
        if val_end == train_end and total >= train_end + 1 and self.val_ratio > 0:
            val_end = train_end + 1

        self.train_set = [self.all_data[i] for i in indices[:train_end]]
        self.val_set = [self.all_data[i] for i in indices[train_end:val_end]]
        self.test_set = [self.all_data[i] for i in indices[val_end:]]

        logger.info(
            f"Dataset split completed. Total={total} | Train={len(self.train_set)} "
            f"| Val={len(self.val_set)} | Test={len(self.test_set)}"
        )

    def reset_epoch(self):
        """Reset the training epoch, shuffle train indices for the next epoch."""
        self.train_indices = list(range(len(self.train_set)))

        rng = random.Random(self.seed + self.current_epoch)
        rng.shuffle(self.train_indices)

        self.current_idx = 0
        self._epoch_boundary = len(self.train_indices)
        self.current_epoch += 1
        logger.info(f"Epoch {self.current_epoch} started with {len(self.train_set)} training samples.")

    def _extend_indices_if_needed(self, need: int):
        """Ensure train_indices has at least `need` items from current_idx onward.

        When remaining items are insufficient, generates the next epoch's
        shuffled indices and appends them so the final batch is always full.
        """
        while len(self.train_indices) - self.current_idx < need:
            next_indices = list(range(len(self.train_set)))
            rng = random.Random(self.seed + self.current_epoch)
            rng.shuffle(next_indices)
            self.train_indices.extend(next_indices)
            self.current_epoch += 1

    def has_next_batch(self) -> bool:
        return self.current_idx < self._epoch_boundary

    def get_next_train_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        if not self.has_next_batch():
            raise StopIteration("No more batches in the current epoch. Call reset_epoch().")

        self._extend_indices_if_needed(batch_size)
        batch_indices = self.train_indices[self.current_idx : self.current_idx + batch_size]
        self.current_idx += batch_size

        return [self.train_set[i] for i in batch_indices]

    def fast_forward_to_step(self, target_step: int, batch_size: int, max_epochs: int) -> int:
        """Replay training data consumption to restore dataloader state.

        Simulates the exact sequence of ``reset_epoch()`` and
        ``get_next_train_batch()`` calls that the training loop would have
        executed up to and including *target_step*.  After this call the
        dataloader is positioned so that the **next** ``get_next_train_batch``
        returns the batch for ``target_step + 1``.

        Returns the epoch number (0-based) that *target_step* belongs to.
        """
        step_counter = 0
        for epoch in range(max_epochs):
            self.reset_epoch()
            while self.has_next_batch():
                step_counter += 1
                self.get_next_train_batch(batch_size)
                if step_counter == target_step:
                    return epoch
        raise ValueError(
            f"Could not reach step {target_step} within {max_epochs} epochs "
            f"(last step reached: {step_counter})"
        )

    def get_validation_set(self) -> List[Dict[str, Any]]:
        return self.val_set

    def get_test_set(self) -> List[Dict[str, Any]]:
        return self.test_set


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Probability-Weighted Sampler
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_RETRY_RATIO = 0.5
_DEFAULT_BOOST_FACTOR = 1.2
_DEFAULT_DECAY_FACTOR = 0.6
_DEFAULT_W_FLOOR = 0.2
_DEFAULT_W_CEILING = 2.0

SAMPLER_STATE_FILENAME = "adaptive_sampler_state.json"

class AdaptiveSampler:
    """Probability-weighted sampler with retry-pool for failed queries.

    Wraps a ``UniversalDataLoader`` and replaces its epoch-based sequential
    iteration with an adaptive two-pool strategy:

    * **Retry pool** — queries that failed in the previous step are given
      priority slots so the optimizer can immediately verify its fixes.
    * **Fresh pool** — remaining slots are filled by weighted sampling from
      the full train set (excluding retry-pool picks), so coverage is
      maintained and overfitting is bounded.

    Weights are updated multiplicatively after each step and clamped to
    ``[w_floor, w_ceiling]``.  A periodic reset restores all weights to 1.0
    to prevent long-term drift.
    """

    def __init__(
        self,
        dataloader: UniversalDataLoader,
        batch_size: int,
        *,
        retry_ratio: float = _DEFAULT_RETRY_RATIO,
        boost_factor: float = _DEFAULT_BOOST_FACTOR,
        decay_factor: float = _DEFAULT_DECAY_FACTOR,
        w_floor: float = _DEFAULT_W_FLOOR,
        w_ceiling: float = _DEFAULT_W_CEILING,
        reset_interval: Optional[int] = None,
        seed: int = 42,
    ):
        if batch_size < 1 or not 0 <= retry_ratio <= 1:
            raise ValueError("batch_size must be positive and retry_ratio must be in [0, 1].")
        if not 0 < decay_factor <= 1 or boost_factor < 1 or not 0 < w_floor <= 1 <= w_ceiling:
            raise ValueError("Invalid adaptive sampling weight bounds or multipliers.")
        if reset_interval is not None and reset_interval < 1:
            raise ValueError("reset_interval must be positive.")
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.retry_ratio = retry_ratio
        self.boost_factor = boost_factor
        self.decay_factor = decay_factor
        self.w_floor = w_floor
        self.w_ceiling = w_ceiling
        self.seed = seed

        self._train_set = dataloader.train_set
        n = len(self._train_set)
        if n == 0:
            raise ValueError("Train set is empty — cannot create AdaptiveSampler")

        if reset_interval is not None:
            self.reset_interval = reset_interval
        else:
            self.reset_interval = 2 * math.ceil(n / batch_size)

        self._id_to_idx: Dict[str, int] = {}
        for i, q in enumerate(self._train_set):
            qid = str(q.get("query_id", f"idx-{i}"))
            self._id_to_idx[qid] = i

        self.weights: List[float] = [1.0] * n
        self.sample_counts: List[int] = [0] * n
        self.prev_failed_indices: List[int] = []
        self.steps_since_reset: int = 0
        self.total_steps: int = 0

        self._rng = random.Random(seed)

    @property
    def total_train_steps(self) -> int:
        """How many steps have been executed so far."""
        return self.total_steps

    def _clamp(self, w: float) -> float:
        return max(self.w_floor, min(self.w_ceiling, w))

    def _weighted_sample_without_replacement(
        self, indices: List[int], k: int
    ) -> List[int]:
        """Sample *k* indices from *indices* using self.weights, no replacement."""
        if k <= 0:
            return []
        if k >= len(indices):
            return list(indices)

        pool = list(indices)
        chosen: List[int] = []
        for _ in range(k):
            ws = [self.weights[i] for i in pool]
            total = sum(ws)
            if total <= 0:
                chosen.extend(self._rng.sample(pool, k - len(chosen)))
                break
            probs = [w / total for w in ws]
            r = self._rng.random()
            cumulative = 0.0
            pick_pos = len(pool) - 1
            for j, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    pick_pos = j
                    break
            chosen.append(pool[pick_pos])
            pool.pop(pick_pos)
        return chosen

    def get_batch(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Sample the next training batch.

        Returns:
            (batch, resampled_ids) where *resampled_ids* lists the
            ``query_id`` values of queries that were in the retry pool
            (failed in the previous step).
        """
        if self.steps_since_reset >= self.reset_interval:
            self._periodic_reset()

        n = len(self._train_set)
        B = min(self.batch_size, n)

        retry_budget = int(math.floor(B * self.retry_ratio))

        # Pool A: retry previous failures
        retry_picks: List[int] = []
        if self.prev_failed_indices:
            valid_failed = [i for i in self.prev_failed_indices if 0 <= i < n]
            if len(valid_failed) <= retry_budget:
                retry_picks = list(valid_failed)
            else:
                retry_picks = self._weighted_sample_without_replacement(
                    valid_failed, retry_budget
                )

        fresh_budget = B - len(retry_picks)

        # Pool B: remaining queries weighted-sampled
        retry_set = set(retry_picks)
        remaining_indices = [i for i in range(n) if i not in retry_set]
        fresh_picks = self._weighted_sample_without_replacement(
            remaining_indices, fresh_budget
        )

        batch_indices = retry_picks + fresh_picks
        self._rng.shuffle(batch_indices)

        for i in batch_indices:
            self.sample_counts[i] += 1
        self.steps_since_reset += 1
        self.total_steps += 1

        batch = [self._train_set[i] for i in batch_indices]
        resampled_ids = [
            str(self._train_set[i].get("query_id", f"idx-{i}"))
            for i in retry_picks
        ]

        return batch, resampled_ids

    def update_weights(self, results: List[Tuple[str, bool]]) -> None:
        """Update sampling weights based on evaluation results.

        Args:
            results: list of ``(query_id, is_correct)`` for each query in the
                     batch that was just evaluated.
        """
        failed: List[int] = []
        for qid, correct in results:
            qid_s = str(qid)
            idx = self._id_to_idx.get(qid_s)
            if idx is None:
                logger.warning("AdaptiveSampler: unknown query_id %r in update", qid_s)
                continue
            if correct:
                self.weights[idx] = self._clamp(self.weights[idx] * self.decay_factor)
            else:
                self.weights[idx] = self._clamp(self.weights[idx] * self.boost_factor)
                failed.append(idx)

        self.prev_failed_indices = failed

    def _periodic_reset(self) -> None:
        """Reset all weights to 1.0 and clear the retry pool."""
        logger.info(
            "AdaptiveSampler: periodic reset after %d steps (interval=%d)",
            self.steps_since_reset, self.reset_interval,
        )
        self.weights = [1.0] * len(self._train_set)
        self.prev_failed_indices = []
        self.steps_since_reset = 0

    # ── Persistence ───────────────────────────────────────────────────────

    def save_state(self, directory: str) -> str:
        """Serialize sampler state to a JSON file.  Returns the file path."""
        path = os.path.join(directory, SAMPLER_STATE_FILENAME)
        state = {
            "weights": self.weights,
            "sample_counts": self.sample_counts,
            "prev_failed_indices": self.prev_failed_indices,
            "steps_since_reset": self.steps_since_reset,
            "total_steps": self.total_steps,
            "rng_state_seed": self.seed,
            "rng_state": self._rng.getstate(),
            "query_ids": [str(q["query_id"]) for q in self._train_set],
            "config": {
                "batch_size": self.batch_size,
                "retry_ratio": self.retry_ratio,
                "boost_factor": self.boost_factor,
                "decay_factor": self.decay_factor,
                "w_floor": self.w_floor,
                "w_ceiling": self.w_ceiling,
                "reset_interval": self.reset_interval,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        return path

    def load_state(self, directory: str) -> bool:
        """Restore sampler state from a previously saved file.

        Returns True if state was loaded, False if no state file found.
        """
        path = os.path.join(directory, SAMPLER_STATE_FILENAME)
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)

            n = len(self._train_set)
            expected_ids = [str(q["query_id"]) for q in self._train_set]
            if state.get("query_ids") != expected_ids or not state.get("rng_state"):
                raise ValueError("Sampler checkpoint does not match this dataset or lacks RNG state.")
            current_config = {k: getattr(self, k) for k in state.get("config", {})}
            if state.get("config") != current_config or state.get("rng_state_seed") != self.seed:
                raise ValueError("Sampler checkpoint parameters differ from this run.")
            weights = state.get("weights", [])
            if len(weights) == n:
                self.weights = [float(w) for w in weights]
            else:
                logger.warning(
                    "AdaptiveSampler: state weights length %d != train_set %d, ignoring",
                    len(weights), n,
                )

            sc = state.get("sample_counts", [])
            if len(sc) == n:
                self.sample_counts = [int(c) for c in sc]

            self.prev_failed_indices = [
                int(i) for i in state.get("prev_failed_indices", []) if 0 <= int(i) < n
            ]
            self.steps_since_reset = int(state.get("steps_since_reset", 0))
            self.total_steps = int(state.get("total_steps", 0))

            rng_state = state["rng_state"]
            self._rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))

            logger.info(
                "AdaptiveSampler: restored state — total_steps=%d, steps_since_reset=%d, "
                "prev_failed=%d",
                self.total_steps, self.steps_since_reset, len(self.prev_failed_indices),
            )
            return True
        except Exception as e:
            logger.warning("AdaptiveSampler: failed to load state from %s: %s", path, e)
            return False
