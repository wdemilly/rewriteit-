"""run_state_v1.py

Persistent run-state layer for simpleapp_v41 (and forward).

Motivation. simpleapp_v40 holds every draft, verify_result, score, and
intermediate state in Python locals for the duration of run_pipeline().
A streamlit websocket disconnect, an uncaught exception in the
scoring path, an OOM, or a browser refresh during a long run discards
every draft already paid for. This module fixes that: each draft is
written to a per-run directory the moment it generates, and a
manifest.json is updated incrementally at every state transition, so a
crash leaves the work recoverable on disk.

Directory layout per run:

  runs/<timestamp>_<short_id>/
    manifest.json                       # incremental state (overwritten atomically)
    outline.txt                         # input outline
    author_construction.txt             # step 2-3 output (when written)
    drafts/
      <draft_run_id>.txt                # draft prose
      <draft_run_id>.meta.json          # tokens, verify_result, scores
    winner/
      <draft_run_id>.txt                # winning chapter (only if we got one)
      summary.txt                       # plain-text report
    crash.txt                           # traceback (only if the pipeline crashed)

Public API:

  RunState.create(outline_text, config, model, app_dir=None) -> RunState
    Creates the run directory and initial manifest. Writes outline.txt.

  state.write_author_construction(text, usage)
  state.write_draft(draft_dict)                  # text + tokens + run_id
  state.write_verify_result(run_id, verify_result)
  state.write_scores(run_id, quality_verdict, quality_reason, quality_score,
                     ai_estimate, ai_estimate_details=None)
  state.record_inner_batch(outer_attempt, inner_attempt, n_generated, n_survivors,
                           n_rejected, draft_run_ids)
  state.write_winner(draft_dict)
  state.write_crash(exc_type, exc_message, traceback_text)
  state.mark_status(status: "running" | "complete" | "crashed" | "no_winner")
  state.finalize()                               # marks completed_at, status
  RunState.list_recent_runs(app_dir=None, n=10) -> list[dict]   # for UI
  RunState.load(run_dir: Path) -> RunState       # rehydrate from disk
"""
from __future__ import annotations
import json
import os
import shutil
import traceback as tb_mod
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


RUNS_SUBDIR = "runs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _atomic_write_text(path: Path, content: str) -> None:
    """Write content to path atomically (write tmp, fsync, rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


@dataclass
class RunState:
    run_dir: Path
    manifest: dict = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------
    @classmethod
    def create(
        cls,
        outline_text: str,
        config: dict,
        model: str,
        app_dir: Optional[Path] = None,
        label: Optional[str] = None,
        app_version: Optional[str] = None,
    ) -> "RunState":
        """Create a new run directory and initialize manifest.

        Directory naming:
          If both label and app_version are None: <timestamp>_<short_id>  (v41 default)
          With version + label:                   <app_version>_<label>_<timestamp>_<short_id>
          With version only:                      <app_version>_<timestamp>_<short_id>
          With label only:                        <label>_<timestamp>_<short_id>

        Matches the v36 pattern: APP_VERSION at the front so runs sort by
        version-then-time, and the outline label embedded so directories
        are scannable without opening the manifest."""
        if app_dir is None:
            app_dir = Path.cwd()
        runs_root = app_dir / RUNS_SUBDIR
        runs_root.mkdir(exist_ok=True)
        short = uuid.uuid4().hex[:6]
        ts = _now_iso()
        parts = []
        if app_version:
            parts.append(app_version)
        if label:
            parts.append(label)
        parts.append(ts)
        parts.append(short)
        run_dir_name = "_".join(parts)
        run_dir = runs_root / run_dir_name
        run_dir.mkdir()
        (run_dir / "drafts").mkdir()

        manifest = {
            "run_dir":              str(run_dir),
            "app_version":          app_version,
            "label":                label,
            "started_at":           ts,
            "completed_at":         None,
            "status":               "running",
            "model":                model,
            "config":               dict(config),
            "outline_chars":        len(outline_text),
            "author_construction_chars": 0,
            "author_usage":         None,
            "outer_batches":        [],
            "drafts":               {},  # draft_run_id -> draft summary
            "winner_run_id":        None,
            "crash":                None,
        }
        state = cls(run_dir=run_dir, manifest=manifest)
        _atomic_write_text(run_dir / "outline.txt", outline_text)
        state._save_manifest()
        return state

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        manifest_path = Path(run_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(run_dir=Path(run_dir), manifest=manifest)

    # -----------------------------------------------------------------
    # Listing for UI
    # -----------------------------------------------------------------
    @classmethod
    def list_recent_runs(
        cls,
        app_dir: Optional[Path] = None,
        n: int = 10,
    ) -> list[dict]:
        """Return up to n most recent run manifests (newest first)."""
        if app_dir is None:
            app_dir = Path.cwd()
        runs_root = app_dir / RUNS_SUBDIR
        if not runs_root.exists():
            return []
        out = []
        for d in sorted(runs_root.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            m_path = d / "manifest.json"
            if not m_path.exists():
                continue
            try:
                m = json.loads(m_path.read_text(encoding="utf-8"))
                out.append(m)
            except Exception:
                continue
            if len(out) >= n:
                break
        return out

    # -----------------------------------------------------------------
    # Writes
    # -----------------------------------------------------------------
    def _save_manifest(self) -> None:
        _atomic_write_json(self.run_dir / "manifest.json", self.manifest)

    def write_author_construction(self, text: str, usage: Optional[dict] = None) -> None:
        _atomic_write_text(self.run_dir / "author_construction.txt", text)
        self.manifest["author_construction_chars"] = len(text)
        if usage is not None:
            self.manifest["author_usage"] = usage
        self._save_manifest()

    def write_draft(self, draft: dict) -> None:
        """Persist a freshly-generated draft. Called BEFORE any verification.
        draft must include keys: run_id, text. Optional: input_tokens, output_tokens."""
        rid = draft["run_id"]
        text = draft.get("text", "")
        _atomic_write_text(self.run_dir / "drafts" / f"{rid}.txt", text)
        meta = {
            "run_id":        rid,
            "words":         len(text.split()),
            "input_tokens":  draft.get("input_tokens"),
            "output_tokens": draft.get("output_tokens"),
            "generated_at":  _now_iso(),
        }
        _atomic_write_json(self.run_dir / "drafts" / f"{rid}.meta.json", meta)
        self.manifest["drafts"][rid] = meta
        self._save_manifest()

    def write_draft_error(self, run_id: str, error: str) -> None:
        """Record that a draft generation failed."""
        meta = {
            "run_id":     run_id,
            "error":      error,
            "errored_at": _now_iso(),
        }
        _atomic_write_json(self.run_dir / "drafts" / f"{run_id}.meta.json", meta)
        self.manifest["drafts"][run_id] = meta
        self._save_manifest()

    def write_verify_result(self, run_id: str, verify_result: dict) -> None:
        """Attach verifier output to a draft's meta file. Updates manifest."""
        meta_path = self.run_dir / "drafts" / f"{run_id}.meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        meta["verify_result"] = verify_result
        _atomic_write_json(meta_path, meta)
        # Compact summary in manifest.
        summary = self.manifest["drafts"].setdefault(run_id, {})
        summary["verify_pass"] = bool(verify_result.get("pass"))
        summary["verify_counts"] = verify_result.get("counts", {})
        self._save_manifest()

    def write_scores(
        self,
        run_id: str,
        quality_verdict: Optional[str] = None,
        quality_reason:  Optional[str] = None,
        quality_score:   Optional[float] = None,
        ai_estimate:     Optional[float] = None,
        ai_estimate_details: Optional[dict] = None,
    ) -> None:
        meta_path = self.run_dir / "drafts" / f"{run_id}.meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        meta["quality_verdict"]     = quality_verdict
        meta["quality_reason"]      = quality_reason
        meta["quality_score"]       = quality_score
        meta["ai_estimate"]         = ai_estimate
        meta["ai_estimate_details"] = ai_estimate_details
        _atomic_write_json(meta_path, meta)
        summary = self.manifest["drafts"].setdefault(run_id, {})
        summary["quality_verdict"] = quality_verdict
        summary["quality_score"]   = quality_score
        summary["ai_estimate"]     = ai_estimate
        self._save_manifest()

    def record_inner_batch(
        self,
        outer_attempt: int,
        inner_attempt: int,
        n_generated: int,
        n_survivors: int,
        n_rejected: int,
        draft_run_ids: list[str],
    ) -> None:
        # Find or create outer-batch record.
        outers = self.manifest["outer_batches"]
        outer = next((o for o in outers if o["outer_attempt"] == outer_attempt), None)
        if outer is None:
            outer = {"outer_attempt": outer_attempt, "inner_batches": []}
            outers.append(outer)
        outer["inner_batches"].append({
            "inner_attempt":   inner_attempt,
            "n_generated":     n_generated,
            "n_survivors":     n_survivors,
            "n_rejected":      n_rejected,
            "draft_run_ids":   list(draft_run_ids),
            "recorded_at":     _now_iso(),
        })
        self._save_manifest()

    def write_diagnosis(self, outer_attempt: int, diagnosis: dict) -> None:
        """Persist a failure_diagnostic output for an outer batch that
        could not produce enough survivors. Written as a .json file in the
        run dir and indexed in the manifest. Safe to call even if Layer 2
        characterization was not run (diagnosis just contains aggregation)."""
        diag_dir = self.run_dir / "diagnoses"
        diag_dir.mkdir(exist_ok=True)
        path = diag_dir / f"outer_{outer_attempt}.json"
        _atomic_write_json(path, diagnosis)

        # Build a compact manifest summary so the UI can list diagnoses
        # without loading every file.
        agg = diagnosis.get("aggregation", {}) or {}
        per_cap = agg.get("per_cap", {}) or {}
        summary = {
            "outer_attempt":   outer_attempt,
            "n_drafts":        agg.get("n_drafts", 0),
            "caps_by_breadth": [
                {
                    "cap_id":           cap_id,
                    "rule_name":        per_cap.get(cap_id, {}).get("rule_name", ""),
                    "drafts_tripped":   per_cap.get(cap_id, {}).get("drafts_tripped", 0),
                    "total_hits":       per_cap.get(cap_id, {}).get("total_hits", 0),
                }
                for cap_id in agg.get("cap_order_by_breadth", [])
            ],
            "has_characterizations": bool(diagnosis.get("characterizations")),
            "path":            str(path),
        }
        self.manifest.setdefault("diagnoses", []).append(summary)
        self._save_manifest()

    def write_winner(self, winner: dict) -> None:
        winner_dir = self.run_dir / "winner"
        winner_dir.mkdir(exist_ok=True)
        rid = winner["run_id"]
        _atomic_write_text(winner_dir / f"{rid}.txt", winner.get("text", ""))
        summary_lines = [
            f"Winning draft: {rid}",
            f"Words: {len(winner.get('text','').split())}",
            f"Quality verdict: {winner.get('quality_verdict','?')}",
            f"Quality score:   {winner.get('quality_score','?')}",
            f"AI estimate:     {winner.get('ai_estimate','?')}",
            f"Quality reason:  {winner.get('quality_reason','')}",
        ]
        _atomic_write_text(winner_dir / "summary.txt", "\n".join(summary_lines))
        self.manifest["winner_run_id"] = rid
        self._save_manifest()

    def write_crash(self, exc_type: str, exc_message: str, traceback_text: str) -> None:
        crash_text = (
            f"Crashed at: {_now_iso()}\n"
            f"Exception type: {exc_type}\n"
            f"Message: {exc_message}\n\n"
            f"Traceback:\n{traceback_text}\n"
        )
        _atomic_write_text(self.run_dir / "crash.txt", crash_text)
        self.manifest["crash"] = {
            "type":           exc_type,
            "message":        exc_message,
            "traceback":      traceback_text,
            "crashed_at":     _now_iso(),
        }
        self.manifest["status"] = "crashed"
        self.manifest["completed_at"] = _now_iso()
        self._save_manifest()

    def mark_status(self, status: str) -> None:
        self.manifest["status"] = status
        self._save_manifest()

    def finalize(self) -> None:
        self.manifest["completed_at"] = _now_iso()
        if self.manifest.get("status") == "running":
            self.manifest["status"] = "complete" if self.manifest.get("winner_run_id") else "no_winner"
        self._save_manifest()

    # -----------------------------------------------------------------
    # Convenience helpers for the UI
    # -----------------------------------------------------------------
    def all_draft_paths(self) -> list[Path]:
        return sorted((self.run_dir / "drafts").glob("*.txt"))

    def winner_path(self) -> Optional[Path]:
        if not self.manifest.get("winner_run_id"):
            return None
        p = self.run_dir / "winner" / f"{self.manifest['winner_run_id']}.txt"
        return p if p.exists() else None

    def zip_archive(self) -> Path:
        """Zip the entire run directory and return the path. Useful for UI download."""
        archive_base = self.run_dir.parent / self.run_dir.name
        zip_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(self.run_dir))
        return Path(zip_path)


# ============================================================================
# Wrapper for exception handling at pipeline boundary
# ============================================================================
def capture_crash(state: Optional[RunState], exc: BaseException) -> None:
    """Best-effort crash capture. Safe to call even if state is None
    (no-op in that case). Never re-raises."""
    if state is None:
        return
    try:
        tb_text = "".join(tb_mod.format_exception(type(exc), exc, exc.__traceback__))
        state.write_crash(
            exc_type=type(exc).__name__,
            exc_message=str(exc),
            traceback_text=tb_text,
        )
    except Exception:
        pass
