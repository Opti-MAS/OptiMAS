"""
Optimizer Agent for MAS backward pass.

MAS structural checks and YAML syntax scans run inside ``finalize_optimization``
(not exposed as separate tools).
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml

from ..agent.base_react_agent import BaseAgent
from ..agent.utils.tool_decorator import mcp_tool
from ..config.loader import ConfigLoader
from ..config.mas_config_loader import get_entry_agent_name_from_config, validate_mas_config
from .task_prompt import build_backward_pass_task_prompt, trajectory_json_attachment_note
from .batch_trajectory_report import build_batch_report_from_train_performance, compact_eval_result_row

logger = logging.getLogger(__name__)

class OptimizerAgent(BaseAgent):
    """
    Optimizer Agent for MAS self-evolution.
    Acts as the optimizer in the backward pass of MAS training.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimization_summary = ""
        # workspace_base_dir lives on global_config (e.g. global_config.yaml), not on the optimizer agent slice.
        self.workspace_base_dir = (
            self.config.get("workspace_base_dir")
            or (self.global_config or {}).get("workspace_base_dir")
            or "./workspace"
        )

    def _backward_pass_template_path(self) -> str:
        """Absolute path to the backward-pass Markdown template.

        Resolved from ``prompt_dict['backward_pass']`` (set in optimizer YAML),
        else ``<optimizer_config_dir>/prompts/backward_pass.md``.
        """
        if self.prompt_dict and self.prompt_dict.get("backward_pass"):
            return self.prompt_dict["backward_pass"]
        config_dir = self.config.get("_config_dir") or self.global_config.get("_config_dir") or "."
        return os.path.normpath(os.path.join(config_dir, "prompts", "backward_pass.md"))

    # Truncation limits for fields written to the optimizer's trajectory JSON.
    # Grader reasoning is the most diagnostic signal — keep it longer.
    _MAX_PREDICTION_LEN = 800
    _MAX_GRADER_LEN = 1200
    _MAX_REASONING_LEN = 600

    async def step(
        self,
        optimization_inputs: Dict[str, Any],
        current_mas_config: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Execute one optimization step (backward pass).

        Args:
            optimization_inputs: Dictionary containing:
                - "train_performance": ``batch_report`` (per-query run, eval, gold, agent metrics,
                  and ``trajectory_artifacts_relpath``). If ``batch_report`` is missing, it is
                  rebuilt from legacy keys ``batch``, ``results``, ``turn_counts``, and ``stats``.
                - "validation_signal": { "current_acc", "acc_diff" }
                - "previous_optimization": str
                - "step_dir": str
            current_mas_config: Current MAS configuration state.

        Returns:
            Tuple of (new_mas_config, optimization_summary)
        """
        self.optimization_summary = ""
        self.is_task_complete = False

        train_perf = optimization_inputs.get("train_performance", {})

        # ── 1. Build the backward-pass prompt ──────────────────────────────
        prompt_path = self._backward_pass_template_path()
        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(
                f"Backward pass template not found: {prompt_path}. "
                "Set agents.optimizer.prompt_dict.backward_pass in the optimizer YAML."
            )
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read().strip()

        batch = train_perf.get("batch", [])

        # ── 2. Write trajectory JSON for the agent to read ─────────────────
        step_dir = optimization_inputs.get("step_dir")
        if step_dir:
            workspace_dir = step_dir
        else:
            workspace_dir = os.path.join(
                self.workspace_base_dir,
                getattr(self.mcp_client, "workspace_path", ""),
            )
            os.makedirs(workspace_dir, exist_ok=True)

        workspace_dir = os.path.abspath(workspace_dir)
        prompt_inputs = dict(optimization_inputs)
        if not prompt_inputs.get("step_dir"):
            prompt_inputs["step_dir"] = workspace_dir

        task_prompt = build_backward_pass_task_prompt(
            template,
            prompt_inputs,
        )

        results = train_perf.get("results", [])
        turn_counts = train_perf.get("turn_counts", [])
        train_stats = train_perf.get("stats", {})
        # Prefer a pre-merged v2 report from the training loop; else rebuild from available keys.
        traj_payload: Any = train_perf.get("batch_report")
        if not traj_payload:
            ean = (
                get_entry_agent_name_from_config(current_mas_config)
                if isinstance(current_mas_config, dict)
                else "entry"
            )
            traj_payload = build_batch_report_from_train_performance(
                train_perf,
                entry_agent_name=ean,
            )
        if not traj_payload:
            # Legacy 3-block shape (no batch_report and insufficient keys to build v2).
            results_compact = []
            for i, r in enumerate(results):
                msg_count = turn_counts[i] if i < len(turn_counts) else None
                results_compact.append(
                    compact_eval_result_row(
                        r,
                        max_prediction_len=self._MAX_PREDICTION_LEN,
                        max_grader_len=self._MAX_GRADER_LEN,
                        max_reasoning_len=self._MAX_REASONING_LEN,
                        mas_conversation_message_count=msg_count,
                        optimizer_facing=True,
                    )
                )
            traj_payload = {
                "aggregated_stats": train_stats,
                "results": results_compact,
                "batch": batch,
                "_note": (
                    "Full trajectories are not stored here to save context. "
                    "Per-sample worker logs are under working_trajectory/ in each sample workspace."
                ),
            }

        traj_file_path = os.path.join(workspace_dir, "current_batch_trajectories.json")
        try:
            with open(traj_file_path, "w", encoding="utf-8") as f:
                json.dump(traj_payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to write current_batch_trajectories: %s", e)

        task_prompt += trajectory_json_attachment_note(prompt_inputs)

        # ── 3. Execute Task ────────────────────────────────────────────────
        result = await self.execute_task(task=task_prompt)
        if not isinstance(result, dict) or not result.get("success") or not self.optimization_summary:
            raise RuntimeError(f"Optimization did not pass finalization: {(result or {}).get('error', 'incomplete step')}")

        optimization_summary = self.optimization_summary
        if not optimization_summary:
            if isinstance(result, dict) and "data" in result:
                optimization_summary = str(result["data"])
            else:
                optimization_summary = "No optimization summary provided."

        # ── 4. Reload the (potentially modified) MAS config ────────────────
        if step_dir:
            mas_config_dir = os.path.join(step_dir, "mas_config")
        else:
            mas_config_dir = os.path.join(workspace_dir, "mas", "current", "config")

        try:
            new_mas_config = ConfigLoader.load_config(mas_config_dir)
        except Exception as e:
            raise RuntimeError(f"Cannot reload finalized MAS configuration: {e}") from e

        return new_mas_config, optimization_summary

    @staticmethod
    def _trunc(s: str, max_len: int) -> str:
        if not s or not isinstance(s, str):
            return s or ""
        return s if len(s) <= max_len else s[:max_len] + "..."

    def _validate_mas_config_dir(self, config_path: str) -> Dict[str, Any]:
        """
        Structural MAS validation for a config directory (config.yaml + validate_mas_config).
        Returns the structural validation result without exposing a separate tool.
        """
        abs_path = self._resolve_workspace_path(config_path)
        config_yaml = os.path.join(abs_path, "config.yaml")

        if not os.path.isdir(abs_path):
            return {"success": False, "data": None, "error": f"MAS config directory not found: {abs_path}"}

        if not os.path.isfile(config_yaml):
            return {"success": False, "data": None, "error": f"config.yaml not found under {abs_path}"}

        try:
            with open(config_yaml, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return {"success": False, "data": None, "error": f"YAML syntax error in config.yaml: {e}"}

        if not isinstance(config, dict):
            return {"success": False, "data": None, "error": "config.yaml did not parse to a dict"}

        result = validate_mas_config(config)
        if result["valid"]:
            return {
                "success": True,
                "data": {
                    "message": "MAS config structure is valid.",
                    "agent_graph": result["agent_graph"],
                },
                "error": None,
            }
        return {
            "success": False,
            "data": {"agent_graph": result.get("agent_graph", {})},
            "error": "MAS structural validation failed: " + "; ".join(result["errors"]),
        }

    def _validate_all_yaml_under_dir(self, root_abs: str) -> Dict[str, Any]:
        """Parse every .yaml/.yml under root; return success + list of error lines."""
        if not os.path.isdir(root_abs):
            return {"success": False, "errors": [f"Not a directory: {root_abs}"]}
        errors: list[str] = []
        for dirpath, _, filenames in os.walk(root_abs):
            for name in sorted(filenames):
                if not (name.endswith(".yaml") or name.endswith(".yml")):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, root_abs)
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    errors.append(f"{rel}: {e}")
                except OSError as e:
                    errors.append(f"{rel}: cannot read ({e})")
        if errors:
            return {"success": False, "errors": errors}
        return {"success": True, "errors": []}

    # ── SKILL.md frontmatter pattern ──────────────────────────────────────
    _FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    _FM_NAME_RE = re.compile(r"^name:\s*(.+)$", re.MULTILINE)
    _FM_DESC_RE = re.compile(r"^description:\s*(.+)$", re.MULTILINE)

    def _validate_skills_and_prompts(self, mas_abs: str) -> Dict[str, Any]:
        """Validate skill files and prompt files referenced by every agent in config.yaml.

        Checks:
          1. Every skill referenced in system_skills / preload_skills / optional_skills
             has a SKILL.md at ``skills/<agent_id>/<skill_name>/SKILL.md``.
          2. Each SKILL.md has valid YAML frontmatter with ``name`` and ``description``.
          3. The frontmatter ``name`` matches the folder name.
          4. The SKILL.md body (after frontmatter) is non-empty.
          5. No orphan skill folders exist directly under ``skills/`` that should be
             under an agent subdirectory.
          6. Prompt files referenced by ``prompt_dict`` actually exist.
          7. ``agents_llm_config.yaml`` contains entries for every agent in ``config.yaml``.
          8. Every ``initial_message.md`` contains ``${task}`` placeholder.
          9. Each agent's ``skill_dir`` points to ``./skills/<agent_id>``, not another agent's.
         10. Agents with optional_skills have ``load_skill`` in available_tools.
         11. Agents with subagent list have ``assign_task_to_agent`` in available_tools.

        Returns dict with ``success`` bool and ``errors`` list.
        """
        config_yaml = os.path.join(mas_abs, "config.yaml")
        if not os.path.isfile(config_yaml):
            return {"success": False, "errors": ["config.yaml not found"]}
        try:
            with open(config_yaml, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            return {"success": False, "errors": [f"Cannot parse config.yaml: {e}"]}

        agents = config.get("agents", {})
        if not isinstance(agents, dict):
            return {"success": False, "errors": ["'agents' in config.yaml is not a dict"]}

        errors: List[str] = []

        # ── Collect all agent_ids ────────────────────────────────────────
        agent_ids = set(agents.keys())

        # ── Check agents_llm_config.yaml consistency ─────────────────────
        llm_yaml = os.path.join(mas_abs, "agents_llm_config.yaml")
        if os.path.isfile(llm_yaml):
            try:
                with open(llm_yaml, "r", encoding="utf-8") as f:
                    llm_config = yaml.safe_load(f) or {}
                llm_agents = set((llm_config.get("agents") or {}).keys())
                missing_in_llm = agent_ids - llm_agents
                extra_in_llm = llm_agents - agent_ids
                for a in sorted(missing_in_llm):
                    errors.append(
                        f"agents_llm_config.yaml: missing entry for agent '{a}' "
                        f"(defined in config.yaml). Copy a deployment block from an existing agent."
                    )
                for a in sorted(extra_in_llm):
                    errors.append(
                        f"agents_llm_config.yaml: contains agent '{a}' not defined in config.yaml. "
                        f"Remove it or add the agent to config.yaml."
                    )
            except Exception as e:
                errors.append(f"agents_llm_config.yaml: parse error: {e}")
        else:
            errors.append(
                "agents_llm_config.yaml not found. Each agent needs a deployment block in this file."
            )

        # ── Per-agent checks ─────────────────────────────────────────────
        for agent_id, agent_cfg in agents.items():
            if not isinstance(agent_cfg, dict):
                errors.append(f"Agent '{agent_id}': config is not a dict")
                continue

            # -- Prompt files --
            prompt_dict = agent_cfg.get("prompt_dict", {})
            if isinstance(prompt_dict, dict):
                for key, rel_path in prompt_dict.items():
                    if not rel_path:
                        continue
                    abs_prompt = os.path.realpath(os.path.join(mas_abs, rel_path))
                    if not Path(abs_prompt).is_relative_to(Path(mas_abs).resolve()):
                        errors.append(f"Agent '{agent_id}': prompt_dict.{key} must stay within mas_config/.")
                        continue
                    if not os.path.isfile(abs_prompt):
                        errors.append(
                            f"Agent '{agent_id}': prompt_dict.{key} points to "
                            f"'{rel_path}' which does not exist."
                        )
                    elif key == "initial_message":
                        try:
                            with open(abs_prompt, "r", encoding="utf-8") as _f:
                                _im_content = _f.read()
                            if "${task}" not in _im_content:
                                errors.append(
                                    f"Agent '{agent_id}': initial_message template at "
                                    f"'{rel_path}' does NOT contain the '${{task}}' placeholder. "
                                    f"The framework substitutes ONLY '${{task}}' with the parent "
                                    f"agent's task_content. Other placeholders like "
                                    f"'${{task_description}}', '${{todo_file_content}}', etc. are NOT "
                                    f"supported and will appear as raw text. Fix: the simplest "
                                    f"correct initial_message.md is just '${{task}}' on one line. "
                                    f"If you need structured prompting, the PARENT agent must "
                                    f"compose everything into task_content before calling "
                                    f"assign_task_to_agent."
                                )
                        except OSError:
                            pass

            # -- Skill directory --
            skill_dir_rel = agent_cfg.get("skill_dir", f"./skills/{agent_id}")
            skill_dir_abs = os.path.normpath(os.path.join(mas_abs, skill_dir_rel))
            if not Path(skill_dir_abs).resolve().is_relative_to(Path(mas_abs).resolve()):
                errors.append(f"Agent '{agent_id}': skill_dir must stay within mas_config/.")
                continue

            # Validate skill_dir points to the agent's own directory
            expected_skill_dir = os.path.normpath(os.path.join(mas_abs, "skills", agent_id))
            if skill_dir_abs != expected_skill_dir:
                # Check if it points to another agent's directory
                skills_root = os.path.normpath(os.path.join(mas_abs, "skills"))
                if skill_dir_abs.startswith(skills_root + os.sep):
                    rel_inside = os.path.relpath(skill_dir_abs, skills_root)
                    top_folder = rel_inside.split(os.sep)[0]
                    if top_folder in agent_ids and top_folder != agent_id:
                        errors.append(
                            f"Agent '{agent_id}': skill_dir is '{skill_dir_rel}' which points "
                            f"to agent '{top_folder}''s skill directory. Each agent MUST have "
                            f"its own independent skill_dir. Fix: set skill_dir to "
                            f"'./skills/{agent_id}' and create skills under that directory."
                        )
                    elif top_folder != agent_id:
                        errors.append(
                            f"Agent '{agent_id}': skill_dir is '{skill_dir_rel}' but the "
                            f"convention requires './skills/{agent_id}'. Fix: set skill_dir to "
                            f"'./skills/{agent_id}'."
                        )

            # Cross-check tool availability with config features
            optional_skills = agent_cfg.get("optional_skills", [])
            available_tools = agent_cfg.get("available_tools", [])
            subagent_list = agent_cfg.get("subagent", [])
            known_tools = getattr(self, "mcp_tool_names", set()) | {
                name for name in dir(BaseAgent)
                if getattr(getattr(BaseAgent, name), "_is_mcp_tool", False)
            }
            if getattr(self, "mcp_tool_names", None):
                unknown = set(available_tools) - known_tools
                if unknown:
                    errors.append(f"Agent '{agent_id}': unavailable tools {sorted(unknown)}.")

            if optional_skills and isinstance(optional_skills, list) and len(optional_skills) > 0:
                if isinstance(available_tools, list) and "load_skill" not in available_tools:
                    errors.append(
                        f"Agent '{agent_id}': has {len(optional_skills)} optional_skills but "
                        f"'load_skill' is not in available_tools. The agent cannot load optional "
                        f"skills without this tool. Fix: add 'load_skill' to available_tools."
                    )

            if subagent_list and isinstance(subagent_list, list) and len(subagent_list) > 0:
                if isinstance(available_tools, list) and "assign_task_to_agent" not in available_tools:
                    errors.append(
                        f"Agent '{agent_id}': has subagent list {subagent_list} but "
                        f"'assign_task_to_agent' is not in available_tools. The agent cannot "
                        f"delegate to sub-agents without this tool. Fix: add "
                        f"'assign_task_to_agent' to available_tools."
                    )

            all_skills: List[str] = []
            for category in ("system_skills", "preload_skills", "optional_skills"):
                skills_list = agent_cfg.get(category, [])
                if isinstance(skills_list, list):
                    all_skills.extend(skills_list)

            for skill_name in all_skills:
                if Path(skill_name).is_absolute() or ".." in Path(skill_name).parts:
                    errors.append(f"Agent '{agent_id}': invalid skill name '{skill_name}'.")
                    continue
                # Mirror load_skill() path resolution priority:
                #   1. <skill_dir>/<skill_name>/SKILL.md  (works for both plain and nested names)
                #   2. <skill_dir>/<parent>/sub_skills/<child>.md  (legacy, only when "/" present)
                if '/' in skill_name:
                    primary_path = os.path.join(skill_dir_abs, skill_name, "SKILL.md")
                    parent, sub = skill_name.split('/', 1)
                    legacy_path = os.path.join(skill_dir_abs, parent, "sub_skills", f"{sub}.md")

                    if os.path.isfile(primary_path):
                        self._validate_skill_md(primary_path, skill_name, agent_id, mas_abs, errors)
                    elif os.path.isfile(legacy_path):
                        pass  # legacy sub_skill format — accepted
                    else:
                        errors.append(
                            f"Agent '{agent_id}': skill '{skill_name}' not found. "
                            f"The loader looks for: "
                            f"(1) '{os.path.relpath(primary_path, mas_abs)}', "
                            f"(2) '{os.path.relpath(legacy_path, mas_abs)}'. "
                            f"Neither exists. Fix: use plain skill names without '/' "
                            f"(e.g., '{sub}' instead of '{skill_name}') and place the "
                            f"SKILL.md at 'skills/{agent_id}/{sub}/SKILL.md'."
                        )
                    continue

                skill_folder = os.path.join(skill_dir_abs, skill_name)
                skill_md = os.path.join(skill_folder, "SKILL.md")

                if not os.path.isdir(skill_folder):
                    # Check if the skill was mistakenly placed under skills/ root
                    root_skills = os.path.join(mas_abs, "skills")
                    wrong_loc = os.path.join(root_skills, skill_name)
                    if os.path.isdir(wrong_loc):
                        errors.append(
                            f"Agent '{agent_id}': skill '{skill_name}' is at "
                            f"'skills/{skill_name}/' but MUST be at "
                            f"'skills/{agent_id}/{skill_name}/'. "
                            f"Move the folder to the correct location."
                        )
                    else:
                        errors.append(
                            f"Agent '{agent_id}': skill folder not found at "
                            f"'{os.path.relpath(skill_folder, mas_abs)}'. "
                            f"Create 'skills/{agent_id}/{skill_name}/SKILL.md'."
                        )
                    continue

                if not os.path.isfile(skill_md):
                    errors.append(
                        f"Agent '{agent_id}': skill '{skill_name}' folder exists at "
                        f"'{os.path.relpath(skill_folder, mas_abs)}' but contains no SKILL.md."
                    )
                    continue

                # Validate SKILL.md content
                self._validate_skill_md(skill_md, skill_name, agent_id, mas_abs, errors)

        # ── Orphan skill folders at skills/ root ─────────────────────────
        root_skills_dir = os.path.join(mas_abs, "skills")
        if os.path.isdir(root_skills_dir):
            for entry in sorted(os.listdir(root_skills_dir)):
                entry_path = os.path.join(root_skills_dir, entry)
                if not os.path.isdir(entry_path):
                    continue
                if entry in agent_ids:
                    continue
                # This is a skill folder placed directly under skills/ instead of skills/<agent_id>/
                has_skill_md = os.path.isfile(os.path.join(entry_path, "SKILL.md"))
                if has_skill_md:
                    errors.append(
                        f"Orphan skill folder 'skills/{entry}/' contains SKILL.md but is not "
                        f"under any agent's skill directory. Skills MUST be placed at "
                        f"'skills/<agent_id>/{entry}/SKILL.md'. Move it under the correct agent."
                    )

        return {"success": len(errors) == 0, "errors": errors}

    def _validate_skill_md(
        self,
        skill_md_path: str,
        skill_name: str,
        agent_id: str,
        mas_abs: str,
        errors: List[str],
    ) -> None:
        """Validate a single SKILL.md file for correct format."""
        if not Path(skill_md_path).resolve().is_relative_to(Path(mas_abs).resolve()):
            errors.append(f"Agent '{agent_id}': skill '{skill_name}' must stay within mas_config/.")
            return
        try:
            with open(skill_md_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            errors.append(
                f"Agent '{agent_id}': skill '{skill_name}' SKILL.md cannot be read: {e}"
            )
            return

        if not content.strip():
            errors.append(
                f"Agent '{agent_id}': skill '{skill_name}' SKILL.md is empty."
            )
            return

        fm_match = self._FRONTMATTER_RE.match(content)
        if not fm_match:
            errors.append(
                f"Agent '{agent_id}': skill '{skill_name}' SKILL.md is missing YAML frontmatter. "
                f"SKILL.md MUST start with:\n"
                f"---\n"
                f"name: {skill_name}\n"
                f"description: <one-line description of what this skill does>\n"
                f"---\n"
                f"Followed by the skill body content."
            )
            return

        fm_text = fm_match.group(1)
        name_match = self._FM_NAME_RE.search(fm_text)
        desc_match = self._FM_DESC_RE.search(fm_text)

        if not name_match:
            errors.append(
                f"Agent '{agent_id}': skill '{skill_name}' SKILL.md frontmatter is missing "
                f"the 'name' field. Add 'name: {skill_name}' between the --- delimiters."
            )
        elif name_match.group(1).strip() != skill_name:
            errors.append(
                f"Agent '{agent_id}': skill '{skill_name}' SKILL.md frontmatter name "
                f"is '{name_match.group(1).strip()}' but the folder name is '{skill_name}'. "
                f"They MUST match exactly."
            )

        if not desc_match:
            errors.append(
                f"Agent '{agent_id}': skill '{skill_name}' SKILL.md frontmatter is missing "
                f"the 'description' field. Add a one-line description between the --- delimiters."
            )
        elif not desc_match.group(1).strip():
            errors.append(
                f"Agent '{agent_id}': skill '{skill_name}' SKILL.md has an empty description. "
                f"Provide a meaningful one-line description so the agent knows when to load it."
            )

        # Body after frontmatter should be non-empty
        body = content[fm_match.end():].strip()
        if not body:
            errors.append(
                f"Agent '{agent_id}': skill '{skill_name}' SKILL.md has valid frontmatter but "
                f"the body is empty. Add the skill's instructions after the --- delimiter."
            )

    # ── TODO completeness check ────────────────────────────────────────
    _TODO_ITEM_RE = re.compile(
        r"^[-*]\s+\[(?P<marker>[^\]]{0,2})\]\s*(?P<text>.*)$"
    )

    def _validate_todo_completeness(self) -> Dict[str, Any]:
        """Check that the optimizer's todo file has no unresolved items.

        Valid markers:
          - ``[x]`` — completed
          - ``[~]`` — dropped / deferred (reason expected on same line)
        Invalid (blocks finalization):
          - ``[ ]`` — not started
          - ``[r]`` — still in progress
          - any other marker that is not ``x`` or ``~``

        Returns ``{"success": True, ...}`` when all items are resolved, or
        ``{"success": False, "errors": [...]}`` listing every offending line.
        """
        todo_rel = self.config.get("todo_file_path") or "optimizer/todo.md"
        todo_abs = self._resolve_workspace_path(todo_rel)

        if not os.path.isfile(todo_abs):
            return {"success": False, "errors": ["Create the step's TODO file and resolve its plan before finalizing."], "todo_path": todo_rel}

        try:
            with open(todo_abs, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as e:
            return {
                "success": False,
                "errors": [f"Cannot read {todo_rel}: {e}"],
                "todo_path": todo_rel,
            }

        unresolved: List[str] = []
        item_count = 0
        for lineno, raw_line in enumerate(lines, start=1):
            line = raw_line.rstrip("\n")
            m = self._TODO_ITEM_RE.match(line.strip())
            if not m:
                continue
            item_count += 1
            marker = m.group("marker").strip().lower()
            text = m.group("text").strip()
            if marker in ("x", "~"):
                if marker == "~" and not re.search(r"\breason\s*:\s*\S", text, re.IGNORECASE):
                    unresolved.append(
                        f"  L{lineno}: `[~]` item has no reason — add why it was dropped: {line.strip()}"
                    )
                continue
            unresolved.append(f"  L{lineno}: unresolved `[{m.group('marker')}]` — {line.strip()}")

        if not item_count:
            unresolved.append("TODO file must contain at least one completed or justified dropped plan item.")

        if unresolved:
            return {
                "success": False,
                "errors": unresolved,
                "todo_path": todo_rel,
            }
        return {"success": True, "errors": [], "todo_path": todo_rel}

    def _resolve_workspace_path(self, path: str) -> str:
        """Resolve a path within the optimizer's current step workspace."""
        root = self.workspace_base_dir
        mcp_ws = getattr(self.mcp_client, "workspace_path", None) or ""
        base = Path(root, mcp_ws).resolve()
        resolved = (base / path).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError("Optimizer paths must stay within the current step workspace.")
        return str(resolved)

    @mcp_tool(input_schema={"task_summary": {"type": "string", "description": "Optimization summary; runs the same mandatory checks as finalize_optimization."}})
    def task_done(self, task_summary: str) -> Dict[str, Any]:
        """Finalize this optimization step after all mandatory checks pass."""
        return self.finalize_optimization(task_summary)

    @mcp_tool(
        input_schema={
            "summary": {
                "type": "string",
                "description": (
                    "A detailed summary of the optimization changes made in this step. "
                    "This tool runs full MAS validation: YAML syntax, structural rules, "
                    "skill file placement and format, prompt file existence, "
                    "agents_llm_config.yaml consistency, and TODO file completeness "
                    "(every item must be [x] done or [~] dropped with reason — no "
                    "orphaned [ ] or [r] items allowed). On failure it returns errors "
                    "so you can fix and call finalize_optimization again."
                ),
            }
        }
    )
    def finalize_optimization(self, summary: str) -> Dict[str, Any]:
        """
        Deliver this optimization step: runs mandatory checks on ``mas_config/``, then finalizes.

        Checks (in order):
          1. Every ``.yaml`` / ``.yml`` under ``mas_config/`` parses.
          2. MAS structural rules (entry agent, subagent references, no delegation cycles).
          3. Skill and prompt validation: skill folder placement under correct agent directory,
             SKILL.md existence, frontmatter format (name + description), body non-empty,
             prompt_dict files exist, agents_llm_config.yaml agent consistency.
          4. TODO file completeness: every checklist item must be ``[x]`` (done) or
             ``[~]`` (dropped with reason). Items still marked ``[ ]`` or ``[r]``
             block finalization.

        If any check fails, returns ``success: false`` with detailed errors — fix issues
        and call this tool again. If all pass, records the summary, marks the optimizer task
        complete (react loop ends), and returns success.
        """
        mas_rel = "mas_config"
        mas_abs = self._resolve_workspace_path(mas_rel)

        # Stage 1: YAML syntax
        yaml_check = self._validate_all_yaml_under_dir(mas_abs)
        if not yaml_check["success"]:
            err_lines = "\n".join(yaml_check["errors"])
            return {
                "success": False,
                "data": {"stage": "yaml_syntax_scan", "mas_config_path": mas_rel},
                "error": (
                    "finalize_optimization blocked: YAML syntax error(s) under mas_config/. "
                    "Fix the listed files and call finalize_optimization again.\n" + err_lines
                ),
            }

        # Stage 2: MAS structural validation
        mas_check = self._validate_mas_config_dir(mas_rel)
        if not mas_check["success"]:
            detail = mas_check.get("error") or "Unknown MAS validation error"
            return {
                "success": False,
                "data": {"stage": "mas_structure", **( mas_check.get("data") or {})},
                "error": (
                    "finalize_optimization blocked: MAS structural validation failed. "
                    "Fix the issues below and call finalize_optimization again.\n" + detail
                ),
            }

        # Stage 3: Skill files, prompt files, agents_llm_config consistency
        sp_check = self._validate_skills_and_prompts(mas_abs)
        if not sp_check["success"]:
            err_lines = "\n".join(sp_check["errors"])
            return {
                "success": False,
                "data": {"stage": "skills_and_prompts", "error_count": len(sp_check["errors"])},
                "error": (
                    "finalize_optimization blocked: skill/prompt/config validation failed. "
                    "Fix ALL errors below and call finalize_optimization again.\n\n" + err_lines
                ),
            }

        # Stage 4: TODO file completeness — no orphaned [ ] or [r] items
        todo_check = self._validate_todo_completeness()
        if not todo_check["success"]:
            err_lines = "\n".join(todo_check["errors"])
            todo_path = todo_check.get("todo_path", "optimizer/todo.md")
            return {
                "success": False,
                "data": {"stage": "todo_completeness", "todo_path": todo_path},
                "error": (
                    f"finalize_optimization blocked: review `{todo_path}`. "
                    "Mark each Plan item [x] or [~] with an explicit 'reason: ...'.\n" + err_lines
                ),
            }
        todo_status = "(4) TODO file — all items resolved."

        self.optimization_summary = summary
        self.is_task_complete = True
        self.task_done_summary = summary
        graph = (mas_check.get("data") or {}).get("agent_graph", {})
        return {
            "success": True,
            "data": (
                "Optimization step finalized. All checks passed: "
                "(1) YAML syntax valid; "
                "(2) MAS structure valid (entry agent, references, no cycles); "
                "(3) all skill SKILL.md files present with correct frontmatter, "
                f"all prompt files exist, agents_llm_config.yaml consistent; "
                f"{todo_status} "
                f"Agent graph: {json.dumps(graph, ensure_ascii=False)}. "
                "Summary recorded; optimizer task is complete."
            ),
            "error": None,
        }
