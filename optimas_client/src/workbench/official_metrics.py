"""
Official WorkBench evaluation metrics (mirrors ``WorkBench/src/evals/utils.py``).

Imported separately from ``utils.py`` so evaluation does not require LangChain LLM
packages (``langchain_openai``, etc.) used only for inference scripts.
"""
from __future__ import annotations

import ast
from typing import List, Tuple

from .env import ensure_workbench_on_path


def _execute_action(action: str):
    """Dispatch only benchmark tool calls with literal arguments."""
    from src.tools.toolkits import all_tools

    call = ast.parse(action, mode="eval").body
    if not isinstance(call, ast.Call):
        raise ValueError("Expected a WorkBench tool call.")
    function = call.func
    if isinstance(function, ast.Attribute) and function.attr == "func":
        function = function.value
    if not (isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name)):
        raise ValueError("Expected domain.tool.func(...).")
    name = f"{function.value.id}.{function.attr}"
    tools = {tool.name: tool.func for tool in all_tools}
    if name not in tools or any(keyword.arg is None for keyword in call.keywords):
        raise ValueError("Unknown WorkBench tool or expanded keyword arguments.")
    args = [ast.literal_eval(arg) for arg in call.args]
    kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    return tools[name](*args, **kwargs)


def _domains():
    ensure_workbench_on_path()
    from src.tools import (  # noqa: WPS433
        calendar,
        email,
        analytics,
        project_management,
        customer_relationship_manager,
    )

    return [calendar, email, analytics, project_management, customer_relationship_manager]


def execute_actions_and_reset_state(actions: List[str]) -> Tuple:
    ensure_workbench_on_path()
    from src.tools import (  # noqa: WPS433
        calendar,
        email,
        analytics,
        project_management,
        customer_relationship_manager,
    )

    for domain in _domains():
        domain.reset_state()

    for action in actions:
        try:
            _execute_action(action)
        except Exception:
            continue

    new_calendar_state = calendar.CALENDAR_EVENTS.copy()
    new_email_state = email.EMAILS.copy()
    new_analytics_state = analytics.PLOTS_DATA.copy()
    new_project_management_state = project_management.PROJECT_TASKS.copy()
    new_customer_relationship_manager_state = customer_relationship_manager.CRM_DATA.copy()

    for domain in _domains():
        domain.reset_state()

    return (
        True,
        new_calendar_state,
        new_email_state,
        new_analytics_state,
        new_project_management_state,
        new_customer_relationship_manager_state,
    )


def get_function_name(action: str) -> str:
    return ".".join(action.split("(")[0].split(".")[0:2])


def is_exact_match(predicted_actions: List[str], ground_truth_actions: List[str]) -> bool:
    ensure_workbench_on_path()
    from src.tools.toolkits import tools_with_side_effects  # noqa: WPS433

    tools_with_side_effects_names = [str(function.name) for function in tools_with_side_effects]
    predicted_actions_with_side_effects = [
        action
        for action in predicted_actions
        if get_function_name(action) in tools_with_side_effects_names
    ]
    predicted_actions_with_side_effects = sorted(
        [action.lower() for action in predicted_actions_with_side_effects]
    )
    ground_truth_actions_sorted = sorted([action.lower() for action in ground_truth_actions])
    return predicted_actions_with_side_effects == ground_truth_actions_sorted


def is_correct(predicted_actions: List[str], ground_truth_actions: List[str], error: str) -> bool:
    if error:
        return False
    (
        successful_execution,
        predicted_calendar_state,
        predicted_email_state,
        predicted_analytics_state,
        predicted_project_management_state,
        predicted_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(predicted_actions)
    (
        _,
        ground_truth_calendar_state,
        ground_truth_email_state,
        ground_truth_analytics_state,
        ground_truth_project_management_state,
        ground_truth_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(ground_truth_actions)

    def convert_strs_to_lowercase(df):
        fields_not_to_convert = ["status", "list_name", "board"]
        for col in df.columns:
            if col not in fields_not_to_convert:
                df[col] = df[col].str.lower()
        return df

    predicted_calendar_state = convert_strs_to_lowercase(predicted_calendar_state)
    predicted_email_state = convert_strs_to_lowercase(predicted_email_state)
    predicted_analytics_state = convert_strs_to_lowercase(predicted_analytics_state)
    predicted_project_management_state = convert_strs_to_lowercase(
        predicted_project_management_state
    )
    predicted_customer_relationship_manager_state = convert_strs_to_lowercase(
        predicted_customer_relationship_manager_state
    )

    ground_truth_calendar_state = convert_strs_to_lowercase(ground_truth_calendar_state)
    ground_truth_email_state = convert_strs_to_lowercase(ground_truth_email_state)
    ground_truth_analytics_state = convert_strs_to_lowercase(ground_truth_analytics_state)
    ground_truth_project_management_state = convert_strs_to_lowercase(
        ground_truth_project_management_state
    )
    ground_truth_customer_relationship_manager_state = convert_strs_to_lowercase(
        ground_truth_customer_relationship_manager_state
    )

    return (
        successful_execution
        and predicted_calendar_state.equals(ground_truth_calendar_state)
        and predicted_email_state.equals(ground_truth_email_state)
        and predicted_analytics_state.equals(ground_truth_analytics_state)
        and predicted_project_management_state.equals(ground_truth_project_management_state)
        and predicted_customer_relationship_manager_state.equals(
            ground_truth_customer_relationship_manager_state
        )
    )


def has_side_effects(predicted_actions: List[str], ground_truth_actions: List[str]) -> bool:
    ensure_workbench_on_path()
    from src.tools import (  # noqa: WPS433
        calendar,
        email,
        analytics,
        project_management,
        customer_relationship_manager,
    )

    for domain in _domains():
        domain.reset_state()
    original_state = {
        "calendar": calendar.CALENDAR_EVENTS.copy(),
        "email": email.EMAILS.copy(),
        "analytics": analytics.PLOTS_DATA.copy(),
        "project_management": project_management.PROJECT_TASKS.copy(),
        "customer_relationship_manager": customer_relationship_manager.CRM_DATA.copy(),
    }
    (
        _,
        predicted_calendar_state,
        predicted_email_state,
        predicted_analytics_state,
        predicted_project_management_state,
        predicted_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(predicted_actions)

    state_changed = not predicted_calendar_state.equals(original_state["calendar"])
    state_changed |= not predicted_email_state.equals(original_state["email"])
    state_changed |= not predicted_analytics_state.equals(original_state["analytics"])
    state_changed |= not predicted_project_management_state.equals(original_state["project_management"])
    state_changed |= not predicted_customer_relationship_manager_state.equals(
        original_state["customer_relationship_manager"]
    )

    correct = is_correct(predicted_actions, ground_truth_actions, "")
    return state_changed and not correct
