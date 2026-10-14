"""Prompt and Skill loading/formatting utilities."""
import re
from pathlib import Path
from string import Template
from typing import Dict, List, Optional
from datetime import datetime


def load_prompt_template(template_path: str) -> str:
    """Load a prompt template from a file path.

    Args:
        template_path: Path to the template file.

    Returns:
        The template content as a string.
    """
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_skill(skill_name: str, skill_dir: str, global_skill_dir: Optional[str] = None) -> Dict[str, str]:
    """Load a skill and return its name, description, and content.

    Tries paths in priority order:
    1. ``skill_dir/skill_name/SKILL.md``
    2. ``skill_dir/parent/sub_skills/child.md`` (legacy format, when ``/`` is present)
    3. Same patterns under *global_skill_dir* as fallback.

    Returns:
        ``{"name": str, "description": str, "content": str}``
    """
    if Path(skill_name).is_absolute() or ".." in Path(skill_name).parts:
        raise ValueError("Skill names must be relative paths within the skill directory.")
    # Build candidate paths in priority order:
    #   1. <skill_dir>/<skill_name>/SKILL.md  (works for both plain "foo" and nested "agent/foo")
    #   2. <skill_dir>/<parent>/sub_skills/<child>.md  (legacy sub_skill format, only when "/" present)
    candidates: list[Path] = [Path(skill_dir) / skill_name / "SKILL.md"]
    if '/' in skill_name:
        parent, child = skill_name.split('/', 1)
        candidates.append(Path(skill_dir) / parent / "sub_skills" / f"{child}.md")

    skill_path: Optional[Path] = None
    for c in candidates:
        if c.resolve().is_relative_to(Path(skill_dir).resolve()) and c.is_file():
            skill_path = c
            break

    if skill_path is None and global_skill_dir:
        global_candidates: list[Path] = [Path(global_skill_dir) / skill_name / "SKILL.md"]
        if '/' in skill_name:
            parent, child = skill_name.split('/', 1)
            global_candidates.append(Path(global_skill_dir) / parent / "sub_skills" / f"{child}.md")
        for c in global_candidates:
            if c.resolve().is_relative_to(Path(global_skill_dir).resolve()) and c.is_file():
                skill_path = c
                skill_dir = global_skill_dir
                break

    if skill_path is None:
        raise FileNotFoundError(f"Skill file not found: {skill_name} in {skill_dir}" + (f" or {global_skill_dir}" if global_skill_dir else ""))
    
    with open(skill_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract YAML front matter
    front_matter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    
    name = skill_name
    description = "No description"
    skill_content = content.strip()
    
    if front_matter_match:
        yaml_content = front_matter_match.group(1)
        
        # Extract name and description
        name_match = re.search(r'^name:\s*(.+)$', yaml_content, re.MULTILINE)
        desc_match = re.search(r'^description:\s*(.+)$', yaml_content, re.MULTILINE)
        
        if name_match:
            name = name_match.group(1).strip()
        if desc_match:
            description = desc_match.group(1).strip()
        
        # Remove front matter, keep body
        skill_content = content[front_matter_match.end():].strip()
    
    return {
        "name": name,
        "description": description,
        "content": skill_content
    }


def format_prompt_template(
    template: str,
    optional_skills_descp_str: str = "",
    system_protocols_str: str = "",
    today_date: Optional[str] = None
) -> str:
    """Fill a prompt template using ``string.Template`` substitution.

    Args:
        template: Template string using ``$variable`` syntax.
        optional_skills_descp_str: Description block for optional skills.
        system_protocols_str: Full-text injection of system protocols.
        today_date: Today's date (YYYY-MM-DD); auto-generated if ``None``.
    """
    if today_date is None:
        today_date = datetime.now().strftime("%Y-%m-%d")
    
    t = Template(template)
    return t.safe_substitute(
        optional_skills_descp_str=optional_skills_descp_str,
        system_protocols_str=system_protocols_str,
        today_date=today_date,
    )


def replace_tool_placeholders(
    content: str,
    skill_dir: str,
    skill_name: str,
    available_tool_names: Optional[set] = None
) -> str:
    """Replace ``{{tool_name}}`` placeholders in skill content.

    If the tool is available and a description file exists in ``extra_content/``,
    the placeholder is replaced with the file content. Otherwise it is removed
    (the entire line is deleted when the placeholder is the only content).
    """
    if available_tool_names is None:
        available_tool_names = set()
    
    # Find all placeholders (format: {{tool_name}})
    placeholder_pattern = r'\{\{([a-z_][a-z0-9_]*)\}\}'
    placeholders = set(re.findall(placeholder_pattern, content, re.IGNORECASE))
    
    if not placeholders:
        return content
    
    # Check whether extra_content directory exists
    extra_content_dir = Path(skill_dir) / skill_name / "extra_content"
    
    for tool_name in placeholders:
        placeholder = f'{{{{{tool_name}}}}}'
        
        # Try to load tool description file
        tool_content = None
        if tool_name in available_tool_names:
            tool_file = extra_content_dir / f"{tool_name}.md"
            if tool_file.exists() and tool_file.is_file():
                try:
                    with open(tool_file, 'r', encoding='utf-8') as f:
                        tool_content = f.read().strip()
                except Exception:
                    pass
        
        # Replace or remove placeholder
        if tool_content:
            content = content.replace(placeholder, tool_content)
        else:
            # Remove placeholder: delete entire line if placeholder is sole content
            lines = content.split('\n')
            content = '\n'.join(
                '' if line.strip() == placeholder else line.replace(placeholder, '')
                for line in lines
            )
    
    return content


def _resolve_skill_root(
    skill_name: str, skill_dir: str, global_skill_dir: Optional[str] = None
) -> str:
    """Directory that contains ``skill_name/SKILL.md`` (agent dir or global fallback)."""
    if (Path(skill_dir) / skill_name / "SKILL.md").is_file():
        return skill_dir
    if global_skill_dir and (Path(global_skill_dir) / skill_name / "SKILL.md").is_file():
        return global_skill_dir
    return skill_dir


def build_system_prompt_with_skills(
    prompt_template_path: str,
    skill_dir: Optional[str],
    system_skills: Optional[List[str]] = None,
    optional_skills: Optional[List[str]] = None,
    available_tool_names: Optional[set] = None,
    global_skill_dir: Optional[str] = None
) -> str:
    """Build a complete system prompt with protocols and optional skills.

    System protocols inject each ``SKILL.md`` body (with ``{{tool}}`` expansion)
    under its YAML ``name``. Optional skills appear as name + description only.
    """
    # Load prompt template
    template = load_prompt_template(prompt_template_path)
    
    # Load optional skill descriptions (missing items are silently skipped)
    optional_skills_descp_str = ""
    if optional_skills and skill_dir:
        descriptions = []
        for idx, skill_name in enumerate(optional_skills, 1):
            try:
                skill_dict = load_skill(skill_name, skill_dir, global_skill_dir)
                descriptions.append(
                    f"### Skill {idx}: {skill_dict['name']}\n"
                    f"**Description:** {skill_dict['description']}\n"
                )
            except FileNotFoundError:
                continue
        if descriptions:
            optional_skills_descp_str = f"\n\n{'='*20}\n\n".join(descriptions) + f"\n\n{'='*20}"
        else:
            optional_skills_descp_str = "No optional skills available."
    else:
        optional_skills_descp_str = "No optional skills available."
    
    # System protocols: YAML name + SKILL.md body; expand {{tool}} placeholders
    system_protocols_str = ""
    if system_skills and skill_dir:
        blocks: List[str] = []
        for idx, skill_name in enumerate(system_skills, 1):
            try:
                skill_dict = load_skill(skill_name, skill_dir, global_skill_dir)
            except FileNotFoundError:
                continue
            display_name = skill_dict.get("name", skill_name)
            root = _resolve_skill_root(skill_name, skill_dir, global_skill_dir)
            body = replace_tool_placeholders(
                skill_dict.get("content") or "",
                root,
                skill_name,
                available_tool_names,
            )
            blocks.append(
                f"### Protocol {idx}: {display_name}\n\n"
                f"{body}\n"
            )
        if blocks:
            system_protocols_str = f"\n\n{'='*20}\n\n".join(blocks) + f"\n\n{'='*20}"
        else:
            system_protocols_str = "No system protocols configured."
    else:
        system_protocols_str = "No system protocols configured."
    
    # Fill template
    return format_prompt_template(
        template,
        optional_skills_descp_str=optional_skills_descp_str,
        system_protocols_str=system_protocols_str,
    )
