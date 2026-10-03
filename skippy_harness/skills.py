from pathlib import Path

import yaml
import warnings

SKILL_DIRS = [
    Path.cwd() / ".agents" / "skills",
    Path.home() / ".agents" / "skills",
]


def find_skills():
    """Map each skill name to its description and SKILL.md path."""
    skills = {}
    for directory in reversed(SKILL_DIRS):
        for path in sorted(directory.glob("*/SKILL.md")):
            try:
                text = path.read_text(encoding="utf-8")
                if not text.startswith("---"):
                    raise ValueError("missing YAML frontmatter")
                _, frontmatter, _ = text.split("---", 2)
                meta = yaml.safe_load(frontmatter)
                if not isinstance(meta, dict) or not isinstance(meta.get("name"), str) or not isinstance(meta.get("description"), str):
                    raise ValueError("name and description must be strings")
                description = " ".join(meta["description"].split())
                skills[meta["name"]] = {"description": description, "path": path}
            except (OSError, ValueError, yaml.YAMLError) as err:
                warnings.warn(f"Skipping invalid skill {path}: {err}", stacklevel=2)
    return skills


SKILLS = find_skills()


def skills_prompt():
    return "\n".join(f"- {name}: {s['description']}" for name, s in SKILLS.items())


def read_skill(name: str) -> str:
    """Open a skill and return its full instructions."""
    if name not in SKILLS:
        return f"No skill named '{name}'."
    return SKILLS[name]["path"].read_text(encoding="utf-8")
