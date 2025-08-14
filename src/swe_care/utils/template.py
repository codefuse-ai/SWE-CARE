"""
Utility functions for rendering Jinja2 templates.
"""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template

# Initialize Jinja2 environment
_template_dir = Path(__file__).parent.parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(_template_dir))


def render_template(template_name: str, **context: Any) -> str:
    """
    Render a Jinja2 template with the provided context.

    Args:
        template_name: Name of the template file in the templates directory
        **context: Context variables to pass to the template

    Returns:
        Rendered template string
    """
    template = _jinja_env.get_template(template_name)
    return template.render(**context)


def get_template(template_name: str) -> Template:
    """
    Get a Jinja2 template object.

    Args:
        template_name: Name of the template file in the templates directory

    Returns:
        Jinja2 Template object
    """
    return _jinja_env.get_template(template_name)
