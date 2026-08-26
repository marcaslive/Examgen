# designer/templatetags/exam_filters.py

from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary by key."""
    if isinstance(dictionary, dict):
        return dictionary.get(str(key), '')
    return ''


@register.filter
def percentage(value, total):
    """Calculate percentage."""
    try:
        return round((int(value) / int(total)) * 100, 1)
    except (ValueError, ZeroDivisionError, TypeError):
        return 0


@register.filter
def file_size(value):
    """Format file size in bytes to human readable."""
    try:
        value = int(value)
    except (ValueError, TypeError):
        return '0 B'
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


@register.filter
def multiply(value, arg):
    """Multiply value by argument."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0