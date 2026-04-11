import textwrap

import matplotlib
import matplotlib.pyplot as plt


PAPER_FIGSIZE = (9.5, 6.5)
PAPER_DPI = 300
PAPER_STYLE = {
    "font.family": "serif",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
}


def apply_paper_style():
    plt.rcParams.update(PAPER_STYLE)


def make_figure(figsize=None):
    return plt.subplots(figsize=figsize or PAPER_FIGSIZE)


def wrap_label(text, width=18):
    return textwrap.fill(str(text), width=width, break_long_words=False)


def format_title(text, width=34):
    return wrap_label(text, width=width)


def format_feature_label(name, width=22):
    label = str(name).replace("__", " / ").replace("_", " ")
    label = " ".join(label.split())
    label = label.replace("cpu", "CPU")
    label = label.replace("rss", "RSS")
    label = label.replace("io", "I/O")
    label = label.replace("mb", "MB")
    label = label.replace("ms", "ms")
    return wrap_label(label, width=width)


def maybe_show():
    backend = matplotlib.get_backend().lower()
    non_interactive_backends = {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }
    if backend not in non_interactive_backends:
        plt.show(block=True)


def style_axes(ax, title=None, xlabel=None, ylabel=None, title_width=34, title_pad=10):
    if title is not None:
        ax.set_title(format_title(title, width=title_width), pad=title_pad)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=8)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=8)


def finalize_figure(fig, top=0.94):
    fig.tight_layout(rect=(0.03, 0.03, 0.98, top))


def prompt_yes_no(message, default="n"):
    try:
        value = input(message).strip().lower()
    except EOFError:
        value = default
    return value or default


def prompt_filename(message, default):
    try:
        value = input(f"{message} [{default}]: ").strip()
    except EOFError:
        return default
    return value or default
