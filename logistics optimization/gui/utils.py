# gui/utils.py
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk

# Helper functions to create and grid widgets, separating widget config from grid config

def create_label(parent, text, row, column, **kwargs):
    """Creates and grids a ttk.Label, separating widget/grid kwargs."""
    # Grid options with defaults
    grid_opts = {
        'sticky': kwargs.pop('sticky', "w"), # Default sticky for labels
        'padx': kwargs.pop('padx', 5),
        'pady': kwargs.pop('pady', 2),
        'rowspan': kwargs.pop('rowspan', 1),
        'columnspan': kwargs.pop('columnspan', 1)
    }
    # Remaining kwargs are for the Label constructor (e.g., font, foreground, anchor)
    widget_opts = kwargs

    label = ttk.Label(parent, text=text, **widget_opts)
    label.grid(row=row, column=column, **grid_opts)
    return label

def create_entry(parent, textvariable, row, column, width=10, **kwargs):
    """Creates and grids a ttk.Entry, separating widget/grid kwargs."""
    # Grid options with defaults
    grid_opts = {
        'sticky': kwargs.pop('sticky', "ew"), # Default sticky for entries
        'padx': kwargs.pop('padx', 5),
        'pady': kwargs.pop('pady', 2),
        'rowspan': kwargs.pop('rowspan', 1),
        'columnspan': kwargs.pop('columnspan', 1)
    }
    # Remaining kwargs are for the Entry constructor (e.g., state, justify, validatecommand)
    widget_opts = kwargs
    widget_opts['width'] = width # Ensure width is included

    entry = ttk.Entry(parent, textvariable=textvariable, **widget_opts)
    entry.grid(row=row, column=column, **grid_opts)
    return entry

def create_button(parent, text, command, row, column, **kwargs):
    """
    Creates and grids a ttk.Button, separating widget/grid kwargs.
    Specifically handles the 'state' kwarg for the button constructor.
    """
    # Grid-specific options (pop with defaults)
    grid_opts = {
        'sticky': kwargs.pop('sticky', ""), # Default empty sticky for button
        'padx': kwargs.pop('padx', 5),
        'pady': kwargs.pop('pady', 5),
        'rowspan': kwargs.pop('rowspan', 1),
        'columnspan': kwargs.pop('columnspan', 1),
        'ipadx': kwargs.pop('ipadx', 0),
        'ipady': kwargs.pop('ipady', 0)
    }

    # Remaining kwargs are assumed to be for the Button constructor
    # Example common ones: state, width, style, image, compound
    widget_opts = kwargs

    # Create the button using WIDGET options (including 'state' if passed)
    button = ttk.Button(parent, text=text, command=command, **widget_opts)

    # Grid the button using only GRID options
    button.grid(row=row, column=column, **grid_opts)

    return button

def create_checkbox(parent, text, variable, row, column, **kwargs):
    """Creates and grids a ttk.Checkbutton, separating widget/grid kwargs."""
    # Grid options with defaults
    grid_opts = {
        'sticky': kwargs.pop('sticky', "w"),
        'padx': kwargs.pop('padx', 5),
        'pady': kwargs.pop('pady', 2),
        'rowspan': kwargs.pop('rowspan', 1),
        'columnspan': kwargs.pop('columnspan', 1)
    }
    # Remaining kwargs for Checkbutton constructor (e.g., state, command, style, width)
    widget_opts = kwargs

    checkbox = ttk.Checkbutton(parent, text=text, variable=variable, **widget_opts)
    checkbox.grid(row=row, column=column, **grid_opts)
    return checkbox

# Separator generally doesn't take many widget-specific kwargs via **kwargs
def create_separator(parent, row, column, columnspan=1, orient=tk.HORIZONTAL, **kwargs):
    """Creates and grids a ttk.Separator."""
    # Grid options
    sticky = kwargs.pop('sticky', "ew")
    padx = kwargs.pop('padx', 5) # Use pop or direct value
    pady = kwargs.pop('pady', 5)

    separator = ttk.Separator(parent, orient=orient)
    # Pass explicit grid options, ignore other kwargs for separator
    separator.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky)
    return separator