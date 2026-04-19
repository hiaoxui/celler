# Intro

This project designs an interactive ImageJ tool that segment and track cells.
It uses PyImageJ to build a bridge between ImageJ2 and Python.
Human set instructions on ImageJ GUI, such as which cell to track, and the program will track
the cell through the movie and segment it out automatically.
Human can interrupt this process and refine the segmentation.

# Python

The Python interpreter version is 3.14. Do not maintain backward compatibility on purpose.
For type hints, use typing.
For paths, use pathlib.Path.

When you need to use Python, use the virtual environment defined in .venv.
