# Instruction

Celler helps you to track and segment cells from movies.
It is a semi-interactive toolkit built upon fiji imagej.
This is an instruction about how to use it.

## Setup

Install with Anaconda:
```bash
conda env create -f conda.yaml
```

## ROI types

ROI can be categorized into 2 types. Auto ROI and manual input.
AutoROIs are generated by segmentation algorithms.
Manual inputs are those input by the user.
When user inputs an ROI, they have 2 options:

- Use Polygon. Then the system will trust the input, and keep it without any edits.
- Use other tools, like oval and rectangle. 
The segmentation algorithm will **refine** the user selection by finding its closest cell.
The original input will be deleted and only the refined ROI will be kept.


## First step: Select a cell

When user starts a new cell, before everything else happens, 
they need to select a cell from the first frame to start with.
They may use polygon to accurately segment the cell,
or use other tools and let the algorithm to refine their selection (see previous section).

The system will alert if a nearby cell was tracked before
to avoid repeated trackings.

## Segmentation and tracking

If user did not input anything, the system starts to track the cell until
1. No cell can be found around the target cell. 
This happens when the cell moves out of the scope.
2. Last frame was analyzed.
3. User interrupts the process by hitting the "Deselect" button.

Then the system asks if it should
1. save: save what we have now. User may delete additional frames before saving.
2. discard: exit the current cell without saving.
3. continue: restart tracking (see below for details)

If the user chooses "continue", the system will check user inputs.
Again, user may use polygon to accurately segment the cell,
or use other tools and let the algorithm to refine their selection.
The system will restart tracking from the first edited frame.

## Paper reference

This tool is used by [Ras suppression potentiates rear actomyosin contractility-driven cell polarization and migration](https://www.nature.com/articles/s41556-024-01453-4), published in *Nature Cell Biology*.
The paper reference is

```bibtex
@article {lin2024ras,
  author = {Lin, Yiyan and Pal, Dhiman Sankar and Banerjee, Parijat and Banerjee, Tatsat and Qin, Guanghui and Deng, Yu and Borleis, Jane and Iglesias, Pablo A. and Devreotes, Peter N.},
  title = {Ras suppression potentiates rear actomyosin contractility-driven cell polarization and migration},
  year = {2024},
  URL = {https://www.nature.com/articles/s41556-024-01453-4},
  issue = {7},
  volumn = {26},
  pages = {1062--1076},
  journal = {Nature Cell Biology}
}
```
