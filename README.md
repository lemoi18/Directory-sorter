# FileMover

**FileMover** is a Python utility for organizing and moving files from a source directory to a destination directory based on either filename clustering or a tree-based tokenization method. This tool is especially useful for managing large collections of files with similar naming conventions.

## Features

- **Clustering-Based Organization**: Automatically group files into clusters based on their names. The tool supports two clustering methods:
  - **Levenshtein Distance**: Measures similarity between filenames based on their character differences.
  - **Embeddings**: Uses pre-trained sentence embeddings to determine similarity between filenames.

- **Tree-Based Organization**: Organize files into directories based on the first token of their filenames, split by a delimiter (default is a dash `-`).

- **Optimization**: Automatically find the optimal clustering distance for best results.

- **Cleanup**: Option to remove empty folders after file movement.

## Installation

To use FileMover, you need to have Python 3.7+ installed. Additionally, install the required dependencies:

```bash
pip install numpy scipy scikit-learn sentence-transformers python-Levenshtein
```
## Usage

The script can be run directly from the command line. Below are the usage instructions and examples:

### Command-Line Arguments

- `source_dir` (required): The source directory where files are located.
- `dest_dir` (required): The destination directory where files will be moved.
- `--method` (optional): The method to organize files, either `clustering` (default) or `tree`.
- `--use_embeddings` (optional): If provided, the script will use embeddings for clustering instead of Levenshtein distance.



## Examples

### Clustering-Based File Organization

To organize files using clustering with Levenshtein distance:

```bash
python file_mover.py /path/to/source /path/to/destination --method clustering
```
To organize files using clustering with sentence embeddings:

```bash
python file_mover.py /path/to/source /path/to/destination --method clustering --use_embeddings
```


## Tree-Based File Organization
To organize files into directories based on their filename structure:

```bash
python file_mover.py /path/to/source /path/to/destination --method tree
```


## Methods Overview

### Clustering Method

- **Levenshtein Distance**: The tool calculates the Levenshtein distance between filenames to cluster similar files together. The maximum distance for clustering is optimized automatically based on silhouette scores.

- **Embeddings**: Uses a pre-trained model from `sentence-transformers` to generate embeddings for filenames, allowing for more nuanced clustering.

### Tree Method

Organizes files by splitting filenames using a delimiter (default: `-`) and creating directories based on the first token.


## Error Handling

- **Non-Existent Directories**: The script checks if the provided directories exist and raises an error if not.
- **Permission Errors**: The script skips files that cannot be moved due to permission issues.
- **Missing Files**: If a file is not found during the move operation, it will be skipped with a warning.
