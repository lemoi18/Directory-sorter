import os
import shutil
import argparse
import numpy as np
import Levenshtein
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

class FileMover:
    def __init__(self, source_dir: str, dest_dir: str, max_distance: float = 0.6):
        self.source_dir = os.path.abspath(source_dir)
        self.dest_dir = os.path.abspath(dest_dir)
        self.max_distance = max_distance
        if not os.path.exists(self.source_dir):
            raise ValueError(f"Source directory {self.source_dir} does not exist!")
        if not os.path.exists(self.dest_dir):
            raise ValueError(f"Destination directory {self.dest_dir} does not exist!")

    def move_files(self, use_embeddings: bool = False):
        clusters_to_process = [(self.source_dir, self.dest_dir)]

        # Optimize max_distance before processing
        files = [file for file in os.listdir(self.source_dir) if os.path.isfile(os.path.join(self.source_dir, file))]
        if files:
            max_distance_values = list(np.linspace(0.4, 0.9, 9))
            optimal_max_distance = self.find_optimal_max_distance(files, max_distance_values)
            self.max_distance = optimal_max_distance

        while clusters_to_process:
            source_dir, dest_dir = clusters_to_process.pop(0)
            files = [file for file in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, file))]

            if not files:
                print("No files found in the source directory.")
                continue

            print("Clustering files...")
            if use_embeddings:
                clusters, _ = self._cluster_files_with_embeddings(files)  # New method using embeddings
            else:
                clusters, _ = self._cluster_files(files)  # Existing method
            
            print(f"Found {len(clusters)} clusters.")

            for cluster_id, file_names in clusters.items():
                cluster_substrings = [os.path.splitext(file_name)[0] for file_name in file_names]
                substring_counts = Counter(cluster_substrings)
                most_common_substring = substring_counts.most_common(1)[0][0]
                folder_name = f"{most_common_substring}_{cluster_id}"
                dest_folder = os.path.join(self.dest_dir, folder_name)
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)

                for file_name in file_names:
                    source_path = os.path.join(source_dir, file_name)
                    dest_path = os.path.join(dest_folder, file_name)

                    # Error check for non-existent file
                    if not os.path.exists(source_path)):
                        print(f"File {source_path} does not exist!")
                        continue

                    try:
                        shutil.move(source_path, dest_path)
                    except PermissionError:
                        print(f"Skipping {file_name}: Permission denied.")

                if os.path.exists(source_dir):
                    clusters_to_process.append((source_dir, dest_folder))

    def _cluster_files(self, files: List[str]) -> (dict, List[int]):
        if not files:
            print("No files to cluster.")
            return {}, []

        # Create a distance matrix
        num_files = len(files)
        distance_matrix = np.zeros((num_files, num_files))
        for i, f1 in enumerate(files):
            for j, f2 in enumerate(files):
                if i < j:
                    # Distance based on substrings
                    substring_distance = Levenshtein.distance(self.extract_substring(f1), self.extract_substring(f2))
                    normalized_substring_distance = substring_distance / max(len(self.extract_substring(f1)),
                                                                             len(self.extract_substring(f2)))

                    # Distance based on full filenames
                    full_distance = Levenshtein.distance(f1, f2)
                    normalized_full_distance = full_distance / max(len(f1), len(f2))

                    # Combined distance (you can adjust weights as needed)
                    combined_distance = 0.3 * normalized_substring_distance + 0.7 * normalized_full_distance

                    distance_matrix[i, j] = combined_distance
                    distance_matrix[j, i] = combined_distance

        # Convert the redundant matrix to a condensed one
        condensed_distance_matrix = squareform(distance_matrix)

        # Hierarchical clustering
        linkage_matrix = linkage(condensed_distance_matrix, method='average')
        labels = fcluster(linkage_matrix, self.max_distance, criterion='distance')

        cluster_groups = {}
        for file, label in zip(files, labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(file)

        unique_clusters = len(np.unique(labels))
        print(f"Number of unique clusters: {unique_clusters}")

        return cluster_groups, labels

    def _cluster_files_with_embeddings(self, files: List[str]) -> (dict, List[int]):
        if not files:
            print("No files to cluster.")
            return {}, []

        # Load a pre-trained model to generate embeddings
        model = self._load_embedding_model()

        # Generate embeddings for each file name
        embeddings = model.encode(files)

        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Hierarchical clustering based on cosine similarity
        clustering_model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=1 - self.max_distance)
        labels = clustering_model.fit_predict(1 - similarity_matrix)  # 1 - similarity because we need a distance metric

        cluster_groups = {}
        for file, label in zip(files, labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(file)

        unique_clusters = len(np.unique(labels))
        print(f"Number of unique clusters based on embeddings: {unique_clusters}")

        return cluster_groups, labels

    @lru_cache(maxsize=None)
    def _load_embedding_model(self):
        return SentenceTransformer('all-MiniLM-L6-v2')

    @lru_cache(maxsize=1024)
    def extract_substring(self, filename: str) -> str:
        length = len(filename)
        substring_length = length // 2  # Half the length of the filename
        start = (length - substring_length) // 2
        end = start + substring_length
        return filename[start:end]

    def find_optimal_max_distance(self, files: List[str], max_distance_values: List[float]) -> float:
        best_score = -1  # initialize with worst possible silhouette score
        best_max_distance = None

        for max_distance in max_distance_values:
            self.max_distance = max_distance
            _, labels = self._cluster_files(files)

            unique_clusters = len(np.unique(labels))
            if unique_clusters == 1 or unique_clusters == len(files):
                # Skip silhouette score computation if only 1 cluster or each file is its own cluster
                continue

            # Create a distance matrix
            num_files = len(files)
            distance_matrix = np.zeros((num_files, num_files))
            for i, f1 in enumerate(files):
                for j, f2 in enumerate(files):
                    if i < j:
                        distance = Levenshtein.distance(f1, f2)
                        normalized_distance = distance / max(len(f1), len(f2))
                        distance_matrix[i, j] = normalized_distance
                        distance_matrix[j, i] = normalized_distance

            # Compute silhouette score using the distance matrix and the labels
            score = silhouette_score(distance_matrix, labels, metric='precomputed')

            if score > best_score:
                best_score = score
                best_max_distance = max_distance

        return best_max_distance

    def cleanup_empty_folders(self, start_dir):
        for dirpath, dirnames, filenames in os.walk(start_dir, topdown=False):
            # Check if the directory is indeed a directory and not a file.
            for dirname in dirnames:
                full_dirpath = os.path.join(dirpath, dirname)
                if os.path.isdir(full_dirpath) and not os.listdir(full_dirpath):
                    os.rmdir(full_dirpath)
                    print(f"Removed empty folder: {full_dirpath}")

    def build_tree(self, files: List[str]) -> dict:
        tree = defaultdict(dict)
        for f in files:
            node = tree
            tokens = f.split('-')  # Tokenize the file name based on dashes
            for token in tokens:
                node = node.setdefault(token, {})
        return tree
    
    def create_dirs_from_tree(self, tree, current_path):
        for token, sub_tree in tree.items():
            new_dir = os.path.join(current_path, token)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            self.create_dirs_from_tree(sub_tree, new_dir)

    def move_files_tree_based(self):
        files = [f for f in os.listdir(self.source_dir) if os.path.isfile(os.path.join(self.source_dir, f))]
        if not files:
            print("No files found in the source directory.")
            return

        for file_name in files:
            # Split the filename on '-' and use the first token as the folder name.
            tokens = file_name.split('-')
            folder_name = tokens[0]  # The directory should be named after the first token
            dest_folder = os.path.join(self.dest_dir, folder_name)

            # Create the destination folder if it does not exist
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            # Construct the source and destination file paths
            source_path = os.path.join(self.source_dir, file_name)
            dest_path = os.path.join(dest_folder, file_name)

            # Move the file
            if os.path.exists(source_path):
                try:
                    shutil.move(source_path, dest_path)
                    print(f"Moved file {file_name} to {dest_folder}")
                except PermissionError:
                    print(f"Skipping {file_name}: Permission denied.")
            else:
                print(f"File {source_path} does not exist!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Move files from source to destination based on clustering or tree structure.")
    parser.add_argument('source_dir', type=str, help="Source directory where files are located.")
    parser.add_argument('dest_dir', type=str, help="Destination directory where files will be moved.")
    parser.add_argument('--method', type=str, choices=['clustering', 'tree'], default='clustering', help="Method to organize files: 'clustering' or 'tree'.")
    parser.add_argument('--use_embeddings', action='store_true', help="Use embeddings for clustering instead of Levenshtein distance.")
    
    args = parser.parse_args()

    file_mover = FileMover(args.source_dir, args.dest_dir)

    if args.method == 'clustering':
        file_mover.move_files(use_embeddings=args.use_embeddings)
    elif args.method == 'tree':
        file_mover.move_files_tree_based()

    print(f"Files moved to {args.dest_dir}.")
