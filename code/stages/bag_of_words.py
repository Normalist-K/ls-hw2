import numpy as np
import pickle
from pyturbo import Stage


class BagOfWords(Stage):

    """
    Input: features [N x D]
    Output: bag-of-words [W]
    """

    def allocate_resource(self, resources, *, weight_path):
        with open(weight_path, 'rb') as f:
            self.weight = pickle.load(f)
        return [resources]

    def get_bag_of_words(self, features: np.ndarray) -> np.ndarray:
        """
        features: [N x D]

        Return: count of each word, [W]
        """
        # TODO: Generate bag of words
        # Calculate pairwise distance between each feature and each cluster,
        # assign each feature to the nearest cluster, and count

        norm_weight = self.weight / np.linalg.norm(self.weight, axis=-1)[:, np.newaxis]
        norm_features = features / np.linalg.norm(features, axis=-1)[:, np.newaxis]
        similarity = norm_features @ norm_weight.T
        clusters = np.argmax(similarity, axis=1)
        bag = np.zeros(self.weight.shape[0])
        for c in clusters:
            bag[c] += 1

        return bag

    def get_video_feature(self, bags: np.ndarray) -> np.ndarray:
        """
        bags: [B x W]

        Return: pooled vector, [W]
        """
        # TODO: Aggregate frame-level bags into a video-level feature.

        return np.sum(bags, axis=0)

    def process(self, task):
        features = task.content
        bags = []
        for frame_features in features:
            bag = self.get_bag_of_words(frame_features)
            assert isinstance(bag, np.ndarray)
            assert bag.shape == self.weight.shape[:1]
            bags.append(bag)
        bags = np.stack(bags)
        video_bag = self.get_video_feature(bags)
        assert isinstance(video_bag, np.ndarray)
        assert video_bag.shape == self.weight.shape[:1]
        return task.finish(video_bag)
