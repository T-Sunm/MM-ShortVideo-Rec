import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper: <user, item, rating, visual> Tensor → Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor, visual_tensor):
        self.user_tensor   = user_tensor
        self.item_tensor   = item_tensor
        self.target_tensor = target_tensor
        self.visual_tensor = visual_tensor

    def __getitem__(self, index):
        return (self.user_tensor[index], self.item_tensor[index],
                self.target_tensor[index], self.visual_tensor[index])

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings, visual_embeddings: dict):
        """
        args:
            ratings: pd.DataFrame — ['userId', 'itemId', 'rating', 'timestamp']
            visual_embeddings: dict — {itemId (int): tensor(768)}
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        self.visual_embeddings = visual_embeddings
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        self.negatives = self._sample_negative(ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

    def _binarize(self, ratings):
        ratings = deepcopy(ratings)
        ratings.loc[ratings['rating'] > 0, 'rating'] = 1.0
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split"""
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test  = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        interact_status = (ratings.groupby('userId')['itemId']
                           .apply(set).reset_index()
                           .rename(columns={'itemId': 'interacted_items'}))
        interact_status['negative_items']   = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def _get_visual(self, item_id: int) -> torch.Tensor:
        return self.visual_embeddings.get(item_id, torch.zeros(768))

    def instance_a_train_loader(self, num_negatives, batch_size):
        users, items, ratings, visuals = [], [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(list(x), num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            visuals.append(self._get_visual(int(row.itemId)))
            for neg in row.negatives:
                users.append(int(row.userId))
                items.append(int(neg))
                ratings.append(0.0)
                visuals.append(self._get_visual(int(neg)))
        dataset = UserItemRatingDataset(
            user_tensor=torch.LongTensor(users),
            item_tensor=torch.LongTensor(items),
            target_tensor=torch.FloatTensor(ratings),
            visual_tensor=torch.stack(visuals),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, test_visuals     = [], [], []
        negative_users, negative_items, negative_visuals = [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            test_visuals.append(self._get_visual(int(row.itemId)))
            for neg in row.negative_samples:
                negative_users.append(int(row.userId))
                negative_items.append(int(neg))
                negative_visuals.append(self._get_visual(int(neg)))
        return [
            torch.LongTensor(test_users),
            torch.LongTensor(test_items),
            torch.stack(test_visuals),
            torch.LongTensor(negative_users),
            torch.LongTensor(negative_items),
            torch.stack(negative_visuals),
        ]
