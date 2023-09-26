import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split


def calculate_mean_f1(df, pred_labels, strict_equality=False):
    assert len(df) == len(pred_labels)
    
    unique_labels = np.unique(pred_labels)
    label_to_indices = {label: np.where(pred_labels == label)[0] for label in unique_labels}
    
    f1_values = []
    for i in range(len(df)):
        ground_truth_indices = np.where(df['label_group'].iloc[i] == df['label_group'])[0]
        predicted_indices = label_to_indices[pred_labels[i]]
        intersection_size = np.intersect1d(ground_truth_indices, predicted_indices).shape[0]
        
        precision = intersection_size / len(predicted_indices)
        recall = intersection_size / len(ground_truth_indices)
        f1_cur = 2 * precision * recall / (precision + recall)
        if strict_equality and f1_cur < 1 - 0.001:
            f1_cur = 0
        f1_values.append(f1_cur)
    
    return np.mean(f1_values)


def get_metric(col):
    def f1_score(row):
        n = len(np.intersect1d(row.target,row[col]))
        return 2 * n / (len(row.target) + len(row[col]))
    return f1_score


def build_target(df):
    tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
    df['target'] = df.label_group.map(tmp)
    return df


def get_subsample(df, sizes_ratio=0.2, drop_index=True):
    sub_df, _ = train_test_split(df, train_size=sizes_ratio, shuffle=True, random_state=0)
    label_group_vc = sub_df.label_group.value_counts().copy()
    labels_to_drop = label_group_vc[label_group_vc == 1].index
    sub_df = sub_df[~sub_df.label_group.isin(labels_to_drop)]
    if drop_index:
        return sub_df.reset_index()
    return sub_df


def cos_similarity_predict(df, feature_matrix=None, vectorizer=None, threshold_range=[0.99]):
    if feature_matrix is None:
        assert vectorizer is not None
        feature_matrix = vectorizer.fit_transform(df.title).toarray()
    
    eps = 10**(-3)
    if not 1 - eps < np.linalg.norm(feature_matrix[0]) < 1 + eps:
        row_norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        feature_matrix_normalized = feature_matrix / row_norms
    else:
        feature_matrix_normalized = feature_matrix
    
    preds = [[] for i in range(len(threshold_range))]
    metric_values = []
    
    chunk_size = 1024 * 4
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        left_bound = i * chunk_size
        right_bound = min(len(df), (i + 1) * chunk_size)
        
        similarity_sub_matrix = (feature_matrix_normalized @ feature_matrix_normalized[left_bound:right_bound].T).T
        
        for threshold_id, threshold in enumerate(threshold_range):
            for vector_id in range(right_bound - left_bound):
                indicies = np.where(similarity_sub_matrix[vector_id] > threshold)[0]
                pred_for_posting = df.iloc[indicies].posting_id.values
                preds[threshold_id].append(pred_for_posting)
     
    for i in range(len(threshold_range)):
        df['preds'] = preds[i]
        df['f1'] = df.apply(get_metric('preds'), axis=1)
        metric_value = df.f1.mean()
        metric_values.append(metric_value)
    return max(metric_values), threshold_range[np.argmax(metric_values)]