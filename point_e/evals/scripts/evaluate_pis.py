"""
评估一批点云的P-IS。

点云批处理应保存到npz文件中，其中有
arr_0形状的键[N x K x 3]，其中K是每个的维数
点云，N是云的数量。

Evaluate P-IS of a batch of point clouds.

The point cloud batch should be saved to an npz file, where there is an
arr_0 key of shape [N x K x 3], where K is the dimensionality of each
point cloud and N is the number of clouds.
"""

import argparse

from point_e.evals.feature_extractor import PointNetClassifier, get_torch_devices
from point_e.evals.fid_is import compute_inception_score
from point_e.evals.npz_stream import NpzStreamer

#计算批次数据的 P-IS，并将结果打印出来。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("batch", type=str)
    args = parser.parse_args()

    print("creating classifier...")
    clf = PointNetClassifier(devices=get_torch_devices(), cache_dir=args.cache_dir)

    print("computing batch predictions")
    _, preds = clf.features_and_preds(NpzStreamer(args.batch))
    print(f"P-IS: {compute_inception_score(preds)}")


if __name__ == "__main__":
    main()
