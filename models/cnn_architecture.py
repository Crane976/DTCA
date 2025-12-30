# models/cnn_architecture.py (ADAPTIVE & ROBUST VERSION)
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Classifier(nn.Module):
    """一个动态适应输入维度的一维卷积分类器"""

    def __init__(self, feature_dim):  # feature_dim 现在主要用于文档目的
        super(CNN_Classifier, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # 使用padding=1保持长度
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 长度减半
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 长度再次减半
        )

        # ✅ 核心修改: 自适应池化层
        # 无论经过卷积和池化后, 序列长度L变成了多少,
        # 这个层都会把它强制池化成一个固定的长度, 比如 8
        self.adaptive_pool = nn.AdaptiveMaxPool1d(8)

        self.flatten = nn.Flatten()

        # 全连接层
        self.fc1 = nn.Linear(64 * 8, 128)  # 输入维度现在是固定的
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (N, feature_dim)
        x = x.unsqueeze(1)  # -> (N, 1, feature_dim)

        out = self.conv_block1(x)
        out = self.conv_block2(out)

        out = self.adaptive_pool(out)  # -> (N, 64, 8)

        out = self.flatten(out)  # -> (N, 64 * 8)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        logits = self.fc2(out)
        return logits

    def predict(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)