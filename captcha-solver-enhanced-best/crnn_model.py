import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM layer"""
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        # Use reshape instead of view for better compatibility
        t_rec = recurrent.reshape(T * b, h)
        output = self.embedding(t_rec)
        output = output.reshape(T, b, -1)
        return output


class CRNN(nn.Module):
    """
    CRNN: CNN + RNN + CTC Loss
    Best for variable-length text recognition in images
    """
    def __init__(self, img_height, num_chars, num_hidden=512):
        super(CRNN, self).__init__()
        
        # CNN layers for feature extraction
        self.cnn = nn.Sequential(
            # Conv 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x100
            
            # Conv 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x50
            
            # Conv 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 8x50
            
            # Conv 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Conv 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 4x50
            
            # Conv 7
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # 3x49
        )
        
        # RNN layers for sequence modeling
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512 * 3, num_hidden, num_hidden),
            BidirectionalLSTM(num_hidden, num_hidden, num_chars)
        )

    def forward(self, input):
        # CNN feature extraction
        conv = self.cnn(input)  # [B, C, H, W]
        
        # Prepare for RNN: reshape to [W, B, C*H]
        b, c, h, w = conv.size()
        
        # Make sure tensor is contiguous before reshaping
        conv = conv.contiguous()
        conv = conv.view(b, c * h, w)
        conv = conv.permute(2, 0, 1)  # [W, B, C*H]
        
        # Make contiguous again after permute
        conv = conv.contiguous()
        
        # RNN sequence modeling
        output = self.rnn(conv)  # [W, B, num_chars]
        
        return output