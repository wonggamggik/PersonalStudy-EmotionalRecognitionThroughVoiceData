import torch
import numpy as np
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import torch.nn as nn

# Regression Head 정의
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# EmotionModel 정의
class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1) # 평균 풀링
        logits = self.classifier(hidden_states)
        return hidden_states, logits


# 디바이스 설정 (CPU 또는 GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 모델 및 프로세서 로드
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)

# 예제 입력: 1초 길이의 무음 신호 (16000 Hz)
sampling_rate = 16000
signal = np.zeros((1, sampling_rate), dtype=np.float32)

def process_func(x: np.ndarray, sampling_rate: int, embeddings: bool = False) -> np.ndarray:
    # 전처리
    inputs = processor(x, sampling_rate=sampling_rate, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    
    # 모델 추론
    with torch.no_grad():
        hidden_states, logits = model(input_values)
    
    # embeddings 여부에 따라 반환
    # embeddings=True면 hidden_states 반환, 아니면 logits 반환
    output = hidden_states if embeddings else logits
    return output.cpu().numpy()

# 감정 예측
pred = process_func(signal, sampling_rate)
print("Arousal, Dominance, Valence:", pred)

# 임베딩 추출
emb = process_func(signal, sampling_rate, embeddings=True)
print("Embeddings shape:", emb.shape)
