from .ppo import PPOWrapper
from .utils import HistoryLoggerCallback
from .compressed_history_callback import CompressedHistoryLoggerCallback

ALGORITHM = {
    'PPO': PPOWrapper,
}