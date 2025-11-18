from .ppo import PPOWrapper
from .utils import HistoryLoggerCallback
from .compressed_history_callback import CompressedHistoryLoggerCallback
from .compressed_callback import CompressedHistoryCallback

ALGORITHM = {
    'PPO': PPOWrapper,
}