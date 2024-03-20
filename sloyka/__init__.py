from loguru import logger
import sys

from .src import (
    EventDetection,
    Geocoder,
    TextClassifiers,
    GeoDataGetter,
    Semgraph,
    VkPostGetter,
    Streets, 
    VkCommentsParser,
    NER_parklike,
    CommentsReply
)

__all__ = [
    "EventDetection",
    "TextClassifiers",
    "Geocoder",
    "GeoDataGetter",
    "Semgraph",
    "VkPostGetter", 
    "Streets", 
    "VkCommentsParser",
    "NER_parklike",
    "CommentsReply"
]

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:MM-DD HH:mm}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)