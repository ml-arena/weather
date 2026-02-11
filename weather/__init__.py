"""
Weather temperature prediction environment.
"""
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

from .env import Env

__all__ = ['Env', 'PKG_DIR']
