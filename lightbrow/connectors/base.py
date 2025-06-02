
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from enum import Enum
from collections import defaultdict


class AccessError(Exception):
    """Custom exception for access-related errors."""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class AccessLevel(Enum):
    """Enumeration for different access levels."""
    FULL_ACCESS = "full"
    READ_ONLY = "read_only"
    NO_ACCESS = "no_access"
    UNKNOWN = "unknown"


@dataclass
class FileItem:
    """Represents a file or directory item."""
    name: str
    path: str
    is_directory: bool
    size: Optional[int] = None
    last_modified: Optional[datetime] = None
    access_level: AccessLevel = AccessLevel.UNKNOWN
    access_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'path': self.path,
            'is_directory': self.is_directory,
            'size': self.size,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
            'access_level': self.access_level.value,
            'access_error': self.access_error
        }


class BaseConnector(ABC):
    """Base class for storage connectors."""
    def __init__(self, auth_config: Optional[Dict[str, str]] = None, default_prefix: str = "s3://", debug: bool = False):
        self.auth_config = auth_config or {}
        self._index_cache: Dict[str, List[FileItem]] = {} # Cache for list_items results per path
        self._indexing_status: Dict[str, bool] = {}
        self._access_errors: Dict[str, str] = {}  # path -> error message
        self._executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4) # Use more workers
        
        # New path-based index: bucket -> {item_full_path: FileItem}
        self._path_index: Dict[str, Dict[str, FileItem]] = defaultdict(dict)
        self.default_prefix = default_prefix
        self.debug = debug
    @abstractmethod
    async def list_items(self, path: str = "") -> List[FileItem]:
        """List items in the given path."""
        pass
    
    @abstractmethod
    async def get_item_info(self, path: str) -> FileItem:
        """Get detailed information about a specific item."""
        pass
    
    @abstractmethod
    async def check_access(self, path: str) -> AccessLevel:
        """Check access level for a given path."""
        pass
    @abstractmethod
    def start_background_indexing(self, bucket: str, max_depth: Optional[int] = None):
        """Start background indexing for a bucket."""
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    def is_indexing(self, bucket: str) -> bool:
        """Check if bucket is currently being indexed."""
        return self._indexing_status.get(bucket, False)
    
    def get_cached_items(self, path: str) -> Optional[List[FileItem]]:
        """Get cached items for a path (from list_items direct cache)."""
        return self._index_cache.get(path)
    
    def get_access_error(self, path: str) -> Optional[str]:
        """Get access error message for a path."""
        return self._access_errors.get(path)
    
    def _cache_items(self, list_operation_path: str, items: List[FileItem]):
        """
        Caches items from a list operation.
        Populates the _index_cache for the specific list_operation_path.
        Populates the _path_index for broader, path-based searching.
        """
        self._index_cache[list_operation_path] = items

        if not items:
            return

        # Determine bucket from the first item's path (they should all be in the same bucket from one list_items call)
        # or use the list_operation_path to extract the bucket.
        first_item_path = items[0].path
        bucket = self._extract_bucket_from_path(first_item_path)
        
        if not bucket and list_operation_path: # Fallback if items might be empty or pathless
             bucket = self._extract_bucket_from_path(list_operation_path)

        if bucket:
            bucket_path_map = self._path_index[bucket]
            for item in items:
                bucket_path_map[item.path] = item # Add or update the item in the main path index

    def get_all_items_from_path_index(self, bucket: str) -> List[FileItem]:
        """Retrieves all FileItems for a given bucket from the _path_index."""
        if bucket in self._path_index:
            return list(self._path_index[bucket].values())
        return []

    def _extract_bucket_from_path(self, path: str) -> Optional[str]:
        """Helper to extract bucket from a full S3 path. Returns None if path is not a valid S3 path or no bucket."""
        if path.startswith(self.default_prefix):
            parts = path[len(self.default_prefix):].split('/', 1)
            if parts[0]: # Bucket name exists
                return parts[0]
        return None 
    
    def _cache_access_error(self, path: str, error_message: str):
        """Cache access error for a path."""
        self._access_errors[path] = error_message


