"""
SimpleSearch class for performing regex-based searches on indexed paths.
This class uses a BaseConnector to access indexed paths and allows for
searching with custom regex patterns derived from user queries.
"""


from typing import Dict, List, Optional, Tuple, Any
import json
import re

from functools import lru_cache
from ..connectors.base import BaseConnector, FileItem, AccessLevel






class SimpleSearch:
    """Search using indexed paths and custom regex compilation."""
    def __init__(self, connector: BaseConnector, debug: bool = False):
        self.connector = connector
        self.debug = debug
    @lru_cache(maxsize=256) # Increased cache size
    def _compile_search_regex(self, query: str) -> re.Pattern:
        """
        Compiles the user's search query into a regular expression.
        - Handles `*` as a non-greedy wildcard `.*?`.
        - Handles `<s>` for start-of-string anchor `^`.
        - Handles `<e>` for end-of-string anchor `$`.
        - All other characters are treated literally.
        - Search is case-insensitive.
        """
        core_pattern = query
        anchor_start = False
        anchor_end = False

        if core_pattern.startswith("<s>"):
            core_pattern = core_pattern[3:]
            anchor_start = True
        if core_pattern.endswith("<e>"):
            core_pattern = core_pattern[:-3]
            anchor_end = True

        # Handle cases where core_pattern becomes empty after stripping <s>/<e>
        # or if the original query was special (e.g., just "*")
        if not core_pattern: # e.g., query was "<s><e>" or "<s>" or "<e>"
            if anchor_start and anchor_end: # Query was "<s><e>"
                final_regex_pattern = "^$"    # Match empty string exactly
            elif anchor_start: # Query was "<s>"
                final_regex_pattern = "^"     # Match start, effectively an empty prefix
            elif anchor_end: # Query was "<e>"
                final_regex_pattern = "$"     # Match end, effectively an empty suffix
            else: 
                # This case (empty core_pattern, no anchors) means original query was empty or whitespace.
                # The search() method should handle this, but defensively:
                final_regex_pattern = ".*" # Match anything if somehow passed here
        elif core_pattern == "*": # Core pattern is a single wildcard
            final_regex_pattern = ".*"  # Greedy match anything
        else:
            # Split by '*' and escape parts, then join with '.*?'
            parts = core_pattern.split('*')
            escaped_parts = [re.escape(part) for part in parts]
            final_regex_pattern = ".*?".join(escaped_parts)

        if anchor_start:
            final_regex_pattern = "^" + final_regex_pattern
        if anchor_end:
            final_regex_pattern = final_regex_pattern + "$"
        
        try:
            return re.compile(final_regex_pattern, re.IGNORECASE)
        except re.error as e:
            if self.debug: print(f"Regex compilation error for pattern '{final_regex_pattern}' from query '{query}': {e}")
            # Fallback to a safe "match nothing" regex or re-raise
            return re.compile(r"(?!)") # A regex that never matches

    def search(self, query: str, bucket: str, search_path_prefix_key: Optional[str] = None) -> List[FileItem]:
        """
        Performs a search based on the query against item paths in the specified bucket.
        Uses the _path_index from the connector.
        """
        stripped_query = query.strip()
        # Allow "*" as a valid query, otherwise, empty/whitespace queries return no results.
        if not stripped_query and query != "*":
            return []
        if not bucket:
            if self.debug: print("Search error: Bucket name not provided.")
            return []

        try:
            regex = self._compile_search_regex(query) # Use original query for compilation
        except Exception as e: # Catch any error during regex compilation just in case
            if self.debug: print(f"Failed to compile search regex for query '{query}': {e}")
            return []

        results: List[FileItem] = []
        seen_paths: set[str] = set()
        
        # Construct the S3 search prefix path for filtering, if provided
        full_s3_search_prefix = ""
        if search_path_prefix_key:
            # Normalize: ensure it starts with s3://bucket/ and ends with /
            base_bucket_path = f"{self.connector.default_prefix}{bucket}/"
            norm_key = search_path_prefix_key.strip('/')
            if norm_key:
                full_s3_search_prefix = f"{base_bucket_path}{norm_key}/"
            else: # search_path_prefix_key was just "/" or "" or "   "
                full_s3_search_prefix = base_bucket_path
        
        # Get all items for the bucket from the connector's path index
        all_items_in_bucket = self.connector.get_all_items_from_path_index(bucket)

        for item in all_items_in_bucket:
            if item.path in seen_paths:
                continue
            if item.access_level == AccessLevel.NO_ACCESS: # Skip inaccessible items
                continue

            # Apply the optional path prefix filter
            if full_s3_search_prefix:
                if not item.path.startswith(full_s3_search_prefix):
                    continue
            
            # Perform the regex search on the item's full path
            if regex.search(item.path):
                results.append(item)
                seen_paths.add(item.path)
                if len(results) >= 500:  # Limit number of search results
                    if self.debug: print(f"Search limit of 500 results reached for query: '{query}'")
                    break
        
        return sorted(results, key=lambda x: (not x.is_directory, x.name[0].lower() != query[0].lower() if query else False)) # Sort by directory first, then name, prioritize query match
