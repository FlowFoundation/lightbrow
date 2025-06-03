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
    @abstractmethod
    def start_background_indexing(self, bucket: str, max_depth: Optional[int] = None):
        """Start background indexing for a bucket."""
        raise NotImplementedError("This method should be implemented in the subclass.")


class S3Connector(BaseConnector):
    """S3-specific connector implementation."""
    
    def __init__(self, auth_config: Optional[Dict[str, str]] = None, default_prefix: str = "s3://", debug: bool = False):
        super().__init__(auth_config, default_prefix, debug)
        
        self._setup_s3_client()
        
    def _setup_s3_client(self):
        """Setup S3 client with authentication."""
        access_key = self.auth_config.get('access_key') or os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = self.auth_config.get('secret_key') or os.getenv('AWS_SECRET_ACCESS_KEY')
        endpoint_url = self.auth_config.get('endpoint_url')
        region = self.auth_config.get('region') or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        session_kwargs = {'region_name': region}
        if access_key and secret_key:
            session_kwargs.update({
                'aws_access_key_id': access_key,
                'aws_secret_access_key': secret_key
            })
        
        session = boto3.Session(**session_kwargs)
        client_kwargs = {}
        if endpoint_url:
            client_kwargs['endpoint_url'] = endpoint_url
            # Ensure virtual host addressing for MinIO/Ceph if endpoint_url is used and it's not AWS S3
            # Default is path style for custom endpoints, virtual for AWS.
            # Forcing path style if it's common for the user's setup:
            client_kwargs['config'] = Config(s3={'addressing_style': 'path'}) 
            
        self.s3_client = session.client('s3', **client_kwargs)
    
    async def check_access(self, path: str) -> AccessLevel:
        """Check access level for S3 path."""
        bucket, key = self._parse_s3_path(path)
        loop = asyncio.get_event_loop()

        if not bucket: 
            try:
                await loop.run_in_executor(self._executor, self.s3_client.list_buckets)
                return AccessLevel.READ_ONLY 
            except ClientError as e:
                self._cache_access_error(path or self.default_prefix, self._format_s3_error(e, path or self.default_prefix))
                return AccessLevel.NO_ACCESS
            except Exception as e: # Catch any other unexpected errors
                self._cache_access_error(path or self.default_prefix, f"Unexpected error checking access for '{path or self.default_prefix}': {str(e)}")
                return AccessLevel.NO_ACCESS

        try:
            await loop.run_in_executor(
                self._executor,
                lambda: self.s3_client.list_objects_v2(Bucket=bucket, Prefix=key, Delimiter='/', MaxKeys=1)
            )
            return AccessLevel.READ_ONLY
        except ClientError as e:
            self._cache_access_error(path, self._format_s3_error(e, path))
            # Distinguish between "path does not exist but bucket is accessible" vs "access denied to bucket/path"
            if e.response.get('Error', {}).get('Code') == 'NoSuchKey':
                 # If parent is listable, this specific key doesn't exist, but path up to it might be fine.
                 # For simplicity, if list_objects_v2 fails on the prefix, treat as no_access to that prefix for now.
                 # A more granular check could try listing the parent.
                 return AccessLevel.NO_ACCESS 
            return AccessLevel.NO_ACCESS
        except Exception as e:
            self._cache_access_error(path, f"Unexpected error checking access for '{path}': {str(e)}")
            return AccessLevel.NO_ACCESS
    
    async def list_items(self, path: str = "") -> List[FileItem]:
        """List items in S3 path with access control."""
        # Check access to the path itself first
        access_level = await self.check_access(path)
        if access_level == AccessLevel.NO_ACCESS:
            error_msg = self.get_access_error(path) or f"Access denied to path: {path}."
            # Ensure error is cached if not already
            if not self.get_access_error(path): self._cache_access_error(path, error_msg)
            raise AccessError(error_msg, "AccessDeniedToList")
        
        bucket, prefix = self._parse_s3_path(path)
        if not bucket: # Root path, list buckets
            return await self._list_buckets()
        
        # Check cache for this specific listing path
        cached = self.get_cached_items(path)
        if cached: return cached
        
        try:
            items = await self._list_objects(bucket, prefix)
            self._cache_items(path, items) # Cache these specific listed items and update global path index
            return items
        except AccessError: # Propagate AccessError from _list_objects
            raise
        except Exception as e:
            error_msg = f"Unexpected error listing items in {path}: {str(e)}"
            self._cache_access_error(path, error_msg)
            raise AccessError(error_msg, "ListItemsUnexpectedError")


    def _format_s3_error(self, error: ClientError, path: str) -> str:
        """Format S3 error message for user display."""
        error_code = "UnknownS3Error"
        error_message = str(error)

        if hasattr(error, 'response') and error.response and 'Error' in error.response:
            error_code = error.response['Error'].get('Code', error_code)
            error_message = error.response['Error'].get('Message', error_message)
        elif isinstance(error, NoCredentialsError):
            error_code = 'NoCredentialsError'
            error_message = "AWS credentials not found."
        
        # User-friendly messages
        if error_code == 'AccessDenied': return f"Access Denied: Cannot access '{path}'. Please check permissions and policies."
        if error_code == 'Forbidden': return f"Forbidden: Access to '{path}' is restricted. Please check policies."
        if error_code == 'NoSuchBucket': return f"Bucket Not Found: The bucket for '{path}' does not exist or is not accessible."
        if error_code == 'NoSuchKey': return f"Object or Prefix Not Found: '{path}' does not exist."
        if error_code == 'InvalidAccessKeyId': return "Invalid AWS Access Key ID. Please check your credentials."
        if error_code == 'SignatureDoesNotMatch': return "AWS Signature Mismatch. Please check your secret key, region, and endpoint configuration."
        if error_code == 'NoCredentialsError': return f"No AWS Credentials found for '{path}'. Please configure your access key and secret key."
        if error_code == 'ExpiredToken': return f"Expired Token: The security token included in the request is expired for '{path}'."
        return f"S3 Error accessing '{path}': {error_code}. Details: {error_message}."
    
    async def get_item_info(self, path: str) -> FileItem:
        """Get S3 object information with access control."""
        bucket, key = self._parse_s3_path(path)
        loop = asyncio.get_event_loop()

        # Handle root "s3://"
        if not bucket:
            access = await self.check_access(self.default_prefix)
            err = self.get_access_error(self.default_prefix) if access == AccessLevel.NO_ACCESS else None
            return FileItem("S3", self.default_prefix, True, access_level=access, access_error=err)
        
        # Handle bucket root "s3://bucket/"
        if not key: 
            access = await self.check_access(path)
            err = self.get_access_error(path) if access == AccessLevel.NO_ACCESS else None
            return FileItem(bucket, path, True, access_level=access, access_error=err)

        # Handle object or directory prefix "s3://bucket/key" or "s3://bucket/folder/"
        item_name = key.rstrip('/').split('/')[-1]
        try:
            # If it doesn't end with '/', try head_object (could be file or empty prefix)
            if not key.endswith('/'):
                response = await loop.run_in_executor(self._executor, lambda: self.s3_client.head_object(Bucket=bucket, Key=key))
                return FileItem(item_name, path, False, 
                                response.get('ContentLength'), response.get('LastModified'), 
                                AccessLevel.READ_ONLY) # If head_object succeeds, we have read access to metadata

            # If it ends with '/', it's explicitly a directory/prefix. Check its listability.
            else: # key.endswith('/')
                access = await self.check_access(path) # check_access uses list_objects_v2 with MaxKeys=1
                err = self.get_access_error(path) if access == AccessLevel.NO_ACCESS else None
                return FileItem(item_name, path, True, access_level=access, access_error=err)

        except ClientError as e:
            formatted_error = self._format_s3_error(e, path)
            self._cache_access_error(path, formatted_error)
            
            # After head_object fails (e.g. 404 for non-file, or 403),
            # or if it was a directory path initially, we re-check access using list_objects_v2 logic.
            access_lvl_after_fail = await self.check_access(path)
            
            # Determine if it's a directory: if it ends with '/', or if it's listable after a head_object failure.
            is_dir = key.endswith('/') or (access_lvl_after_fail == AccessLevel.READ_ONLY and e.response.get('Error', {}).get('Code') == 'NoSuchKey')
            
            # If access is NO_ACCESS, use the error from the most relevant operation (head_object or list_objects)
            final_error = self.get_access_error(path) # Get potentially updated error from check_access

            return FileItem(item_name, path, is_dir, 
                            access_level=access_lvl_after_fail, 
                            access_error=final_error if access_lvl_after_fail == AccessLevel.NO_ACCESS else None)
        except Exception as e: # Catch-all for other unexpected errors
            error_msg = f"Unexpected error getting item info for {path}: {str(e)}"
            self._cache_access_error(path, error_msg)
            # Best guess for item_name if key is empty (e.g. path was just "s3://bucket")
            name_guess = item_name if item_name else bucket 
            return FileItem(name_guess, path, True, # Assume directory on unknown error
                            access_level=AccessLevel.UNKNOWN, access_error=error_msg)

    async def _list_buckets(self) -> List[FileItem]:
        """List S3 buckets."""
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(self._executor, self.s3_client.list_buckets)
            bucket_items = []
            for b_info in response.get('Buckets', []):
                b_name = b_info['Name']
                b_path = f"{self.default_prefix}{b_name}/"
                # Check access for each bucket individually. This can be slow.
                # Consider making this check optional or deferring it.
                access = await self.check_access(b_path)
                err = self.get_access_error(b_path) if access == AccessLevel.NO_ACCESS else None
                bucket_items.append(FileItem(
                    b_name, b_path, True, 
                    last_modified=b_info.get('CreationDate'),
                    access_level=access,
                    access_error=err
                ))
            return bucket_items
        except ClientError as e:
            root_path_display = self.default_prefix.rstrip('/') or "S3 root"
            err_msg = self._format_s3_error(e, root_path_display)
            self._cache_access_error(self.default_prefix, err_msg) # Cache error for the root
            raise AccessError(err_msg, e.response.get('Error', {}).get('Code', 'ListBucketsClientError'))
        except Exception as e:
            root_path_display = self.default_prefix.rstrip('/') or "S3 root"
            err_msg = f"Unexpected error listing buckets: {str(e)}"
            self._cache_access_error(self.default_prefix, err_msg)
            raise AccessError(err_msg, "ListBucketsUnexpectedError")


    async def _list_objects(self, bucket: str, prefix: str = "") -> List[FileItem]:
        """Helper to list objects and common prefixes for a given bucket/prefix."""
        items_map: Dict[str, FileItem] = {} # Use a map to handle potential duplicates from S3 listing edge cases
        loop = asyncio.get_event_loop()
        current_s3_prefix_path = f"{self.default_prefix}{bucket}/{prefix}"

        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            # Use run_in_executor for the entire pagination process if the library doesn't support async iteration directly
            # This simplifies the async/await pattern for boto3's paginators.
            
            async def fetch_page(token=None):
                params = {'Bucket': bucket, 'Prefix': prefix, 'Delimiter': '/'}
                if token: params['ContinuationToken'] = token
                return await loop.run_in_executor(self._executor, lambda: self.s3_client.list_objects_v2(**params))

            next_token = None
            while True:
                page = await fetch_page(next_token)
                
                # Process CommonPrefixes (directories)
                for common_prefix_obj in page.get('CommonPrefixes', []):
                    dir_key = common_prefix_obj['Prefix']
                    if dir_key == prefix and prefix: continue # Skip the prefix itself if listed as common
                    
                    dir_name = dir_key.rstrip('/').split('/')[-1]
                    if not dir_name: continue # Should not happen with valid S3 keys
                    
                    full_dir_s3_path = f"{self.default_prefix}{bucket}/{dir_key}"
                    if full_dir_s3_path not in items_map:
                         # Optimistic: if listed as common prefix, assume readable for now.
                         # Full check_access on click/interaction.
                        items_map[full_dir_s3_path] = FileItem(dir_name, full_dir_s3_path, True, access_level=AccessLevel.READ_ONLY)

                # Process Contents (files)
                for obj_content in page.get('Contents', []):
                    obj_key = obj_content['Key']
                    if obj_key == prefix: continue # Skip the prefix itself if listed as an object (e.g. an empty file representing a folder)

                    # Determine name relative to current prefix for display
                    name_in_folder = obj_key
                    if prefix and obj_key.startswith(prefix):
                        name_in_folder = obj_key[len(prefix):]
                    
                    if '/' in name_in_folder.rstrip('/'): continue # Skip items in sub-subfolders (should be caught by CommonPrefixes)
                    if not name_in_folder: continue # Skip if name becomes empty (e.g. obj_key was same as prefix)

                    file_s3_path = f"{self.default_prefix}{bucket}/{obj_key}"
                    if file_s3_path not in items_map:
                        items_map[file_s3_path] = FileItem(
                            name_in_folder.split('/')[-1], # Get the actual file/object name
                            file_s3_path, False,
                            size=obj_content.get('Size'), last_modified=obj_content.get('LastModified'),
                            access_level=AccessLevel.READ_ONLY # Optimistic
                        )
                
                if page.get('IsTruncated'):
                    next_token = page.get('NextContinuationToken')
                    if not next_token: break # Should not happen if IsTruncated is true, but safety break
                else:
                    break # No more pages
            
            return sorted(list(items_map.values()), key=lambda x: (not x.is_directory, x.name.lower()))

        except ClientError as e: 
            err_msg = self._format_s3_error(e, current_s3_prefix_path)
            self._cache_access_error(current_s3_prefix_path, err_msg)
            raise AccessError(err_msg, e.response.get('Error', {}).get('Code', 'ListObjectsClientError'))
        except Exception as e: 
            err_msg = f"Error listing objects in {current_s3_prefix_path}: {str(e)}"
            self._cache_access_error(current_s3_prefix_path, err_msg)
            raise AccessError(err_msg, "ListObjectsInternalError")

    async def _paginate_async(self, paginator, **kwargs): # This was a helper, ensure it's used or remove if direct executor use is preferred
        """Async wrapper for paginator. (Potentially replaced by direct executor use in _list_objects)"""
        loop = asyncio.get_event_loop()
        # This creates a blocking iterator; might be better to wrap individual page fetches
        page_iterator = await loop.run_in_executor(self._executor, lambda: paginator.paginate(**kwargs))
        for page in page_iterator: yield page 
    
    def _parse_s3_path(self, path: str) -> Tuple[Optional[str], str]:
        """Parses s3://bucket/key type paths."""
        if not path: return None, ""
        
        path_no_scheme = path
        if path.startswith(self.default_prefix): 
            path_no_scheme = path[len(self.default_prefix):]
        elif path.startswith("s3:/") and not path.startswith("s3://"): # Handle "s3:/bucket/key"
            path_no_scheme = path[4:]
            if path_no_scheme.startswith("/"): path_no_scheme = path_no_scheme[1:]


        if not path_no_scheme: return None, "" # Path was just "s3://" or "s3:/"
        
        parts = path_no_scheme.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket if bucket else None, key
    
    def start_background_indexing(self, bucket: str, max_depth: Optional[int] = None):
        """Synchronous method to start background indexing for a bucket."""
        if not bucket:
            if self.debug: print("Error: Bucket name cannot be empty for background indexing.")
            return
        if self.is_indexing(bucket):
            if self.debug: print(f"Indexing already in progress for bucket: {bucket}")
            return
        
        self._indexing_status[bucket] = True
        if self.debug: print(f"Starting background indexing for bucket: {bucket} up to depth: {max_depth}")

        def thread_target():
            # Each thread needs its own event loop if using asyncio within the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # The _index_bucket_recursive should use this loop for its async operations
                loop.run_until_complete(self._index_bucket_recursive(bucket, "", max_depth, 0))
            except Exception as e:
                if self.debug: print(f"Error during background indexing task for bucket {bucket}: {e}")
                import traceback; traceback.print_exc()
            finally:
                self._indexing_status[bucket] = False
                if self.debug: print(f"Background indexing finished or stopped for bucket: {bucket}")
                loop.close()
        
        indexing_thread = threading.Thread(target=thread_target, daemon=True)
        indexing_thread.start()

    async def _index_bucket_recursive(self, bucket: str, prefix: str, max_depth: Optional[int], current_depth: int):
        """Recursively index bucket contents."""
        if max_depth is not None and current_depth >= max_depth:
            if self.debug: print(f"Max depth {max_depth} reached for '{prefix}' in '{bucket}'. Stopping this branch.")
            return
        
        current_s3_path = f"{self.default_prefix}{bucket}/{prefix}"
        try:
            if self.debug: print(f"Indexing (depth {current_depth}): {current_s3_path}")
            # list_items will use check_access, and if successful, list and cache.
            items = await self.list_items(current_s3_path) # This populates _path_index via _cache_items
            
            for item in items:
                if item.is_directory and item.access_level != AccessLevel.NO_ACCESS:
                    # Extract the sub-prefix from the item's full path to recurse
                    _bucket_from_item, sub_prefix_key = self._parse_s3_path(item.path)
                    if _bucket_from_item != bucket: # Safety check, should not happen
                        if self.debug: print(f"Warning: Item {item.path} in bucket {bucket} has different bucket {_bucket_from_item} in its path.")
                        continue
                    if sub_prefix_key == prefix : # Avoid infinite loop on the prefix itself if it's listed as a directory.
                        continue

                    # Ensure sub_prefix_key ends with '/' if it's a directory for consistent S3 prefixing
                    if not sub_prefix_key.endswith('/') and item.is_directory:
                        sub_prefix_key += '/'
                    
                    await self._index_bucket_recursive(bucket, sub_prefix_key, max_depth, current_depth + 1)
                elif item.is_directory and item.access_level == AccessLevel.NO_ACCESS:
                    if self.debug: print(f"Skipping inaccessible directory during index: {item.path} ({item.access_error})")
        except AccessError as e:
            if self.debug: print(f"Access denied while indexing {current_s3_path}: {e.message}. Stopping this branch.")
            # Error is already cached by check_access or list_items
        except Exception as e:
            if self.debug: print(f"Unexpected error while indexing {current_s3_path}: {e}. Stopping this branch.")
            if not self.get_access_error(current_s3_path): # Cache if not already set
                 self._cache_access_error(current_s3_path, f"Unexpected indexing error: {str(e)}")
            import traceback; traceback.print_exc()


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