# LightBrow

[![Test](https://github.com/FlowFoundation/lightbrow/actions/workflows/test.yml/badge.svg)](https://github.com/FlowFoundation/lightbrow/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/lightbrow.svg)](https://badge.fury.io/py/lightbrow)
[![Python Version](https://img.shields.io/pypi/pyversions/lightbrow.svg)](https://pypi.org/project/lightbrow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Fast and Fully Controlled Data Visualization Library built on top of Plotly, Dash, and Polars.

## Features
- Lightweight and fast s3 browser-based or jupyter notebook for s3 or any compatible object storage.
- Built on top of dash and mantine components for a modern UI.
- Supports deep and fast search through large directories.

## Installation

```bash
pip install lightbrow
```

## Quick Start

```python

#This script demonstrates how to set up and run the Lightbrow application using a MinIO S3-compatible storage service.
import os
from lightbrow.connectors import S3Connector
from lightbrow.browsers import S3Browser

# Example usage
if __name__ == "__main__":
    # IMPORTANT: Configure your MinIO credentials and endpoint here or via environment variables
    # For MinIO, access_key and secret_key are typically required.
    # Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set in your environment,
    # OR uncomment and set them in auth_config.
    # If you are using from aws s3 or any other S3-compatible service, you can set the credentials accordingly.
    minio_auth_config = {
        'access_key': os.getenv('MINIO_ACCESS_KEY', 'myaccesskey'),  # Replace or set env var
        'secret_key': os.getenv('MINIO_SECRET_KEY', 'mysecretkey'),  # Replace or set env var
        'endpoint_url': os.getenv('MINIO_ENDPOINT_URL', 'http://127.0.0.1:9000'), # Example: 'http://minio.example.com:9000'
        # 'region': 'us-east-1' # Boto3 requires a region, but MinIO is region-agnostic. 'us-east-1' is a common default.
    }

    # Check if placeholder keys are still present
    if 'YOUR_MINIO_ACCESS_KEY' in minio_auth_config['access_key'] or \
       'YOUR_MINIO_SECRET_KEY' in minio_auth_config['secret_key']:
        print("WARNING: MinIO access_key or secret_key placeholders are being used.")
        print("Please configure your actual MinIO credentials in the script or via environment variables (MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_ENDPOINT_URL).")
        # You might want to exit or use a default non-functional connector here
        # For demonstration, we'll proceed, but it likely won't connect.
    
    # Create S3 connector instance for MinIO
    minio_s3_connector = S3Connector(auth_config=minio_auth_config, default_prefix="s3a://") # default_prefix is "s3://"
    
    # You can define multiple buckets, they will all use the same connector instance here.
    # If different buckets need different credentials/endpoints, create separate S3Connector instances.
    bucket_connector_list = [
        ('sandbox', minio_s3_connector), # Replace 'sandboxbucket' with your actual MinIO bucket name
        ('experiment', minio_s3_connector), # Add more buckets as needed
        # ('publicbucket', S3Connector()) # Example for a public AWS S3 bucket (no auth_config)
    ]
    
    # Create browser application instance
    # max_depth for background indexing can be adjusted. None means index indefinitely deep.
    s3_browser_app = S3Browser(bucket_connector_pairs=bucket_connector_list, max_depth=None)
    
    # Run the application
    s3_browser_app.run(debug=True)
```

## Visualization Types


## Dashboard Layout

- bucket chooser: Allows users to select which bucket to view.
- browser: Displays the contents of the selected bucket.
- address bar: Shows the current path in the bucket.
- copy button: Copies the current path to the clipboard.
- search bar: Enables searching through the contents of the bucket.
- file viewer: Displays the entire information of the selected file - pop up.

## Notes

- LightBrow is designed to be a read-only browser for S3-compatible object storage.
- It does not support write operations like uploading, deleting, renaming, or downloading files.
- These operations may be implemented in the future. But currently it still under consideration because of the privacy and security implications.


## Write your own connector

You can write your own connector by inheriting from the `BaseConnector` class and implementing the required methods. This allows you to connect to any object storage service that is compatible with the S3 API.

```python
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
```
Check out the [base connector](./lightbrow/connectors/base.py) for more details on how to implement your own connector.

## Development

### Setup

1. Clone the repository
```bash
git clone https://github.com/FlowFoundation/lightbrow
cd blazingplot
```

2. Install development dependencies
```bash
pip install -e ".[dev]"
```

### Testing

```bash
pytest
```

### Building

```bash
python -m build
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
