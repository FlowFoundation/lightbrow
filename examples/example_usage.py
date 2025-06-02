"""
Example usage of Lightbrow with MinIO S3 Connector

This script demonstrates how to set up and run the Lightbrow application using a MinIO S3-compatible storage service.
"""
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