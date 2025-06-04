#!/usr/bin/env python3
"""
Deployment script for SAP HANA AI Toolkit to Together.ai.

This script deploys the SAP HANA AI Toolkit to Together.ai's dedicated
endpoint service, providing GPU-accelerated inference capabilities.
"""

import os
import sys
import argparse
import logging
import json
from src.hana_ai.api.together_endpoint import deploy_to_together

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Deploy SAP HANA AI Toolkit to Together.ai'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='together.yaml',
        help='Path to Together.ai deployment configuration file'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='Together.ai API key (overrides environment variable)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output file for deployment information'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the deployment script."""
    args = parse_arguments()
    
    # Validate config file
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        logger.error(
            "Together.ai API key not provided. Set the TOGETHER_API_KEY environment variable "
            "or pass it with --api-key."
        )
        sys.exit(1)
    
    try:
        # Deploy to Together.ai
        logger.info(f"Deploying SAP HANA AI Toolkit to Together.ai using config: {args.config}")
        deployment_info = deploy_to_together(config_path=args.config, api_key=api_key)
        
        # Print deployment information
        logger.info("Deployment successful!")
        logger.info(f"Deployment ID: {deployment_info.get('deployment_id')}")
        logger.info(f"Status: {deployment_info.get('status')}")
        logger.info(f"Endpoint URL: {deployment_info.get('endpoint_url')}")
        logger.info(f"Model: {deployment_info.get('model')}")
        
        # Save deployment information if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(deployment_info, f, indent=2)
            logger.info(f"Deployment information saved to: {args.output}")
        
        # Provide instructions for using the endpoint
        logger.info("\nTo use the deployed endpoint:")
        logger.info(f"1. Update your application to use the endpoint URL: {deployment_info.get('endpoint_url')}")
        logger.info("2. Make API calls to the endpoint with your Together.ai API key for authentication")
        logger.info("3. Monitor the deployment through the Together.ai dashboard")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()