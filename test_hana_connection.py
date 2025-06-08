#!/usr/bin/env python3
"""
SAP HANA Connection Test Script

This script tests the connection to SAP HANA Cloud using environment variables
or command-line arguments. It's useful for verifying production credentials
before deployment.

Usage:
  python test_hana_connection.py 
  python test_hana_connection.py --host <host> --port <port> --user <user> --password <password>
"""

import os
import argparse
import sys
from dotenv import load_dotenv
import traceback

# Set up argument parsing
parser = argparse.ArgumentParser(description='Test connection to SAP HANA.')
parser.add_argument('--env-file', type=str, help='Path to .env file with HANA credentials')
parser.add_argument('--host', type=str, help='HANA host address')
parser.add_argument('--port', type=int, help='HANA port number')
parser.add_argument('--user', type=str, help='HANA username')
parser.add_argument('--password', type=str, help='HANA password')
parser.add_argument('--schema', type=str, help='HANA schema name')
parser.add_argument('--encrypt', action='store_true', default=True, help='Use encryption for connection')

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text, color):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.ENDC}")

def test_connection(host, port, user, password, schema=None, encrypt=True):
    """Test connection to SAP HANA."""
    try:
        # Try to import necessary modules
        print_colored("Importing hana_ml...", Colors.BLUE)
        try:
            from hana_ml.dataframe import ConnectionContext
        except ImportError:
            print_colored("Error: hana_ml package not installed. Installing...", Colors.YELLOW)
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "hana-ml"])
            from hana_ml.dataframe import ConnectionContext
        
        print_colored(f"Connecting to SAP HANA at {host}:{port}...", Colors.BLUE)
        
        # Create connection context
        connection = ConnectionContext(
            address=host,
            port=port,
            user=user,
            password=password,
            encrypt=encrypt,
            sslValidateCertificate=False
        )
        
        # Test connection with simple query
        print_colored("Connection established. Running test query...", Colors.GREEN)
        result = connection.sql("SELECT 'Connection test successful' AS result FROM DUMMY").collect()
        print_colored(f"Query result: {result.iloc[0]['result']}", Colors.GREEN)
        
        # Test schema if provided
        if schema:
            print_colored(f"Testing access to schema {schema}...", Colors.BLUE)
            try:
                connection.sql(f"SET SCHEMA {schema}").collect()
                print_colored(f"Successfully accessed schema {schema}", Colors.GREEN)
            except Exception as e:
                print_colored(f"Error accessing schema {schema}: {str(e)}", Colors.RED)
                return False
        
        # Close connection
        connection.close()
        print_colored("Connection test completed successfully!", Colors.GREEN)
        return True
        
    except Exception as e:
        print_colored(f"Error connecting to SAP HANA: {str(e)}", Colors.RED)
        print_colored("Error details:", Colors.RED)
        traceback.print_exc()
        return False

def main():
    """Main function to parse arguments and run the test."""
    args = parser.parse_args()
    
    # Load environment variables from file if specified
    if args.env_file:
        load_dotenv(args.env_file)
        print_colored(f"Loaded environment from {args.env_file}", Colors.BLUE)
    else:
        # Try to load from default .env files
        for env_file in ['production-environment.env', '.env', 'deployment/nvidia-t4/t4-environment.env']:
            if os.path.exists(env_file):
                load_dotenv(env_file)
                print_colored(f"Loaded environment from {env_file}", Colors.BLUE)
                break
    
    # Get connection parameters from arguments or environment variables
    host = args.host or os.environ.get('HANA_HOST')
    port = args.port or int(os.environ.get('HANA_PORT', 443))
    user = args.user or os.environ.get('HANA_USER')
    password = args.password or os.environ.get('HANA_PASSWORD')
    schema = args.schema or os.environ.get('HANA_SCHEMA')
    encrypt = args.encrypt if args.encrypt is not None else os.environ.get('HANA_ENCRYPT', 'true').lower() == 'true'
    
    # Check if required parameters are provided
    if not host or not user or not password:
        print_colored("Error: Missing required connection parameters.", Colors.RED)
        print_colored("Please provide --host, --user, and --password arguments or set environment variables:", Colors.RED)
        print_colored("  HANA_HOST, HANA_USER, HANA_PASSWORD", Colors.RED)
        parser.print_help()
        return 1
    
    print_colored("=================================", Colors.HEADER)
    print_colored("  SAP HANA Connection Test Tool  ", Colors.HEADER)
    print_colored("=================================", Colors.HEADER)
    print_colored(f"Host: {host}", Colors.CYAN)
    print_colored(f"Port: {port}", Colors.CYAN)
    print_colored(f"User: {user}", Colors.CYAN)
    print_colored(f"Schema: {schema if schema else 'Default'}", Colors.CYAN)
    print_colored(f"Encrypt: {encrypt}", Colors.CYAN)
    print()
    
    # Test the connection
    success = test_connection(host, port, user, password, schema, encrypt)
    
    if success:
        print_colored("\n✓ Connection test PASSED", Colors.GREEN + Colors.BOLD)
        return 0
    else:
        print_colored("\n✗ Connection test FAILED", Colors.RED + Colors.BOLD)
        return 1

if __name__ == "__main__":
    sys.exit(main())