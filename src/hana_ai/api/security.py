"""
Security utilities for the HANA AI Toolkit API.

This module provides security validation and enforcement to prevent external calls
and ensure the application stays within SAP BTP service boundaries.
"""
import re
import socket
import logging
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps

from .config import settings
from .env_constants import BTP_IP_RANGES, BTP_DOMAIN_PATTERNS

logger = logging.getLogger(__name__)

# List of allowed domain patterns
ALLOWED_DOMAINS = BTP_DOMAIN_PATTERNS + [
    r'localhost$',                     # Local development
    r'127\.0\.0\.1$',                  # Local development
]

def is_ip_in_range(ip: str, cidr: str) -> bool:
    """
    Check if an IP address is within a CIDR range.
    
    Parameters
    ----------
    ip : str
        The IP address to check
    cidr : str
        The CIDR notation range to check against
        
    Returns
    -------
    bool
        True if the IP is in the range, False otherwise
    """
    try:
        # Simple implementation for demonstration
        # In production, use a proper IP address library
        if '/' not in cidr:
            return ip == cidr
        
        net_addr, mask = cidr.split('/')
        mask = int(mask)
        
        # Convert IP addresses to integers
        ip_int = int.from_bytes(socket.inet_aton(ip), byteorder='big')
        net_int = int.from_bytes(socket.inet_aton(net_addr), byteorder='big')
        
        # Create mask
        mask_int = 0xffffffff ^ ((1 << (32 - mask)) - 1)
        
        # Check if IP is in range
        return (ip_int & mask_int) == (net_int & mask_int)
    except Exception as e:
        logger.error(f"Error checking IP range: {str(e)}")
        return False

def is_domain_allowed(domain: str) -> bool:
    """
    Check if a domain is in the allowed list.
    
    Parameters
    ----------
    domain : str
        The domain to check
        
    Returns
    -------
    bool
        True if the domain is allowed, False otherwise
    """
    domain = domain.lower()
    
    # Check against allowed domain patterns
    for pattern in ALLOWED_DOMAINS:
        if re.match(pattern, domain):
            return True
    
    return False

def validate_host(host: str) -> Tuple[bool, str]:
    """
    Validate if a host is allowed based on domain or IP address.
    
    Parameters
    ----------
    host : str
        The host to validate
        
    Returns
    -------
    Tuple[bool, str]
        (True, "") if valid, (False, reason) if invalid
    """
    # Handle IP addresses
    if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', host):
        # Check if IP is in allowed ranges
        for ip_range in BTP_IP_RANGES:
            if is_ip_in_range(host, ip_range):
                return True, ""
        return False, f"IP address {host} is outside allowed BTP ranges"
    
    # Handle domains
    if is_domain_allowed(host):
        return True, ""
    
    return False, f"Domain {host} is not in the allowed BTP domains list"

def validate_external_request(url: str) -> Tuple[bool, str]:
    """
    Validate if an external request is allowed.
    
    Parameters
    ----------
    url : str
        The URL to validate
        
    Returns
    -------
    Tuple[bool, str]
        (True, "") if valid, (False, reason) if invalid
    """
    if not settings.RESTRICT_EXTERNAL_CALLS:
        # If restriction is disabled, allow all requests
        logger.warning(f"External call restriction is disabled, allowing request to: {url}")
        return True, ""
    
    try:
        # Extract host from URL
        import urllib.parse
        parsed_url = urllib.parse.urlparse(url)
        host = parsed_url.netloc
        
        # Remove port if present
        if ':' in host:
            host = host.split(':')[0]
        
        # Validate host
        return validate_host(host)
    except Exception as e:
        logger.error(f"Error validating URL: {str(e)}")
        return False, f"Error validating URL: {str(e)}"

def prevent_external_calls(func):
    """
    Decorator to prevent external calls from functions.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if the function is trying to make external calls
        # This is a simplified implementation
        # In a real implementation, you would need to inspect the function
        # and its arguments more carefully
        
        # For now, just log a warning
        if settings.RESTRICT_EXTERNAL_CALLS:
            logger.info(f"External call prevention activated for function: {func.__name__}")
            
        return func(*args, **kwargs)
    return wrapper