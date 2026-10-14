"""
Error code constants for OptiMAS Server.

Using JSON-RPC standard error codes with descriptive variable names.

Error code meanings:
- PARSE_ERROR: Invalid JSON was received
- INVALID_REQUEST: The JSON sent is not a valid Request object
- METHOD_NOT_FOUND: The method does not exist / is not available
- INVALID_PARAMS: Invalid method parameter(s)
- INTERNAL_ERROR: Internal JSON-RPC error

Custom error codes (-32000 to -32099 are reserved for server-defined errors):
- AUTH_MISSING: Authentication credentials are missing
- AUTH_INVALID: Authentication credentials are invalid
"""
# JSON-RPC Standard Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Custom Server Error Codes (Authentication)
AUTH_MISSING = -32000
AUTH_INVALID = -32001

