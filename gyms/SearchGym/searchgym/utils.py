import http.client
import json
import time
import os
from pathlib import Path
from urllib.parse import urlparse

def get_serper_api_key():
    """Get Serper API key from secret.json or environment variable."""
    # Try to load from secret.json in project root
    try:
        secret_path = Path(__file__).parent.parent / "secret.json"
        if secret_path.exists():
            with open(secret_path) as f:
                return json.load(f)["serper_key"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    
    # Try to load from current directory secret.json
    try:
        with open("secret.json") as f:
            return json.load(f)["serper_key"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    
    # Try environment variable
    api_key = os.getenv("SERPER_API_KEY")
    if api_key:
        return api_key
    
    raise ValueError("Serper API key not found. Please provide it in secret.json or SERPER_API_KEY environment variable.")

def search_serper(query, num=10, base_url=None, api_key=None):
    """
    Perform a search using Serper API.
    
    Args:
        query: Search query string
        num: Maximum number of results to return
        base_url: Custom base URL for Serper API (e.g., proxy URL). 
                  If None or empty, uses default "google.serper.dev"
        api_key: API key for authentication. If None, uses get_serper_api_key()
    
    Returns:
        Formatted search results string
    """
    search_start_time = time.time()
    print(f"[TIMING] [search_serper] Starting search for query: '{query[:50]}...' (truncated)")
    
    # Get API key if not provided
    if api_key is None:
        api_key = get_serper_api_key()
    
    # Handle mock API key for testing
    if api_key == "mock_api_key_for_testing":
        return f"Mock search results for '{query}':\n1. Mock Result 1\n- Snippet: This is a mock search result for testing purposes.\n2. Mock Result 2\n- Snippet: Another mock result to demonstrate the search functionality."
    
    # Determine connection parameters
    if base_url and base_url.strip():
        # Custom base URL provided (e.g., proxy)
        parsed_url = urlparse(base_url)
        hostname = parsed_url.hostname
        path = parsed_url.path or "/search"
        use_ssl = parsed_url.scheme == "https"
        
        # For custom proxy URLs, typically use Authorization header
        # For default Serper API, use X-API-KEY header
        if "google.serper.dev" not in hostname:
            # Custom proxy - use Bearer token
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
        else:
            # Default Serper API - use X-API-KEY
            headers = {
                'X-API-KEY': api_key,
                'Content-Type': 'application/json'
            }
    else:
        # Default Serper API
        hostname = "google.serper.dev"
        path = "/search"
        use_ssl = True
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
    
    payload = json.dumps({
        "q": query,
        "num": 15,
    })

    # Set connection timeout (10 seconds)
    timeout = 10
    conn = None
    
    try:
        try_time = 0
        max_retries = 3
        data = None
        
        while try_time <= max_retries:
            retry_start_time = time.time()
            print(f"[TIMING] [search_serper] Retry attempt {try_time + 1}/{max_retries + 1}")
            
            # Create a new connection for each retry to avoid stale connections
            conn_create_start = time.time()
            if use_ssl:
                conn = http.client.HTTPSConnection(hostname, timeout=timeout)
            else:
                conn = http.client.HTTPConnection(hostname, timeout=timeout)
            conn_create_time = time.time() - conn_create_start
            if conn_create_time > 1:
                print(f"[TIMING] [search_serper] Connection creation took {conn_create_time:.2f}s")
            
            try:
                request_start = time.time()
                conn.request("POST", path, payload, headers)
                request_time = time.time() - request_start
                print(f"[TIMING] [search_serper] HTTP request sent in {request_time:.2f}s")
                
                response_start = time.time()
                res = conn.getresponse()
                response_wait_time = time.time() - response_start
                print(f"[TIMING] [search_serper] Waiting for HTTP response took {response_wait_time:.2f}s")
                if response_wait_time > 5:
                    print(f"[WARNING] [search_serper] HTTP response wait time is slow: {response_wait_time:.2f}s")
                if response_wait_time > 30:
                    print(f"[ERROR] [search_serper] HTTP response wait time is very slow: {response_wait_time:.2f}s - THIS MAY BE THE BOTTLENECK!")
                
                # Check status code
                if res.status != 200:
                    error_msg = f"HTTP {res.status}: {res.reason}"
                    conn.close()
                    retry_total_time = time.time() - retry_start_time
                    print(f"[TIMING] [search_serper] Retry attempt {try_time + 1} failed after {retry_total_time:.2f}s: {error_msg}")
                    if try_time >= max_retries:
                        return f"Search Error: {error_msg}"
                    try_time += 1
                    time.sleep(2)
                    continue
                
                read_start = time.time()
                data = res.read()
                read_time = time.time() - read_start
                print(f"[TIMING] [search_serper] Reading response body took {read_time:.2f}s")
                
                decode_start = time.time()
                data = data.decode("utf-8")
                data = json.loads(data)
                decode_time = time.time() - decode_start
                print(f"[TIMING] [search_serper] Decoding and parsing JSON took {decode_time:.2f}s")
                conn.close()
                
                retry_total_time = time.time() - retry_start_time
                print(f"[TIMING] [search_serper] Retry attempt {try_time + 1} total time: {retry_total_time:.2f}s")
                
                # Break if we have results (either organic or answerBox)
                if data.get("organic", []) or data.get("answerBox"):
                    total_search_time = time.time() - search_start_time
                    print(f"[TIMING] [search_serper] Search completed successfully in {total_search_time:.2f}s")
                    break
                
                # If no results after retries, return empty results
                if try_time >= max_retries:
                    return f"Search results for '{query}': No results found."
                
                try_time += 1
                time.sleep(2)
                
            except (http.client.HTTPException, json.JSONDecodeError, Exception) as e:
                conn.close()
                retry_total_time = time.time() - retry_start_time
                print(f"[TIMING] [search_serper] Retry attempt {try_time + 1} exception after {retry_total_time:.2f}s: {str(e)}")
                if try_time >= max_retries:
                    return f"Search Error: {str(e)}"
                try_time += 1
                time.sleep(2)
        
        if data is None:
            return "Search Error: Failed to get response after retries"
        
        # Process search results
        output = ""
        index = 1
        answer_box = data.get("answerBox", {}) or {}
        if answer_box and isinstance(answer_box, dict) and 'title' in answer_box:
            if "answer" in answer_box:
                output += f"{str(index)}. {answer_box['title']}\n- Answer: {answer_box['answer']}\n"
                index += 1
            elif 'snippet' in answer_box:
                output += f"{str(index)}. {answer_box['title']}\n- Snippet: {answer_box['snippet']}\n"
                index += 1
            
        
        if index > num:
            return output.strip()
        
        for item in data.get("organic", []):
            if 'title' in item and 'snippet' in item:
                output += f"{str(index)}. {item['title']}\n- Snippet: {item['snippet']}\n"
                index += 1
            if index > num:
                return output.strip()
        
        total_search_time = time.time() - search_start_time
        print(f"[TIMING] [search_serper] Total search time (including processing): {total_search_time:.2f}s")
        return output.strip()
    
    except Exception as e:
        total_search_time = time.time() - search_start_time
        error = f"Search Error: {e}"
        print(f"[TIMING] [search_serper] Search failed after {total_search_time:.2f}s: {error}")
        return error
    finally:
        # Ensure connection is closed (it may already be closed in the loop)
        if conn is not None:
            try:
                conn.close()
            except:
                pass

if __name__ == "__main__":
    query = "How's the weather in Beijing"
    # Try to use SERPER_BASE_URL from environment if available
    base_url = os.getenv("SERPER_BASE_URL", None)
    print(search_serper(query, num=3, base_url=base_url))
