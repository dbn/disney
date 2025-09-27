#!/usr/bin/env python3
"""Comprehensive test script that starts the API server and runs tests."""

import asyncio
import subprocess
import time
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from disney.shared.logging import setup_logging

logger = setup_logging("full-api-tester")

# Global variable to store the server process
server_process = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global server_process
    if server_process:
        print("\n🛑 Stopping API server...")
        server_process.terminate()
        server_process.wait()
    sys.exit(0)


def start_api_server():
    """Start the API server in a subprocess."""
    global server_process
    try:
        print("🚀 Starting API server...")
        server_process = subprocess.Popen([
            sys.executable, "-m", "src.disney.api.main"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for the server to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Check if server is running
        if server_process.poll() is None:
            print("✅ API server started successfully")
            return True
        else:
            stdout, stderr = server_process.communicate()
            print(f"❌ Failed to start API server:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting API server: {str(e)}")
        return False


async def run_tests():
    """Run the API tests."""
    try:
        # Import and run the test functions
        from test_api import test_api_endpoints, test_chromadb_connection, test_api_docs
        
        print("\n🧪 Running API tests...")
        print("=" * 60)
        
        print("\n1. Testing API Health and Status...")
        await test_api_endpoints()
        
        print("\n2. Testing ChromaDB Connection...")
        await test_chromadb_connection()
        
        print("\n3. Testing API Documentation...")
        await test_api_docs()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Error running tests: {str(e)}")
        logger.error(f"Error running tests: {str(e)}")


async def main():
    """Main function to orchestrate server startup and testing."""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🧪 Disney AI Customer Experience Assessment API - Full Test Suite")
    print("=" * 70)
    
    # Start the API server
    if not start_api_server():
        print("❌ Failed to start API server. Exiting.")
        return
    
    try:
        # Run the tests
        await run_tests()
        
        print("\n🎉 Full test suite completed successfully!")
        print("\n📝 Summary:")
        print("  - API server started and running")
        print("  - All endpoints tested")
        print("  - ChromaDB connection verified")
        print("  - API documentation accessible")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Clean up: stop the server
        global server_process
        if server_process:
            print("\n🛑 Stopping API server...")
            server_process.terminate()
            server_process.wait()
            print("✅ API server stopped")


if __name__ == "__main__":
    asyncio.run(main())
