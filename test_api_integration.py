#!/usr/bin/env python3
"""
Comprehensive API integration test script to verify all endpoints
and ensure frontend compatibility.
"""

import requests
import json
import sys
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
    
    def test_endpoint(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Test a specific endpoint and return results."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            return {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                "url": url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def run_all_tests(self) -> bool:
        """Run all API tests and return True if all pass."""
        tests = [
            ("GET", "/"),
            ("GET", "/health"),
            ("GET", "/supported-formats"),
            ("GET", "/docs"),
            ("GET", "/openapi.json"),
        ]
        
        print("🧪 Running API Integration Tests...\n")
        
        all_passed = True
        for method, endpoint in tests:
            print(f"Testing {method} {endpoint}...")
            result = self.test_endpoint(method, endpoint)
            
            if result["success"] and 200 <= result["status_code"] < 300:
                print(f"✅ PASS - {result['status_code']}")
                if endpoint == "/supported-formats":
                    self.validate_supported_formats(result["data"])
            else:
                print(f"❌ FAIL - {result.get('status_code', 'ERROR')}")
                if "error" in result:
                    print(f"   Error: {result['error']}")
                all_passed = False
            
            self.results.append({
                "endpoint": endpoint,
                "method": method,
                **result
            })
        
        return all_passed
    
    def validate_supported_formats(self, data: Dict[str, Any]) -> None:
        """Validate the supported formats response structure."""
        if not isinstance(data, dict):
            print("❌ Response is not a JSON object")
            return
        
        if "supportedFormats" not in data:
            print("❌ Missing 'supportedFormats' key")
            return
        
        formats = data["supportedFormats"]
        if not isinstance(formats, list):
            print("❌ supportedFormats is not an array")
            return
        
        if len(formats) == 0:
            print("⚠️ supportedFormats array is empty")
            return
        
        print(f"✅ Found {len(formats)} supported formats")
        for fmt in formats:
            print(f"   - {fmt}")
    
    def generate_report(self) -> str:
        """Generate a detailed test report."""
        report = {
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r.get("success") and 200 <= r.get("status_code", 0) < 300),
                "failed": len(self.results) - sum(1 for r in self.results if r.get("success") and 200 <= r.get("status_code", 0) < 300)
            },
            "results": self.results
        }
        
        return json.dumps(report, indent=2)

if __name__ == "__main__":
    # Check if custom URL provided
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    tester = APITester(base_url)
    all_passed = tester.run_all_tests()
    
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    if all_passed:
        print("🎉 All tests passed! The API is ready for frontend integration.")
    else:
        print("⚠️ Some tests failed. Check the detailed report above.")
    
    # Save report to file
    with open("api_test_report.json", "w") as f:
        f.write(tester.generate_report())
    
    print(f"\n📄 Detailed report saved to: api_test_report.json")
    
    sys.exit(0 if all_passed else 1)