#!/usr/bin/env python3
"""
Simple test script to verify OpenAI integration
"""

import os

# Reads from the OPENAI_API_KEY environment variable - never hardcode a real key here.
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

try:
    print("🔧 Testing OpenAI API connection...")

    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY environment variable is not set. "
                         "Set it first, e.g. export OPENAI_API_KEY=sk-...")

    # Check OpenAI version first
    import openai
    print(f"📦 OpenAI library version: {openai.__version__}")
    
    # Try new client initialization
    try:
        from openai import OpenAI
        print("🔄 Trying new OpenAI client (v1.0+)...")
        
        # Try with clean initialization first
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as init_error:
            if 'proxies' in str(init_error).lower():
                print("🔧 Proxies error detected, trying environment variable method...")
                # Clear proxy environment variables
                proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
                for var in proxy_vars:
                    if var in os.environ:
                        del os.environ[var]
                client = OpenAI()  # Use environment variable only
            else:
                raise init_error
        
        # Simple test
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hello, can you help with medical questions?"}
            ],
            max_tokens=50
        )
        
        print("✅ New OpenAI client test successful!")
        print(f"📥 Response: {response.choices[0].message.content}")
        
    except Exception as e1:
        print(f"❌ New OpenAI client failed: {e1}")
        raise Exception(f"OpenAI client failed: {e1}")
    
    print("\n🎉 Your chatbot is ready to use!")
    
except Exception as e:
    print(f"❌ OpenAI API test failed: {e}")
    print("🔄 The app will use deterministic educational FAQ instead")
    print("💡 The local chatbot uses a deterministic educational FAQ when no API key is configured.")
    print("\n🛠️ To fix OpenAI issues:")
    print("1. Check your API key is valid")
    print("2. Ensure you have credits in your OpenAI account")
    print("3. Try: pip install --upgrade openai==1.3.7")
    print("4. Check your internet connection")