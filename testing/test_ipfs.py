#!/usr/bin/env python3
"""
Test IPFS Image Access
Tests multiple IPFS gateways to see which ones work
"""

import requests
from PIL import Image
import io

def test_ipfs_access():
    """Test accessing the enlightened being image from multiple IPFS gateways"""
    
    ipfs_hash = "QmQRAK6oDejNqCGEQEPwtSQRHw3eL8bkUf93odRnuFAmji"
    
    gateways = [
        f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}",
        f"https://ipfs.io/ipfs/{ipfs_hash}",
        f"https://cloudflare-ipfs.com/ipfs/{ipfs_hash}",
        f"https://dweb.link/ipfs/{ipfs_hash}",
        f"https://gateway.ipfs.io/ipfs/{ipfs_hash}",
        f"https://ipfs.infura.io/ipfs/{ipfs_hash}",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'image/*,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    print("🧪 Testing IPFS Gateway Access")
    print("=" * 60)
    
    working_gateways = []
    
    for i, url in enumerate(gateways):
        try:
            print(f"\n{i+1}. Testing: {url}")
            
            response = requests.get(url, timeout=10, headers=headers, stream=True)
            
            print(f"   Status: {response.status_code}")
            print(f"   Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
            print(f"   Content-Length: {response.headers.get('Content-Length', 'Unknown')}")
            
            if response.status_code == 200:
                # Try to load as image
                try:
                    image = Image.open(response.raw)
                    print(f"   ✅ SUCCESS - Image loaded: {image.size} {image.mode}")
                    working_gateways.append(url)
                except Exception as e:
                    print(f"   ❌ FAILED - Not a valid image: {e}")
            else:
                print(f"   ❌ FAILED - HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError as e:
            print(f"   ❌ FAILED - Connection error: {e}")
        except requests.exceptions.Timeout as e:
            print(f"   ❌ FAILED - Timeout: {e}")
        except Exception as e:
            print(f"   ❌ FAILED - Error: {e}")
    
    print("\n" + "=" * 60)
    print("📊 RESULTS:")
    print(f"✅ Working gateways: {len(working_gateways)}")
    print(f"❌ Failed gateways: {len(gateways) - len(working_gateways)}")
    
    if working_gateways:
        print(f"\n🎉 Best gateway to use: {working_gateways[0]}")
        
        # Test BLIP captioning with working gateway
        try:
            print(f"\n🤖 Testing BLIP captioning...")
            
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            response = requests.get(working_gateways[0], timeout=10, headers=headers, stream=True)
            image = Image.open(response.raw).convert("RGB")
            
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50, max_new_tokens=30)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            print(f"📝 BLIP Caption: {caption}")
            print(f"✅ BLIP analysis successful!")
            
        except ImportError:
            print("⚠️ BLIP not available - install transformers and torch to test")
        except Exception as e:
            print(f"❌ BLIP test failed: {e}")
    else:
        print(f"\n❌ No working gateways found!")
        print("💡 This might be a network/DNS issue on your machine")

if __name__ == "__main__":
    test_ipfs_access()