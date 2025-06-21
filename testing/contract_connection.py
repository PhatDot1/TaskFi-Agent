#!/usr/bin/env python3
"""
Test script to verify contract connection and data parsing
"""

import os
import json
from dotenv import load_dotenv
from web3 import Web3
from pathlib import Path

load_dotenv()

def test_contract_connection():
    print("🧪 Testing TaskFi Contract Connection")
    print("=" * 50)
    
    # Setup Web3
    w3 = Web3(Web3.HTTPProvider("https://sepolia.infura.io/v3/f8a842101cf241c68380d2e9a14a2ab3"))
    
    if not w3.is_connected():
        print("❌ Failed to connect to Ethereum Sepolia")
        return
        
    print(f"✅ Connected to Ethereum Sepolia")
    print(f"📦 Latest block: {w3.eth.get_block('latest')['number']}")
    
    # Load ABI
    try:
        abi_path = Path("abi/TaskFi.json")
        if abi_path.exists():
            with open(abi_path, 'r') as f:
                abi_data = json.load(f)
                if 'abi' in abi_data:
                    abi = abi_data['abi']
                else:
                    abi = abi_data
            print("✅ ABI loaded from abi/TaskFi.json")
        else:
            print("❌ ABI file not found at abi/TaskFi.json")
            return
    except Exception as e:
        print(f"❌ Failed to load ABI: {e}")
        return
        
    # Setup contract
    contract_address = w3.to_checksum_address("0xBB28f99330B5fDffd96a1D1D5D6f94345B6e1229")
    contract = w3.eth.contract(address=contract_address, abi=abi)
    
    print(f"🏗️ Contract address: {contract_address}")
    
    try:
        # Test getCurrentTaskId
        total_tasks = contract.functions.getCurrentTaskId().call()
        print(f"📊 Total tasks: {total_tasks}")
        
        # Test getTask for each task
        for task_id in range(min(total_tasks, 3)):  # Test first 3 tasks
            try:
                print(f"\n🔍 Testing task {task_id}:")
                task_tuple = contract.functions.getTask(task_id).call()
                
                print(f"  📋 Task tuple length: {len(task_tuple)}")
                print(f"  🆔 Task ID: {task_tuple[0]}")
                print(f"  👤 User: {task_tuple[1]}")
                print(f"  📝 Description: {task_tuple[2][:50]}...")
                print(f"  💰 Deposit: {w3.from_wei(task_tuple[3], 'ether')} ETH")
                print(f"  ⏰ Deadline: {task_tuple[4]}")
                print(f"  📊 Status: {task_tuple[5]}")
                print(f"  🔗 Proof: {task_tuple[6][:50] if task_tuple[6] else 'None'}...")
                print(f"  🎨 NFT URI: {task_tuple[7][:50] if task_tuple[7] else 'None'}...")
                print(f"  🎯 NFT Token ID: {task_tuple[8]}")
                print(f"  📅 Created At: {task_tuple[9]}")
                
            except Exception as e:
                print(f"  ❌ Failed to fetch task {task_id}: {e}")
                
    except Exception as e:
        print(f"❌ Contract call failed: {e}")
        return
        
    print("\n✅ Contract connection test completed!")
    
    # Check environment
    print("\n🔧 Environment Check:")
    private_key = os.getenv("PRIVATE_KEY")
    if private_key:
        if private_key.startswith('0x'):
            private_key = private_key[2:]
        try:
            account = w3.eth.account.from_key(private_key)
            balance = w3.eth.get_balance(account.address)
            print(f"  🔑 Account: {account.address}")
            print(f"  💰 Balance: {w3.from_wei(balance, 'ether')} ETH")
        except Exception as e:
            print(f"  ❌ Invalid private key: {e}")
    else:
        print("  ⚠️ No private key configured")
        
    print(f"  🔗 Opik API Key: {'✅ Set' if os.getenv('OPIK_API_KEY') else '❌ Missing'}")
    print(f"  🤖 Jenius Token: {'✅ Set' if os.getenv('JENIUS_TOKEN') else '❌ Missing'}")

if __name__ == "__main__":
    test_contract_connection()