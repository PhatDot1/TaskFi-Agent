#!/usr/bin/env python3
"""
Submit Proof Only - for testing agent response
Submits proof for an existing task
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from web3 import Web3
import json
from pathlib import Path

load_dotenv()

async def submit_proof_only():
    """Submit proof for an existing task"""
    
    if len(sys.argv) < 2:
        print("❌ Usage: python submit_proof_only.py <TASK_ID>")
        print("   Example: python submit_proof_only.py 5")
        return
        
    try:
        task_id = int(sys.argv[1])
    except ValueError:
        print("❌ Task ID must be a number")
        return
    
    print(f"📸 TaskFi - Submit Proof for Task {task_id}")
    print("="*60)
    
    # Setup Web3
    w3 = Web3(Web3.HTTPProvider("https://sepolia.infura.io/v3/f8a842101cf241c68380d2e9a14a2ab3"))
    
    if not w3.is_connected():
        print("❌ Failed to connect to Ethereum Sepolia")
        return
        
    print(f"✅ Connected to Ethereum Sepolia")
    
    # Load ABI
    abi_path = Path("abi/TaskFi.json")
    if abi_path.exists():
        with open(abi_path, 'r') as f:
            abi_data = json.load(f)
            abi = abi_data['abi'] if 'abi' in abi_data else abi_data
    else:
        print("❌ ABI file not found at abi/TaskFi.json")
        return
        
    # Setup contract
    contract_address = w3.to_checksum_address("0x559B8F2476C923A418114ABFD3704Abf88d43776")
    contract = w3.eth.contract(address=contract_address, abi=abi)
    
    # Setup account
    private_key = os.getenv("PRIVATE_KEY")
    if not private_key:
        print("❌ PRIVATE_KEY not found in environment variables")
        return
        
    if private_key.startswith('0x'):
        private_key = private_key[2:]
        
    account = w3.eth.account.from_key(private_key)
    print(f"🔑 Account: {account.address}")
    
    # Check task exists and status
    try:
        task_tuple = contract.functions.getTask(task_id).call()
        print(f"\n📋 Current Task Status:")
        print(f"🆔 Task ID: {task_id}")
        print(f"👤 Owner: {task_tuple[1]}")
        print(f"📝 Description: {task_tuple[2]}")
        print(f"📊 Status: {['InProgress', 'Complete', 'Failed'][task_tuple[5]]}")
        print(f"🔗 Current Proof: {task_tuple[6] if task_tuple[6] else 'None'}")
        
        # Check if we're the owner
        if task_tuple[1].lower() != account.address.lower():
            print(f"❌ You are not the owner of this task!")
            return
            
        # Check if task is in progress
        if task_tuple[5] != 0:  # 0 = InProgress
            print(f"❌ Task is not in progress!")
            return
            
    except Exception as e:
        print(f"❌ Task {task_id} not found or error: {e}")
        return
        
    # Submit proof
    proof_url = "https://gateway.pinata.cloud/ipfs/QmQRAK6oDejNqCGEQEPwtSQRHw3eL8bkUf93odRnuFAmji"
    print(f"\n🖼️ Submitting Proof: {proof_url}")
    
    try:
        # Build transaction
        function = contract.functions.submitProof(task_id, proof_url)
        gas_estimate = function.estimate_gas({'from': account.address})
        
        transaction = function.build_transaction({
            'from': account.address,
            'gas': gas_estimate,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account.address)
        })
        
        # Sign and send
        signed_txn = w3.eth.account.sign_transaction(transaction, account.key)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        print(f"🚀 Transaction sent: {tx_hash.hex()}")
        
        # Wait for confirmation
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        print(f"✅ Transaction confirmed in block: {receipt.blockNumber}")
        
        # Verify proof submission
        task_tuple = contract.functions.getTask(task_id).call()
        print(f"\n🎉 Proof submitted successfully!")
        print(f"🔗 Proof URL: {task_tuple[6]}")
        print(f"📊 Status: {['InProgress', 'Complete', 'Failed'][task_tuple[5]]}")
        
        print(f"\n🤖 Now your agent should process this task!")
        print(f"   The agent will:")
        print(f"   1. See task {task_id} has proof submitted")
        print(f"   2. Download and analyze the image")
        print(f"   3. Verify if it matches the task description")
        print(f"   4. Approve or reject the task")
        
    except Exception as e:
        print(f"❌ Error submitting proof: {e}")

if __name__ == "__main__":
    asyncio.run(submit_proof_only())