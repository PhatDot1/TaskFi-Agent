#!/usr/bin/env python3
"""
Submit Task Only - for testing the complete workflow
Creates task, then you can manually submit proof later to test agent
"""

import os
import asyncio
import time
from dotenv import load_dotenv
from web3 import Web3
import json
from pathlib import Path

load_dotenv()

async def submit_task_only():
    """Submit just the task, no proof"""
    print("🎨 TaskFi - Submit Task Only")
    print("Creating task: 'Create an art of an enlightened male being'")
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
    balance = w3.eth.get_balance(account.address)
    
    print(f"🔑 Account: {account.address}")
    print(f"💰 Balance: {w3.from_wei(balance, 'ether')} ETH")
    
    # Task details
    description = "Create an art of an enlightened male being"
    timeline_hours = 24  # 24 hours
    deposit = w3.to_wei(0.011, 'ether')  # 0.011 ETH
    
    print(f"\n📝 Task Description: {description}")
    print(f"⏰ Timeline: {timeline_hours} hours")
    print(f"💰 Deposit: {w3.from_wei(deposit, 'ether')} ETH")
    
    # Get current task count
    current_tasks = contract.functions.getCurrentTaskId().call()
    new_task_id = current_tasks
    print(f"📊 This will be task ID: {new_task_id}")
    
    try:
        # Build transaction
        function = contract.functions.submitTask(description, timeline_hours)
        gas_estimate = function.estimate_gas({
            'from': account.address,
            'value': deposit
        })
        
        transaction = function.build_transaction({
            'from': account.address,
            'gas': gas_estimate,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account.address),
            'value': deposit
        })
        
        # Sign and send
        signed_txn = w3.eth.account.sign_transaction(transaction, account.key)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        print(f"\n🚀 Transaction sent: {tx_hash.hex()}")
        
        # Wait for confirmation
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        print(f"✅ Transaction confirmed in block: {receipt.blockNumber}")
        
        # Verify task creation
        task_tuple = contract.functions.getTask(new_task_id).call()
        print(f"\n🎉 Task created successfully!")
        print(f"🆔 Task ID: {new_task_id}")
        print(f"📝 Description: {task_tuple[2]}")
        print(f"📊 Status: InProgress")
        print(f"🔗 Proof: None (ready for submission)")
        
        print(f"\n📋 Next Steps:")
        print(f"1. Use this command to submit proof:")
        print(f"   python submit_proof_only.py {new_task_id}")
        print(f"2. Or run the agent to see it waiting for proof:")
        print(f"   python taskfi_agent_fixed.py")
        
    except Exception as e:
        print(f"❌ Error submitting task: {e}")

if __name__ == "__main__":
    asyncio.run(submit_task_only())